"""
orchestrator/nodes/literature_node.py
──────────────────────────────────────
Calls LiteratureAgent directly — avoids file I/O round-trip.

If search_results are already in state (populated by search_node),
their PDF/abstract URLs are extracted and passed as web_urls so the
LiteratureAgent can fetch and chunk them via process_web_resources.

This is the only interface the unmodified LiteratureAgent exposes for
injecting external content — AgentInput.use_web_resources + web_urls.
"""

from __future__ import annotations
import logging
from langsmith import traceable

from ..state import GraphState
from database import repository as repo
from literature_agent.agent import LiteratureAgent
from literature_agent.config import AgentInput

logger = logging.getLogger(__name__)

# Maximum number of paper URLs to feed into the agent.
# process_web_resources fetches each URL synchronously, so keep this
# bounded to avoid multi-minute runtimes.
_MAX_WEB_URLS = 8


def _extract_urls_from_papers(papers: list[dict]) -> list[str]:
    """
    Pull the best available URL out of each search-result paper dict.

    Priority order:
      1. pdf_link  — direct PDF (highest quality for chunking)
      2. url       — abstract / landing page (fallback)

    Only HTTP(S) URLs are included; None / empty strings are skipped.
    Deduplication preserves order.
    """
    seen: set[str] = set()
    urls: list[str] = []

    for paper in papers:
        for key in ("pdf_link", "url"):
            candidate = (paper.get(key) or "").strip()
            if candidate.startswith("http") and candidate not in seen:
                seen.add(candidate)
                urls.append(candidate)
                break               # one URL per paper is enough

    return urls[:_MAX_WEB_URLS]


def _run_agent(input_data: AgentInput):
    agent = LiteratureAgent()
    return agent.generate_review(input_data)


_traced_literature = traceable(name="literature_agent_call")(_run_agent)


def literature_node(state: GraphState) -> dict:
    project_id = state["project_id"]
    prefs = state.get("preferences", {})
    instruction = state.get("instruction") or state["user_message"]

    # ── Build query (append HITL feedback when regenerating) ─────────────────
    query = instruction
    if state.get("hitl_feedback"):
        query = f"{query}\n\nFeedback: {state['hitl_feedback']}"

    citation_style = prefs.get("citation_style", "APA")
    grounded_only = prefs.get("grounded_only", False)

    # ── Derive web_urls from search_results ───────────────────────────────────
    # search_node populates state["search_results"] as List[dict] with keys:
    #   title, authors, abstract, pdf_link, url, year, citations, source
    #
    # The unmodified LiteratureAgent has NO pre_fetched_papers parameter.
    # Its only external-content hook is:
    #   AgentInput(use_web_resources=True, web_urls=[...])
    # which routes through load_documents → process_web_resources.
    #
    # So we convert paper records → URLs here in the orchestrator layer.
    pre_fetched_papers = state.get("search_results") or []
    web_urls: list[str] = []

    if pre_fetched_papers and not grounded_only:
        web_urls = _extract_urls_from_papers(pre_fetched_papers)
        logger.info(
            "literature_node: extracted %d web_urls from %d search_results",
            len(web_urls), len(pre_fetched_papers),
        )

    # use_web_resources must be True whenever we have URLs to fetch,
    # regardless of the grounded_only flag (grounded_only only suppresses
    # the upstream search_node — if we already have results we use them).
    use_web_resources = bool(web_urls) or (not grounded_only)

    input_data = AgentInput(
        query=query,
        citation_style=citation_style,
        use_web_resources=use_web_resources,
        use_ocr=False,
        web_urls=web_urls,          # ← the real injection point
    )

    logger.info(
        "literature_node: calling agent | use_web_resources=%s | %d urls | query=%r",
        use_web_resources, len(web_urls), query[:80],
    )

    try:
        output = _traced_literature(input_data)
    except Exception as exc:
        logger.error("Literature agent failed: %s", exc)
        return {"error": f"Literature agent error: {exc}", "agent_output": None}

    if not output.success:
        logger.error(
            "Literature agent returned failure: %s", output.error_message
        )
        return {"error": output.error_message, "agent_output": None}

    # ── Persist to DB ─────────────────────────────────────────────────────────
    try:
        analysis_id = repo.save_literature_analysis(
            project_id=project_id,
            review_content=output.literature_review,
            gaps_content="",
            warnings="",
        )
        if output.retrieved_sources:
            source_ids = repo.save_sources(project_id, [
                {
                    "title":           s.get("file", ""),
                    "authors":         s.get("author", ""),
                    "abstract":        "",
                    "url":             "",
                    "doi":             "",
                    "citation_count":  0,
                    "relevance_score": float(s.get("score", 0.0)),
                    "pdf_url":         "",
                }
                for s in output.retrieved_sources
            ])
            repo.save_literature_citations(
                analysis_id=analysis_id,
                source_ids=source_ids,
                formatted_citations=[
                    f"{s.get('author', '')} ({s.get('year', '')}). "
                    f"{s.get('file', '')}, p.{s.get('page', '')}"
                    for s in output.retrieved_sources
                ],
                citation_style=citation_style,
            )
    except Exception as exc:
        logger.warning("DB persistence for literature failed: %s", exc)

    # ── Verification warning ──────────────────────────────────────────────────
    verification = output.verification_report or {}
    unverified_count = len(verification.get("unverified_claims", []))
    overall = verification.get("overall_status", "")
    warning = ""
    if unverified_count > 3:
        warning = (
            f"\n\n⚠️ **Verification: {overall}** — "
            f"{unverified_count} claim(s) could not be grounded in source documents."
        )

    # ── Debug (keep until stable) ─────────────────────────────────────────────
    print(f"\n🔍 output.success       = {output.success}")
    print(f"🔍 output.error_message = {repr(output.error_message)}")
    print(f"🔍 web_urls fed         = {web_urls}")
    print(
        f"🔍 output.literature_review length  = {len(output.literature_review or '')}")
    print(
        f"🔍 output.literature_review preview = {repr((output.literature_review or '')[:200])}")

    existing = state.get("agent_outputs", {})

    return {
        "agent_output":  output.literature_review + warning,
        "agent_outputs": {**existing, "literature": output.literature_review + warning},
        "last_agent":    "literature",
        "error":         None,
        "hitl_action":   None,
        "hitl_feedback": None,
    }
