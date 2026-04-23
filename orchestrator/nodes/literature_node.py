"""
orchestrator/nodes/literature_node.py
──────────────────────────────────────
Calls LiteratureAgent directly — avoids file I/O round-trip.

If search_results are already in state (populated by search_node),
they are injected into the AgentInput so the literature agent can
synthesize from freshly retrieved web papers instead of ChromaDB alone.
"""

from __future__ import annotations
import logging
from langsmith import traceable

from ..state import GraphState
from database import repository as repo
from literature_agent.agent import LiteratureAgent
from literature_agent.config import AgentInput

logger = logging.getLogger(__name__)


def _run_agent(input_data: AgentInput):
    agent = LiteratureAgent()
    return agent.generate_review(input_data)


_traced_literature = traceable(name="literature_agent_call")(_run_agent)


def literature_node(state: GraphState) -> dict:
    project_id = state["project_id"]
    prefs = state.get("preferences", {})
    instruction = state.get("instruction") or state["user_message"]

    # ── Use instruction as the primary query ─────────────────────────────────
    query = instruction
    if state.get("hitl_feedback"):
        query = f"{query}\n\nFeedback: {state['hitl_feedback']}"

    citation_style = prefs.get("citation_style", "APA")
    use_web_resources = not prefs.get("grounded_only", False)

    # ── Inject search_results from search_node when available ─────────────────
    # search_results is List[dict] with keys: title, authors, abstract, pdf_link, url
    pre_fetched_papers = state.get("search_results") or []
    if pre_fetched_papers:
        logger.info(
            "literature_node: consuming %d pre-fetched papers from search_node",
            len(pre_fetched_papers)
        )

    input_data = AgentInput(
        query=query,
        citation_style=citation_style,
        use_web_resources=use_web_resources,
        use_ocr=False,
        web_urls=[],
        # Pass pre-fetched papers if your AgentInput supports it;
        # add this field to AgentInput if not yet present.
        pre_fetched_papers=pre_fetched_papers if pre_fetched_papers else None,
    )

    try:
        output = _traced_literature(input_data)
    except Exception as exc:
        logger.error("Literature agent failed: %s", exc)
        return {"error": f"Literature agent error: {exc}", "agent_output": None}

    if not output.success:
        logger.error("Literature agent returned failure: %s",
                     output.error_message)
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

    # ── Debug ─────────────────────────────────────────────────────────────────
    print(f"\n🔍 output.success       = {output.success}")
    print(f"🔍 output.error_message = {repr(output.error_message)}")
    print(
        f"🔍 output.literature_review length = {len(output.literature_review or '')}")
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
