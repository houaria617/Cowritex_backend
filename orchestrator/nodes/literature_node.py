"""
orchestrator/nodes/literature_node.py
──────────────────────────────────────
Calls LiteratureAgent directly.

Web resource handling is now entirely delegated to the literature agent:
  • grounded_only=True  → agent uses ChromaDB / local PDFs only
  • grounded_only=False → agent fetches web resources itself

The orchestrator no longer extracts URLs from search_results or
passes web_urls — that logic lived here only to compensate for the
old search_node detour, which has been removed.
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

    # ── Append HITL feedback when regenerating ────────────────────────────────
    query = instruction
    if state.get("hitl_feedback"):
        query = f"{query}\n\nFeedback: {state['hitl_feedback']}"

    citation_style = prefs.get("citation_style", "APA")
    grounded_only = prefs.get("grounded_only", False)

    # ── Delegate entirely to the literature agent ─────────────────────────────
    # grounded_only=True  → use_web_resources=False → ChromaDB / local PDFs only
    # grounded_only=False → use_web_resources=True  → agent fetches web resources
    # web_urls is always empty — the agent resolves its own sources internally
    input_data = AgentInput(
        query=query,
        citation_style=citation_style,
        use_web_resources=not grounded_only,
        use_ocr=False,
        web_urls=[],
    )

    logger.info(
        "literature_node: calling agent | use_web_resources=%s | query=%r",
        not grounded_only, query[:80],
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

    existing = state.get("agent_outputs", {})

    return {
        "agent_output":  output.literature_review + warning,
        "agent_outputs": {**existing, "literature": output.literature_review + warning},
        "last_agent":    "literature",
        "error":         None,
        "hitl_action":   None,
        "hitl_feedback": None,
    }
