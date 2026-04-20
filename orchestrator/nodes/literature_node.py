"""
orchestrator/nodes/literature_node.py
Calls run_literature_agent (real implementation) and persists results.
"""

from __future__ import annotations
import logging
import os
from langsmith import traceable
from ..state import GraphState
from database import repository as repo
from literature_agent.agent import run_literature_agent as _raw_lit_agent

logger = logging.getLogger(__name__)

_traced_literature = traceable(name="literature_agent_call")(_raw_lit_agent)


def _read_output_files(project_title: str) -> tuple[str, str]:
    """Read the .txt files the literature agent writes to disk."""
    base = os.path.join("outputs")
    art_path = os.path.join(base, f"{project_title}_state_of_the_art.txt")
    cit_path = os.path.join(base, "citations.txt")

    review = ""
    citations_text = ""
    try:
        with open(art_path, encoding="utf-8") as f:
            review = f.read()
    except FileNotFoundError:
        logger.warning("State-of-the-art file not found: %s", art_path)

    try:
        with open(cit_path, encoding="utf-8") as f:
            citations_text = f.read()
    except FileNotFoundError:
        logger.warning("Citations file not found: %s", cit_path)

    return review, citations_text


def literature_node(state: GraphState) -> dict:
    project_id = state["project_id"]
    prefs = state.get("preferences", {})

    project_title = ""
    try:
        project = repo.get_project(project_id)
        project_title = project.get("title", state["user_message"])
    except Exception as exc:
        logger.warning("Could not fetch project title: %s", exc)
        project_title = state["user_message"]

    citation_style = prefs.get("citation_style", "APA")
    use_web_resources = not prefs.get("grounded_only", False)

    try:
        # run_literature_agent writes files to disk and returns None
        _traced_literature(
            project_title=project_title,
            citation_style=citation_style,
            use_web_resources=use_web_resources,
            save_to_chromadb=False,
        )
    except Exception as exc:
        logger.error("Literature agent failed: %s", exc)
        return {"error": f"Literature agent error: {exc}", "agent_output": None}

    # Read what the agent wrote
    review, citations_raw = _read_output_files(project_title)

    # ── Persist to DB ──────────────────────────────────────────
    warnings_text = ""
    try:
        analysis_id = repo.save_literature_analysis(
            project_id=project_id,
            review_content=review,
            gaps_content="",          # gaps are embedded in the review markdown
            warnings=warnings_text,
        )
        # Parse citation lines into source rows
        citation_lines = [l.strip()
                          for l in citations_raw.splitlines() if l.strip()]
        if citation_lines:
            source_ids = repo.save_sources(project_id, [
                {"title": line, "authors": "", "abstract": "", "url": "",
                 "doi": "", "citation_count": 0, "relevance_score": 0.0, "pdf_url": ""}
                for line in citation_lines
            ])
            repo.save_literature_citations(
                analysis_id=analysis_id,
                source_ids=source_ids,
                formatted_citations=citation_lines,
                citation_style=citation_style,
            )
    except Exception as exc:
        logger.warning("DB persistence for literature failed: %s", exc)

    output_text = review or "Literature review generated — see outputs folder."

    return {
        "agent_output":  output_text,
        "last_agent":    "literature",
        "error":         None,
        "hitl_action":   None,
        "hitl_feedback": None,
    }
