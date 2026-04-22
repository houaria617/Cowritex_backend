"""
orchestrator/nodes/writing_node.py
Calls run_writing_agent (real implementation).
"""

from __future__ import annotations
import logging
from langsmith import traceable
from ..state import GraphState
from database import repository as repo
from writing_agent.agent import run_writing_agent

logger = logging.getLogger(__name__)

_traced_writing = traceable(name="writing_agent_call")(run_writing_agent)


def writing_node(state: GraphState) -> dict:
    section_id = state.get("section_id")
    instruction = state.get("instruction") or state["user_message"]
    prefs = state.get("preferences", {})
    project_id = state["project_id"]

    # ── Load current section content ──
    document = ""
    if section_id:
        try:
            document = repo.get_current_content(section_id) or ""
        except Exception as exc:
            logger.warning("Could not load current content: %s", exc)

    # ── Load surrounding sections for redundancy avoidance ──
    preceding_sections = {}
    next_sections = {}
    try:
        all_sections = repo.get_project_sections(project_id) or []
        current_idx = next(
            (i for i, s in enumerate(all_sections)
             if s.get("section_id") == section_id),
            None
        )
        if current_idx is not None:
            if current_idx > 0:
                prev = all_sections[current_idx - 1]
                preceding_sections[prev.get("name", "previous")] = \
                    prev.get("content", "")
            if current_idx < len(all_sections) - 1:
                nxt = all_sections[current_idx + 1]
                next_sections[nxt.get("name", "next")] = \
                    nxt.get("content", "")
    except Exception as exc:
        logger.warning("Could not load surrounding sections: %s", exc)

    # ── Build context dict ──
    context = {
        "writing_style":  prefs.get("writing_style", "academic"),
        "tone":           prefs.get("tone", "formal"),
        "target_journal": prefs.get("target_journal", ""),
        "citation_style": prefs.get("citation_style", "APA"),
        "language":       prefs.get("language", "English"),
        "grounded_only":  prefs.get("grounded_only", False),
        "sources":        [],
    }

    # ── Inject HITL feedback for regenerate rounds ──
    if state.get("hitl_feedback"):
        instruction = f"{instruction}\n\nFeedback: {state['hitl_feedback']}"

    # ── Call agent ──
    # operation is intentionally NOT passed — auto-detected from instruction
    try:
        output_text = _traced_writing(
            document=document,
            instruction=instruction,
            context=context,
            target_section=section_id,          # agent normalises it internally
            preceding_sections=preceding_sections,
            next_sections=next_sections,
        )
    except Exception as exc:
        logger.error("Writing agent failed: %s", exc)
        return {"error": f"Writing agent error: {exc}", "agent_output": None}

    if output_text.startswith("[ERROR]"):
        return {"error": output_text, "agent_output": None}

    # ── Persist suggestion to DB ──
    if section_id:
        try:
            repo.create_suggestion(
                section_id=section_id,
                suggested_text=output_text,
                instruction=instruction,
                original_text=document or None,
            )
        except Exception as exc:
            logger.warning("Could not save suggestion: %s", exc)

    # ── Update agent_outputs for compound intent support ──
    existing = state.get("agent_outputs", {})

    return {
        "agent_output":  output_text,
        "agent_outputs": {**existing, "writing": output_text},
        "last_agent":    "writing",
        "error":         None,
        "hitl_action":   None,
        "hitl_feedback": None,
    }
