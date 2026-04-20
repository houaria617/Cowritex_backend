"""
orchestrator/nodes/writing_node.py
Calls run_writing_agent (real implementation) and persists the suggestion.
The @traced decorator makes this appear as a child span in LangSmith.
"""

from __future__ import annotations
import logging
from langsmith import traceable
from ..state import GraphState
from database import repository as repo
from writing_agent.agent import run_writing_agent

logger = logging.getLogger(__name__)

# Wrap the external agent call so it shows up in LangSmith traces
_traced_writing = traceable(name="writing_agent_call")(run_writing_agent)


def writing_node(state: GraphState) -> dict:
    section_id = state.get("section_id")
    instruction = state.get("instruction") or state["user_message"]
    prefs = state.get("preferences", {})
    project_id = state["project_id"]

    document = ""
    if section_id:
        try:
            document = repo.get_current_content(section_id) or ""
        except Exception as exc:
            logger.warning("Could not load current content: %s", exc)

    context = {
        "writing_style":  prefs.get("writing_style", "formal"),
        "tone":           prefs.get("tone", "academic"),
        "target_journal": prefs.get("target_journal", ""),
        "citation_style": prefs.get("citation_style", "APA"),
        "language":       prefs.get("language", "English"),
        "grounded_only":  prefs.get("grounded_only", False),
        "sources":        [],   # orchestrator can inject sources here if needed
    }

    # Inject HITL feedback from a reject/regenerate round
    if state.get("hitl_feedback"):
        instruction = f"{instruction}\n\nFeedback: {state['hitl_feedback']}"

    try:
        output_text = _traced_writing(
            document=document,
            instruction=instruction,
            context=context,
        )
    except Exception as exc:
        logger.error("Writing agent failed: %s", exc)
        return {"error": f"Writing agent error: {exc}", "agent_output": None}

    if output_text.startswith("[ERROR]"):
        return {"error": output_text, "agent_output": None}

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

    return {
        "agent_output":  output_text,
        "last_agent":    "writing",
        "error":         None,
        "hitl_action":   None,
        "hitl_feedback": None,
    }
