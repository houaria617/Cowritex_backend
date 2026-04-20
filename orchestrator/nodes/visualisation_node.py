"""
orchestrator/nodes/visualisation_node.py
"""
from __future__ import annotations
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..state import GraphState
from ._llm import get_llm
from database import repository as repo

logger = logging.getLogger(__name__)

_VIZ_SYSTEM = """You are a data extraction assistant for a visualization tool.

Given a user instruction, extract the visualization request and return ONLY JSON:
{
  "viz_type":   "<chart | table | figure>",
  "title":      "<title for the visualization>",
  "data":       { <extracted key-value data, or {} if not provided> },
  "config": {
    "chart_type":    "<bar | line | scatter | pie>",
    "x_label":       "<x-axis label or null>",
    "y_label":       "<y-axis label or null>",
    "format":        "<png | pdf | latex>",
    "color_scheme":  "<default | grayscale | IEEE_blue | custom>",
    "export_size":   "<1920x1080 | A4 | column_width or null>",
    "caption":       "<optional caption>",
    "figure_type":   "<chart | table>"
  },
  "details": "<any extra instructions>"
}
Respond with raw JSON only."""


def visualisation_node(state: GraphState) -> dict:
    # Lazy import — keeps the module importable even if visualization isn't installed
    from visualization import generate_chart, generate_table, export_figure

    project_id = state["project_id"]
    section_id = state.get("section_id")
    instruction = state.get("instruction") or state["user_message"]
    prefs = state.get("preferences", {})
    provider = prefs.get("llm_provider", "groq")

    try:
        llm = get_llm(provider, temperature=0.1)
        response = llm.invoke([
            SystemMessage(content=_VIZ_SYSTEM),
            HumanMessage(content=instruction),
        ])
        parsed = json.loads(response.content.strip())
    except Exception as exc:
        logger.warning("Viz param extraction failed (%s), using defaults", exc)
        parsed = {
            "viz_type": "chart", "title": "Visualization",
            "data": {}, "config": {"format": "png", "figure_type": "chart"},
            "details": instruction,
        }

    viz_type = parsed.get("viz_type", "chart")
    title = parsed.get("title", "Visualization")
    data = parsed.get("data", {})
    config = parsed.get("config", {})
    details = parsed.get("details", "")

    if state.get("hitl_feedback"):
        config["details"] = f"{details}\nFeedback: {state['hitl_feedback']}"

    try:
        if viz_type == "table":
            file_path = generate_table(data, config)
        elif viz_type == "figure":
            file_path = export_figure(data, config)
        else:
            file_path = generate_chart(data, config)
    except Exception as exc:
        logger.error("Visualization module failed: %s", exc)
        return {"error": f"Visualization error: {exc}", "agent_output": None}

    try:
        repo.save_visualization(
            project_id=project_id, section_id=section_id, viz_type=viz_type,
            title=title, raw_data=data, config=config, file_path=file_path,
            export_format=config.get("format", "png"),
            export_size=config.get("export_size"),
            color_scheme=config.get("color_scheme", "default"), details=details,
        )
    except Exception as exc:
        logger.warning("Could not save visualization to DB: %s", exc)

    return {
        "agent_output": f"✅ **{viz_type.capitalize()} generated:** `{file_path}`\n\n**Title:** {title}",
        "last_agent":   "visualize",
        "error":        None,
        "hitl_action":  None,
        "hitl_feedback": None,
    }


logger = logging.getLogger(__name__)

_VIZ_SYSTEM = """You are a data extraction assistant for a visualization tool.

Given a user instruction, extract the visualization request and return ONLY JSON:
{
  "viz_type":   "<chart | table | figure>",
  "title":      "<title for the visualization>",
  "data":       { <extracted key-value data, or {} if not provided> },
  "config": {
    "chart_type":    "<bar | line | scatter | pie>",
    "x_label":       "<x-axis label or null>",
    "y_label":       "<y-axis label or null>",
    "format":        "<png | pdf | latex>",
    "color_scheme":  "<default | grayscale | IEEE_blue | custom>",
    "export_size":   "<1920x1080 | A4 | column_width or null>",
    "caption":       "<optional caption>",
    "figure_type":   "<chart | table>"
  },
  "details": "<any extra instructions>"
}

If data is embedded in the instruction, parse it out.
If format is not mentioned, default to png.
Respond with raw JSON only."""


def visualisation_node(state: GraphState) -> dict:
    project_id = state["project_id"]
    section_id = state.get("section_id")
    instruction = state.get("instruction") or state["user_message"]
    prefs = state.get("preferences", {})
    provider = prefs.get("llm_provider", "groq")

    # ── Extract viz parameters via LLM ──
    try:
        llm = get_llm(provider, temperature=0.1)
        response = llm.invoke([
            SystemMessage(content=_VIZ_SYSTEM),
            HumanMessage(content=instruction),
        ])
        parsed = json.loads(response.content.strip())
    except Exception as exc:
        logger.warning(
            "Viz parameter extraction failed (%s), using defaults", exc)
        parsed = {
            "viz_type": "chart",
            "title":    "Visualization",
            "data":     {},
            "config":   {"format": "png", "figure_type": "chart"},
            "details":  instruction,
        }

    viz_type = parsed.get("viz_type", "chart")
    title = parsed.get("title", "Visualization")
    data = parsed.get("data", {})
    config = parsed.get("config", {})
    details = parsed.get("details", "")

    # Inject hitl feedback for regenerate round
    if state.get("hitl_feedback"):
        details = f"{details}\nFeedback: {state['hitl_feedback']}"
        config["details"] = details

    # ── Call visualization module ──
    try:
        if viz_type == "table":
            file_path = generate_table(data, config)
        elif viz_type == "figure":
            file_path = export_figure(data, config)
        else:
            file_path = generate_chart(data, config)
    except Exception as exc:
        logger.error("Visualization module failed: %s", exc)
        return {"error": f"Visualization error: {exc}", "agent_output": None}

    # ── Persist to DB ──
    try:
        repo.save_visualization(
            project_id=project_id,
            section_id=section_id,
            viz_type=viz_type,
            title=title,
            raw_data=data,
            config=config,
            file_path=file_path,
            export_format=config.get("format", "png"),
            export_size=config.get("export_size"),
            color_scheme=config.get("color_scheme", "default"),
            details=details,
        )
    except Exception as exc:
        logger.warning("Could not save visualization to DB: %s", exc)

    output_text = (
        f"✅ **{viz_type.capitalize()} generated:** `{file_path}`\n\n"
        f"**Title:** {title}\n"
        + (f"**Details:** {details}" if details else "")
    )

    return {
        "agent_output": output_text,
        "last_agent":   "visualize",
        "error":        None,
        "hitl_action":  None,
        "hitl_feedback": None,
    }
