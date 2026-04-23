"""
orchestrator/state.py
──────────────────────
Canonical LangGraph state for CoWriteX.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    # ── Identity ──────────────────────────────────────────────────────────────
    project_id:  str
    section_id:  Optional[str]
    user_id:     str

    # ── Routing ───────────────────────────────────────────────────────────────
    intent:  Literal["write", "literature",
                     "visualize", "search", "chat", "unknown"]
    intents: list[str]           # e.g. ["write", "literature"] or ["search"]
    # "writing" | "literature" | "visualize" | "search"
    last_agent: Optional[str]

    # ── Input ─────────────────────────────────────────────────────────────────
    user_message: str
    instruction:  Optional[str]  # cleaned version from intent classifier

    # ── Agent output ──────────────────────────────────────────────────────────
    # {"write": "...", "literature": "...", "search": "..."}
    agent_outputs: dict
    agent_output:  Optional[str]  # text shown to researcher at HITL

    # ── Search results (populated by search_node) ─────────────────────────────
    search_results: list          # List[dict]  raw paper records
    # formatted markdown summary for display/downstream
    search_summary: Optional[str]

    # ── HITL ──────────────────────────────────────────────────────────────────
    hitl_action: Optional[Literal["approve", "edit", "reject", "regenerate"]]
    # researcher comment on reject / regenerate
    hitl_feedback:     Optional[str]
    # researcher's direct edit (for "edit" action)
    human_edited_text: Optional[str]

    # ── Context ───────────────────────────────────────────────────────────────
    preferences:      dict         # from project_preferences table
    document_context: Optional[str]  # current section text (optional pre-load)

    # ── Conversation history (LangGraph managed) ──────────────────────────────
    messages: Annotated[list, add_messages]

    # ── System errors only ────────────────────────────────────────────────────
    error: Optional[str]
