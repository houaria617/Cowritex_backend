"""
orchestrator/state.py
──────────────────────
Canonical LangGraph state for CoWriteX.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    # ── Identity ──
    project_id: str
    section_id: Optional[str]
    user_id: str

    # ── Routing ──
    intent: Literal["write", "literature", "visualize", "chat", "unknown"]
    intents: list[str]           # NEW — e.g. ["write", "literature"]
    last_agent: Optional[str]          # "writing" | "literature" | "visualize"

    # ── Input ──
    user_message: str
    instruction: Optional[str]         # cleaned version from intent classifier

    # ── Agent output ──
    agent_outputs: dict         # NEW — {"write": "...", "literature": "..."}
    agent_output: Optional[str]        # text shown to researcher at HITL

    # ── HITL ──
    hitl_action: Optional[Literal["approve", "edit", "reject", "regenerate"]]
    # researcher comment on reject / regenerate
    hitl_feedback: Optional[str]
    # researcher's direct edit (for "edit" action)
    human_edited_text: Optional[str]

    # ── Context ──
    preferences: dict                  # from project_preferences table
    # current section text (optional pre-load)
    document_context: Optional[str]

    # ── Conversation history (LangGraph managed) ──
    messages: Annotated[list, add_messages]

    # ── System errors only ──
    error: Optional[str]
