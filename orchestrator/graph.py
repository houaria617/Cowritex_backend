"""
orchestrator/graph.py
──────────────────────
Builds and compiles the LangGraph state machine.

Flow summary
────────────
intent_classifier
    │
    ├─► "search"     → search_node → persist → output → END
    │                  (auto-approved — no HITL for factual retrieval)
    │
    ├─► "literature" → literature_node → merge → hitl
    │     (grounded_only=True  → agent uses ChromaDB only)
    │     (grounded_only=False → agent fetches web resources itself)
    │
    ├─► "write"      → writing       → merge → hitl
    ├─► "visualize"  → visualisation → merge → hitl
    │
    ├─► "chat"       → chat → output → END
    └─► "unknown" / error → error_handler → END

HITL resume actions (literature / writing / visualisation only):
    approve    → persist → output → END
    edit       → edit    → persist → output → END
    regenerate → back to the originating agent
    reject     → back to the originating agent (with feedback)
"""

from langgraph.graph import StateGraph, END

from .state import GraphState
from .router import route_intent, route_hitl
from .nodes import (
    intent_classifier_node,
    search_node,
    writing_node,
    literature_node,
    visualisation_node,
    chat_node,
    hitl_node,
    edit_node,
    persist_node,
    output_node,
    error_node,
    merge_node,
)


def build_graph(checkpointer):
    g = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("intent_classifier", intent_classifier_node)
    g.add_node("search",            search_node)
    g.add_node("writing",           writing_node)
    g.add_node("literature",        literature_node)
    g.add_node("visualisation",     visualisation_node)
    g.add_node("chat",              chat_node)
    g.add_node("merge",             merge_node)
    g.add_node("hitl",              hitl_node)
    g.add_node("edit",              edit_node)
    g.add_node("persist",           persist_node)
    g.add_node("output",            output_node)
    g.add_node("error_handler",     error_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    g.set_entry_point("intent_classifier")

    # ── intent_classifier → agents ────────────────────────────────────────────
    g.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "search":        "search",
            "literature":    "literature",
            "writing":       "writing",
            "visualisation": "visualisation",
            "chat":          "chat",
            "error_handler": "error_handler",
        },
    )

    # ── search fast-path: auto-approved, skip merge + HITL ───────────────────
    g.add_edge("search", "persist")

    # ── AI agents → merge → HITL ──────────────────────────────────────────────
    for agent in ("writing", "literature", "visualisation"):
        g.add_edge(agent, "merge")

    g.add_edge("merge", "hitl")

    # ── chat fast-path ────────────────────────────────────────────────────────
    g.add_edge("chat", "output")

    # ── HITL branching (literature / writing / visualisation only) ────────────
    g.add_conditional_edges(
        "hitl",
        route_hitl,
        {
            "persist":       "persist",
            "edit":          "edit",
            "writing":       "writing",
            "literature":    "literature",
            "visualisation": "visualisation",
        },
    )

    # ── Finalisation ──────────────────────────────────────────────────────────
    g.add_edge("edit",          "persist")
    g.add_edge("persist",       "output")
    g.add_edge("output",        END)
    g.add_edge("error_handler", END)

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl"],
    )
