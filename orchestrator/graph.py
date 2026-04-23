"""
orchestrator/graph.py
──────────────────────
Builds and compiles the LangGraph state machine.

Flow summary
────────────
intent_classifier
    │
    ├─► "search"                         → search_node → merge → hitl
    │
    ├─► "literature" + grounded_only=False → search_node → literature → merge → hitl
    ├─► "literature" + grounded_only=True  →              literature → merge → hitl
    │
    ├─► "write"                          → writing      → merge → hitl
    ├─► "visualize"                      → visualisation → merge → hitl
    │
    ├─► "chat"                           → chat → output → END
    └─► "unknown" / error                → error_handler → END

Parallel fan-out example:
    intents = ["write", "literature"]
        → writing + literature run, both feed merge → hitl

HITL resume actions:
    approve    → persist → output → END
    edit       → edit    → persist → output → END
    regenerate → back to the originating agent
    reject     → back to the originating agent (with feedback)
"""

from langgraph.graph import StateGraph, END

from .state import GraphState
from .router import route_intent, route_after_search, route_hitl
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
    g.add_node("search",            search_node)          # NEW
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

    # ── intent_classifier → fan-out ───────────────────────────────────────────
    # "search" intent and "literature + web" both go to search_node first.
    # Everything else routes directly to its agent.
    g.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "search":        "search",       # explicit search intent
            "search_first":  "search",       # literature intent, grounded_only=False
            "writing":       "writing",
            "literature":    "literature",   # grounded_only=True path
            "visualisation": "visualisation",
            "chat":          "chat",
            "error_handler": "error_handler",
        },
    )

    # ── After search_node: go to literature (if compound) or merge (if pure search) ──
    g.add_conditional_edges(
        "search",
        route_after_search,
        {
            "literature": "literature",   # search results handed off to lit node
            "merge":      "merge",        # pure "search" intent stops at merge/HITL
        },
    )

    # ── Agent → merge (all non-chat agents converge here) ────────────────────
    for agent in ("writing", "literature", "visualisation"):
        g.add_edge(agent, "merge")

    g.add_edge("merge", "hitl")

    # ── chat fast-path ────────────────────────────────────────────────────────
    g.add_edge("chat", "output")

    # ── HITL branching ────────────────────────────────────────────────────────
    g.add_conditional_edges(
        "hitl",
        route_hitl,
        {
            "persist":       "persist",
            "edit":          "edit",
            "writing":       "writing",
            "literature":    "literature",
            "visualisation": "visualisation",
            "search":        "search",       # regenerate a pure search
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
