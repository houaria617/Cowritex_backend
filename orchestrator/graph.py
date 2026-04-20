"""
orchestrator/graph.py
──────────────────────
Builds and compiles the LangGraph state machine.

Usage:
    from langgraph.checkpoint.memory import MemorySaver          # dev
    # from langgraph.checkpoint.postgres import PostgresSaver    # prod (Supabase)

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer)

    # First invocation (new thread)
    config = {"configurable": {"thread_id": "some-uuid"}}
    result = graph.invoke(initial_state, config)

    # Resume after HITL pause
    graph.update_state(config, {"hitl_action": "approve"})
    result = graph.invoke(None, config)
"""

from langgraph.graph import StateGraph, END

from .state import GraphState
from .router import route_intent, route_hitl
from .nodes import (
    intent_classifier_node,
    writing_node,
    literature_node,
    visualisation_node,
    chat_node,
    hitl_node,
    edit_node,
    persist_node,
    output_node,
    error_node,
)


def build_graph(checkpointer):
    g = StateGraph(GraphState)

    # ── Register nodes ──
    g.add_node("intent_classifier", intent_classifier_node)
    g.add_node("writing",           writing_node)
    g.add_node("literature",        literature_node)
    g.add_node("visualisation",     visualisation_node)
    g.add_node("chat",              chat_node)
    g.add_node("hitl",              hitl_node)
    g.add_node("edit",              edit_node)
    g.add_node("persist",           persist_node)
    g.add_node("output",            output_node)
    g.add_node("error_handler",             error_node)

    # ── Entry point ──
    g.set_entry_point("intent_classifier")

    # ── Intent routing ──
    g.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "write":      "writing",
            "literature": "literature",
            "visualize":  "visualisation",
            "chat":       "chat",
            "unknown":    "error_handler",
        },
    )

    # ── Agent → HITL (agents that need researcher approval) ──
    for agent in ("writing", "literature", "visualisation"):
        g.add_edge(agent, "hitl")

    # ── Chat bypasses HITL — goes straight to output ──
    g.add_edge("chat", "output")

    # ── HITL routing (single conditional edge — no duplicate) ──
    g.add_conditional_edges(
        "hitl",
        route_hitl,
        {
            "persist":      "persist",
            "edit":         "edit",
            "writing":      "writing",       # reject/regenerate → back to agent
            "literature":   "literature",
            "visualisation": "visualisation",
        },
    )

    # ── Post-HITL flow ──
    g.add_edge("edit",    "persist")
    g.add_edge("persist", "output")
    g.add_edge("output",  END)
    g.add_edge("error_handler",   END)

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl"],   # graph pauses HERE waiting for researcher
    )
