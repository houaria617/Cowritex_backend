"""
tests/unit/test_router.py
Unit tests for the routing functions — no I/O, no LLM, no DB.
"""
import pytest
from orchestrator.router import route_intent, route_hitl, _last_agent_node


def s(**kw):
    base = dict(
        project_id="p1", user_id="u1", section_id="s1",
        user_message="test", intent="write", last_agent=None,
        instruction=None, agent_output=None, hitl_action=None,
        hitl_feedback=None, human_edited_text=None,
        preferences={}, document_context=None, messages=[], error=None,
    )
    base.update(kw)
    return base


# ── route_intent ──────────────────────────────────────────────

@pytest.mark.parametrize("intent,expected", [
    ("write",      "write"),
    ("literature", "literature"),
    ("visualize",  "visualize"),
    ("chat",       "chat"),
    ("unknown",    "unknown"),
])
def test_route_intent_all_values(intent, expected):
    assert route_intent(s(intent=intent)) == expected


def test_route_intent_missing_defaults_to_unknown():
    state = s()
    del state["intent"]
    assert route_intent(state) == "unknown"


# ── route_hitl ────────────────────────────────────────────────

def test_route_hitl_approve_returns_persist():
    assert route_hitl(s(hitl_action="approve")) == "persist"


def test_route_hitl_edit_returns_edit():
    assert route_hitl(s(hitl_action="edit")) == "edit"


def test_route_hitl_reject_writing():
    assert route_hitl(
        s(hitl_action="reject", last_agent="writing")) == "writing"


def test_route_hitl_reject_literature():
    assert route_hitl(
        s(hitl_action="reject", last_agent="literature")) == "literature"


def test_route_hitl_regenerate_visualise():
    assert route_hitl(s(hitl_action="regenerate",
                      last_agent="visualize")) == "visualisation"


def test_route_hitl_none_defaults_to_persist():
    assert route_hitl(s(hitl_action=None)) == "persist"


# ── _last_agent_node ─────────────────────────────────────────

@pytest.mark.parametrize("agent,node", [
    ("writing",    "writing"),
    ("literature", "literature"),
    ("visualize",  "visualisation"),
    (None,         "intent_classifier"),
    ("garbage",    "intent_classifier"),
])
def test_last_agent_node_mapping(agent, node):
    assert _last_agent_node(s(last_agent=agent)) == node
