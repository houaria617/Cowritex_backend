"""
tests/integration/test_graph_flow.py
Integration tests for the full LangGraph orchestrator flow.
All LLM + agent calls are mocked — no real credentials needed.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from langgraph.checkpoint.memory import MemorySaver


# ── Import graph lazily inside fixtures so import errors are clear ──

def _build():
    from orchestrator.graph import build_graph

    return build_graph(MemorySaver())


@pytest.fixture(scope="module")
def graph():
    return _build()


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _state(user_message: str, intent: str = "write") -> dict:
    return {
        "project_id": "proj-int-1", "user_id": "user-int-1",
        "section_id": "sec-int-1", "user_message": user_message,
        "intent": intent, "last_agent": None, "instruction": user_message,
        "agent_output": None, "hitl_action": None, "hitl_feedback": None,
        "human_edited_text": None,
        "preferences": {
            "llm_provider": "groq", "writing_style": "formal",
            "tone": "academic", "citation_style": "APA",
            "language": "English", "grounded_only": False,
        },
        "document_context": None, "messages": [], "error": None,
    }


# ── Write → HITL → Approve ────────────────────────────────────

class TestWriteApproveFlow:

    @patch("orchestrator.nodes.writing_node.repo")
    @patch("orchestrator.nodes.writing_node._traced_writing")
    @patch("orchestrator.nodes.intent_classifier_node.repo")
    @patch("orchestrator.nodes.intent_classifier_node.get_llm")
    def test_write_pauses_at_hitl(
        self, mock_llm, mock_ic_repo, mock_agent, mock_write_repo, graph
    ):
        mock_llm.return_value.invoke.return_value = MagicMock(
            content='{"intent": "write", "instruction": "Draft an introduction"}'
        )
        mock_ic_repo.get_preferences.return_value = {}
        mock_agent.return_value = "Generated introduction text."
        mock_write_repo.get_current_content.return_value = ""
        mock_write_repo.create_suggestion.return_value = "sugg-1"

        config = _cfg("thread-write-1")
        graph.invoke(_state("Draft an introduction"), config)

        snap = graph.get_state(config)
        assert "hitl" in snap.next, "Graph should be paused at HITL"

    @patch("orchestrator.nodes.persist_node.repo")
    @patch("orchestrator.nodes.hitl_node.repo")
    @patch("orchestrator.nodes.writing_node.repo")
    @patch("orchestrator.nodes.writing_node._traced_writing")
    @patch("orchestrator.nodes.intent_classifier_node.repo")
    @patch("orchestrator.nodes.intent_classifier_node.get_llm")
    def test_approve_completes_flow(
        self, mock_llm, mock_ic_repo, mock_agent, mock_write_repo,
        mock_hitl_repo, mock_persist_repo, graph
    ):
        mock_llm.return_value.invoke.return_value = MagicMock(
            content='{"intent": "write", "instruction": "Draft an introduction"}'
        )
        mock_ic_repo.get_preferences.return_value = {}
        mock_agent.return_value = "Generated introduction."
        mock_write_repo.get_current_content.return_value = ""
        mock_write_repo.create_suggestion.return_value = "sugg-2"
        mock_hitl_repo.get_pending_suggestion.return_value = {"id": "sugg-2"}

        # persist_node calls repo.db.table(...) for progress update — mock it fully
        mock_persist_repo.db = MagicMock()
        mock_persist_repo.db.table.return_value.select.return_value \
            .eq.return_value.execute.return_value.data = []
        mock_persist_repo.save_message.return_value = None
        mock_persist_repo.save_new_version.return_value = None
        mock_persist_repo.get_current_content.return_value = "Generated introduction."

        config = _cfg("thread-approve-1")
        graph.invoke(_state("Draft an introduction"), config)

        graph.update_state(config, {"hitl_action": "approve"})
        final = graph.invoke(None, config)

        snap = graph.get_state(config)
        assert not snap.next, "Graph should be complete"
        assert final.get("agent_output") == "Generated introduction."


# ── Write → HITL → Reject → Regenerate ───────────────────────

class TestWriteRejectFlow:

    @patch("orchestrator.nodes.hitl_node.repo")
    @patch("orchestrator.nodes.writing_node.repo")
    @patch("orchestrator.nodes.writing_node._traced_writing")
    @patch("orchestrator.nodes.intent_classifier_node.repo")
    @patch("orchestrator.nodes.intent_classifier_node.get_llm")
    def test_reject_reruns_writing_agent(
        self, mock_llm, mock_ic_repo, mock_agent, mock_write_repo,
        mock_hitl_repo, graph
    ):
        call_count = {"n": 0}

        def agent_side_effect(**kwargs):
            call_count["n"] += 1
            return f"Draft v{call_count['n']}."

        mock_llm.return_value.invoke.return_value = MagicMock(
            content='{"intent": "write", "instruction": "Draft intro"}'
        )
        mock_ic_repo.get_preferences.return_value = {}
        mock_agent.side_effect = agent_side_effect
        mock_write_repo.get_current_content.return_value = ""
        mock_write_repo.create_suggestion.return_value = "sugg-3"
        mock_hitl_repo.get_pending_suggestion.return_value = {"id": "sugg-3"}

        config = _cfg("thread-reject-1")
        graph.invoke(_state("Draft intro"), config)

        graph.update_state(config, {
            "hitl_action": "reject",
            "hitl_feedback": "Too short.",
        })
        graph.invoke(None, config)

        snap = graph.get_state(config)
        assert "hitl" in snap.next, "Should pause again for second draft"
        assert call_count["n"] == 2, "Agent should have been called twice"


# ── Write → HITL → Edit ───────────────────────────────────────

class TestWriteEditFlow:

    @patch("orchestrator.nodes.persist_node.repo")
    @patch("orchestrator.nodes.edit_node.repo")
    @patch("orchestrator.nodes.hitl_node.repo")
    @patch("orchestrator.nodes.writing_node.repo")
    @patch("orchestrator.nodes.writing_node._traced_writing")
    @patch("orchestrator.nodes.intent_classifier_node.repo")
    @patch("orchestrator.nodes.intent_classifier_node.get_llm")
    def test_edit_saves_human_version(
        self, mock_llm, mock_ic_repo, mock_agent, mock_write_repo,
        mock_hitl_repo, mock_edit_repo, mock_persist_repo, graph
    ):
        mock_llm.return_value.invoke.return_value = MagicMock(
            content='{"intent": "write", "instruction": "Draft methodology"}'
        )
        mock_ic_repo.get_preferences.return_value = {}
        mock_agent.return_value = "AI draft."
        mock_write_repo.get_current_content.return_value = ""
        mock_write_repo.create_suggestion.return_value = "sugg-4"
        mock_hitl_repo.get_pending_suggestion.return_value = {"id": "sugg-4"}
        mock_edit_repo.save_new_version.return_value = None
        mock_persist_repo.db = MagicMock()
        mock_persist_repo.db.table.return_value.select.return_value \
            .eq.return_value.execute.return_value.data = []
        mock_persist_repo.save_message.return_value = None
        mock_persist_repo.save_new_version.return_value = None

        config = _cfg("thread-edit-1")
        graph.invoke(_state("Draft methodology"), config)

        graph.update_state(config, {
            "hitl_action":       "edit",
            "human_edited_text": "Researcher's improved version.",
        })
        final = graph.invoke(None, config)

        assert final.get("agent_output") == "Researcher's improved version."
        mock_edit_repo.save_new_version.assert_called_once()


# ── Chat (no HITL) ────────────────────────────────────────────

class TestChatFlow:

    @patch("orchestrator.nodes.chat_node.repo")
    @patch("orchestrator.nodes.chat_node.get_llm")
    @patch("orchestrator.nodes.intent_classifier_node.repo")
    @patch("orchestrator.nodes.intent_classifier_node.get_llm")
    def test_chat_completes_without_hitl(
        self, mock_ic_llm, mock_ic_repo, mock_chat_llm, mock_chat_repo, graph
    ):
        mock_ic_llm.return_value.invoke.return_value = MagicMock(
            content='{"intent": "chat", "instruction": "What is RAG?"}'
        )
        mock_ic_repo.get_preferences.return_value = {}
        mock_chat_llm.return_value.invoke.return_value = MagicMock(
            content="RAG stands for Retrieval-Augmented Generation."
        )
        mock_chat_repo.get_recent_messages.return_value = []
        mock_chat_repo.save_message.return_value = None

        config = _cfg("thread-chat-1")
        final = graph.invoke(_state("What is RAG?", intent="chat"), config)

        snap = graph.get_state(config)
        assert not snap.next, "Chat should complete without HITL pause"
        assert "RAG" in final.get("agent_output", "")


# ── Error flow ────────────────────────────────────────────────

class TestErrorFlow:

    @patch("orchestrator.nodes.intent_classifier_node.repo")
    @patch("orchestrator.nodes.intent_classifier_node.get_llm")
    def test_unknown_intent_reaches_error_node(
        self, mock_llm, mock_repo, graph
    ):
        mock_llm.return_value.invoke.return_value = MagicMock(
            content='{"intent": "unknown", "instruction": "???"}'
        )
        mock_repo.get_preferences.return_value = {}

        config = _cfg("thread-error-1")
        final = graph.invoke(_state("asdfghjkl???", intent="unknown"), config)

        snap = graph.get_state(config)
        assert not snap.next
        assert "not sure" in (final.get("agent_output") or "").lower()
