"""
tests/unit/test_nodes.py
Unit tests for every orchestrator node.
All external I/O (DB, LLM, agents) is mocked.
"""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock


# ── State factory ─────────────────────────────────────────────

def s(**kw) -> dict:
    base = dict(
        project_id="proj-1", user_id="user-1", section_id="sec-1",
        user_message="Write an introduction",
        intent="write", last_agent=None,
        instruction="Write an introduction",
        agent_output=None, hitl_action=None,
        hitl_feedback=None, human_edited_text=None,
        preferences={"llm_provider": "groq", "writing_style": "formal",
                     "tone": "academic", "citation_style": "APA",
                     "language": "English", "grounded_only": False},
        document_context=None, messages=[], error=None,
    )
    base.update(kw)
    return base


# ──────────────────────────────────────────────────────────────
# intent_classifier_node
# ──────────────────────────────────────────────────────────────

class TestIntentClassifier:

    @patch("orchestrator.nodes.intent_classifier_node.repo")
    @patch("orchestrator.nodes.intent_classifier_node.get_llm")
    def test_classifies_write(self, mock_get_llm, mock_repo):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"intent": "write", "instruction": "Write an introduction"}'
        )
        mock_get_llm.return_value = llm
        mock_repo.get_preferences.return_value = {"llm_provider": "groq"}

        from orchestrator.nodes.intent_classifier_node import intent_classifier_node # type: ignore
        result = intent_classifier_node(s(preferences={}))

        assert result["intent"] == "write"
        assert result["error"] is None

    @patch("orchestrator.nodes.intent_classifier_node.repo")
    @patch("orchestrator.nodes.intent_classifier_node.get_llm")
    def test_falls_back_to_unknown_on_llm_error(self, mock_get_llm, mock_repo):
        mock_get_llm.side_effect = RuntimeError("no API key")
        mock_repo.get_preferences.return_value = {}

        from orchestrator.nodes.intent_classifier_node import intent_classifier_node
        result = intent_classifier_node(s(preferences={}))

        assert result["intent"] == "unknown"

    @patch("orchestrator.nodes.intent_classifier_node.repo")
    @patch("orchestrator.nodes.intent_classifier_node.get_llm")
    def test_invalid_intent_coerced_to_unknown(self, mock_get_llm, mock_repo):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"intent": "banana", "instruction": "do stuff"}'
        )
        mock_get_llm.return_value = llm
        mock_repo.get_preferences.return_value = {}

        from orchestrator.nodes.intent_classifier_node import intent_classifier_node
        result = intent_classifier_node(s(preferences={}))
        assert result["intent"] == "unknown"

    @patch("orchestrator.nodes.intent_classifier_node.repo")
    @patch("orchestrator.nodes.intent_classifier_node.get_llm")
    def test_uses_existing_preferences_without_db_call(self, mock_get_llm, mock_repo):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"intent": "chat", "instruction": "hello"}'
        )
        mock_get_llm.return_value = llm

        from orchestrator.nodes.intent_classifier_node import intent_classifier_node
        intent_classifier_node(s())  # preferences already in state

        mock_repo.get_preferences.assert_not_called()


# ──────────────────────────────────────────────────────────────
# writing_node
# ──────────────────────────────────────────────────────────────

class TestWritingNode:

    @patch("orchestrator.nodes.writing_node.repo")
    @patch("orchestrator.nodes.writing_node._traced_writing")
    def test_success(self, mock_agent, mock_repo):
        mock_agent.return_value = "Clean introduction text."
        mock_repo.get_current_content.return_value = "old content"
        mock_repo.create_suggestion.return_value = "sugg-1"

        from orchestrator.nodes.writing_node import writing_node
        result = writing_node(s())

        assert result["agent_output"] == "Clean introduction text."
        assert result["last_agent"] == "writing"
        assert result["error"] is None
        assert result["hitl_action"] is None

    @patch("orchestrator.nodes.writing_node.repo")
    @patch("orchestrator.nodes.writing_node._traced_writing")
    def test_agent_error_prefix(self, mock_agent, mock_repo):
        mock_agent.return_value = "[ERROR] LLM call failed: timeout"
        mock_repo.get_current_content.return_value = ""

        from orchestrator.nodes.writing_node import writing_node
        result = writing_node(s())
        assert result["agent_output"] is None
        assert "[ERROR]" in result["error"]

    @patch("orchestrator.nodes.writing_node.repo")
    @patch("orchestrator.nodes.writing_node._traced_writing")
    def test_agent_exception(self, mock_agent, mock_repo):
        mock_agent.side_effect = RuntimeError("agent crashed")
        mock_repo.get_current_content.return_value = ""

        from orchestrator.nodes.writing_node import writing_node
        result = writing_node(s())
        assert "Writing agent error" in result["error"]

    @patch("orchestrator.nodes.writing_node.repo")
    @patch("orchestrator.nodes.writing_node._traced_writing")
    def test_hitl_feedback_appended_to_instruction(self, mock_agent, mock_repo):
        mock_agent.return_value = "Revised text."
        mock_repo.get_current_content.return_value = "old"

        from orchestrator.nodes.writing_node import writing_node
        writing_node(s(hitl_feedback="Make it shorter"))

        call_kwargs = mock_agent.call_args
        assert "Make it shorter" in call_kwargs.kwargs.get("instruction",
                                                           call_kwargs.args[1] if len(call_kwargs.args) > 1 else "")


# ──────────────────────────────────────────────────────────────
# hitl_node
# ──────────────────────────────────────────────────────────────

class TestHITLNode:

    @patch("orchestrator.nodes.hitl_node.repo")
    def test_approve_resolves_suggestion_as_accepted(self, mock_repo):
        mock_repo.get_pending_suggestion.return_value = {"id": "sugg-1"}

        from orchestrator.nodes.hitl_node import hitl_node
        result = hitl_node(s(hitl_action="approve"))

        assert result["hitl_action"] == "approve"
        mock_repo.resolve_suggestion.assert_called_once_with(
            suggestion_id="sugg-1", status="accepted", feedback=None
        )

    @patch("orchestrator.nodes.hitl_node.repo")
    def test_reject_marks_suggestion_rejected(self, mock_repo):
        mock_repo.get_pending_suggestion.return_value = {"id": "sugg-1"}

        from orchestrator.nodes.hitl_node import hitl_node
        result = hitl_node(
            s(hitl_action="reject", hitl_feedback="Too verbose"))

        mock_repo.resolve_suggestion.assert_called_once_with(
            suggestion_id="sugg-1", status="rejected", feedback="Too verbose"
        )

    @patch("orchestrator.nodes.hitl_node.repo")
    def test_invalid_action_defaults_to_approve(self, mock_repo):
        mock_repo.get_pending_suggestion.return_value = None

        from orchestrator.nodes.hitl_node import hitl_node
        result = hitl_node(s(hitl_action="BANANA"))
        assert result["hitl_action"] == "approve"

    @patch("orchestrator.nodes.hitl_node.repo")
    def test_no_section_skips_db(self, mock_repo):
        from orchestrator.nodes.hitl_node import hitl_node
        hitl_node(s(section_id=None, hitl_action="approve"))
        mock_repo.get_pending_suggestion.assert_not_called()


# ──────────────────────────────────────────────────────────────
# edit_node
# ──────────────────────────────────────────────────────────────

class TestEditNode:

    @patch("orchestrator.nodes.edit_node.repo")
    def test_saves_human_version(self, mock_repo):
        from orchestrator.nodes.edit_node import edit_node
        result = edit_node(s(human_edited_text="My edited text."))

        assert result["agent_output"] == "My edited text."
        assert result["error"] is None
        mock_repo.save_new_version.assert_called_once_with(
            section_id="sec-1",
            content="My edited text.",
            author_type="human",
            suggestion_id=None,
        )

    def test_no_text_returns_error(self):
        from orchestrator.nodes.edit_node import edit_node
        result = edit_node(s(human_edited_text=None, agent_output=None))
        assert result["error"] is not None
        assert result["agent_output"] is None


# ──────────────────────────────────────────────────────────────
# error_node
# ──────────────────────────────────────────────────────────────

class TestErrorNode:

    def test_unknown_intent_suggests_commands(self):
        from orchestrator.nodes.error_node import error_node
        result = error_node(s(intent="unknown", error=None))
        assert "not sure" in result["agent_output"].lower()
        assert len(result["messages"]) == 1

    def test_system_error_shows_error_message(self):
        from orchestrator.nodes.error_node import error_node
        result = error_node(s(intent="write", error="DB timeout"))
        assert "DB timeout" in result["agent_output"]


# ──────────────────────────────────────────────────────────────
# output_node
# ──────────────────────────────────────────────────────────────

class TestOutputNode:

    def test_approve_adds_status_prefix(self):
        from orchestrator.nodes.output_node import output_node
        result = output_node(
            s(agent_output="Great text.", hitl_action="approve"))
        msg = result["messages"][0].content
        assert "approved" in msg.lower()
        assert "Great text." in msg

    def test_edit_adds_edit_prefix(self):
        from orchestrator.nodes.output_node import output_node
        result = output_node(s(agent_output="Edited.", hitl_action="edit"))
        assert "edited" in result["messages"][0].content.lower()

    def test_no_hitl_action_returns_plain_output(self):
        from orchestrator.nodes.output_node import output_node
        result = output_node(s(agent_output="Chat reply.", hitl_action=None))
        assert "Chat reply." in result["messages"][0].content
