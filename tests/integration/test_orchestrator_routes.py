"""
tests/integration/test_orchestrator_routes.py
──────────────────────────────────────────────
Integration tests for the core orchestrator endpoints:
  POST /projects/{id}/run
  POST /projects/{id}/run/{thread_id}/resume
  GET  /projects/{id}/run/{thread_id}/status

The LangGraph graph is mocked completely — these tests verify
the HTTP contract, not the AI logic.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import TEST_PROJECT_ID, TEST_USER_ID, TEST_THREAD_ID, make_project, make_preferences


# ── Mock graph state returned after a run ────────────────────────────────────

def _fake_state_after_run(intent: str = "write") -> dict:
    return {
        "project_id":   TEST_PROJECT_ID,
        "user_id":      TEST_USER_ID,
        "intent":       intent,
        "intents":      [intent],
        "agent_output": "Here is a drafted introduction paragraph.",
        "agent_outputs": {intent: "Here is a drafted introduction paragraph."},
        "last_agent":   intent,
        "hitl_action":  None,
        "error":        None,
        "messages":     [],
    }


def _fake_state_after_resume() -> dict:
    return {
        **_fake_state_after_run(),
        "hitl_action": "approve",
        "agent_output": "✅ Output approved and saved.\n\nHere is a drafted introduction paragraph.",
    }


@pytest.fixture
def mock_graph():
    """Mock the entire LangGraph graph build so no LLM is called."""
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = _fake_state_after_run()
    fake_graph.get_state.return_value = MagicMock(
        values=_fake_state_after_run())

    with patch("api.routes.orchestrator._get_graph", return_value=fake_graph), \
            patch("api.routes.orchestrator._graph", fake_graph):
        yield fake_graph


# ─────────────────────────────────────────────────────────────────────────────
# POST /projects/{id}/run
# ─────────────────────────────────────────────────────────────────────────────

class TestRunEndpoint:

    def test_run_returns_thread_id_and_output(self, client, mock_repo, mock_graph, auth_headers):
        mock_repo.get_preferences.return_value = make_preferences()
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/run",
            json={"user_message": "Write an introduction about RAG systems"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "thread_id" in data
        assert "agent_output" in data
        assert data["status"] == "pending_review"
        assert isinstance(data["intents"], list)

    def test_run_persists_thread_id(self, client, mock_repo, mock_graph, auth_headers):
        client.post(
            f"/projects/{TEST_PROJECT_ID}/run",
            json={"user_message": "Find papers on transformers"},
            headers=auth_headers,
        )
        mock_repo.save_thread_id.assert_called_once()
        args = mock_repo.save_thread_id.call_args
        assert args[0][0] == TEST_PROJECT_ID   # first positional arg

    def test_run_with_section_id(self, client, mock_repo, mock_graph, auth_headers):
        section_id = str(uuid.uuid4())
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/run",
            json={
                "user_message": "Improve this section",
                "section_id": section_id,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        # Verify section_id was passed into initial_state
        call_args = mock_graph.invoke.call_args
        initial_state = call_args[0][0]
        assert initial_state["section_id"] == section_id

    def test_run_empty_message_rejected(self, client, mock_repo, auth_headers):
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/run",
            json={"user_message": ""},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_run_404_for_nonexistent_project(self, client, mock_repo, auth_headers):
        mock_repo.get_project.return_value = None
        resp = client.post(
            f"/projects/{uuid.uuid4()}/run",
            json={"user_message": "Hello"},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_run_graph_error_returns_500(self, client, mock_repo, mock_graph, auth_headers):
        mock_graph.invoke.side_effect = RuntimeError("LLM API down")
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/run",
            json={"user_message": "Write something"},
            headers=auth_headers,
        )
        assert resp.status_code == 500
        assert "Graph error" in resp.json()["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# POST /projects/{id}/run/{thread_id}/resume
# ─────────────────────────────────────────────────────────────────────────────

class TestResumeEndpoint:

    def test_approve_returns_completed(self, client, mock_repo, mock_graph, auth_headers):
        mock_graph.invoke.return_value = _fake_state_after_resume()
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/run/{TEST_THREAD_ID}/resume",
            json={"hitl_action": "approve"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["thread_id"] == TEST_THREAD_ID

    def test_edit_requires_human_edited_text(self, client, mock_repo, auth_headers):
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/run/{TEST_THREAD_ID}/resume",
            json={"hitl_action": "edit"},   # missing human_edited_text
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "human_edited_text" in resp.json()["detail"]

    def test_edit_with_text_returns_completed(self, client, mock_repo, mock_graph, auth_headers):
        mock_graph.invoke.return_value = _fake_state_after_resume()
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/run/{TEST_THREAD_ID}/resume",
            json={
                "hitl_action": "edit",
                "human_edited_text": "My manually edited version",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_regenerate_returns_pending_review(self, client, mock_repo, mock_graph, auth_headers):
        mock_graph.invoke.return_value = _fake_state_after_run()
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/run/{TEST_THREAD_ID}/resume",
            json={
                "hitl_action": "regenerate",
                "hitl_feedback": "Make it more concise",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending_review"

    def test_reject_with_feedback(self, client, mock_repo, mock_graph, auth_headers):
        mock_graph.invoke.return_value = _fake_state_after_run()
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/run/{TEST_THREAD_ID}/resume",
            json={
                "hitl_action": "reject",
                "hitl_feedback": "Completely wrong topic",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending_review"

    def test_invalid_action_rejected(self, client, auth_headers):
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/run/{TEST_THREAD_ID}/resume",
            json={"hitl_action": "maybe"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_resume_injects_correct_state(self, client, mock_repo, mock_graph, auth_headers):
        """
        After the fix, resume uses:
          graph.update_state(config, hitl_fields)   ← injects decision
          graph.invoke(None, config)                ← continues from checkpoint

        So we verify update_state was called with the correct HITL fields,
        not invoke's first argument (which is now None).
        """
        mock_graph.invoke.return_value = _fake_state_after_resume()
        client.post(
            f"/projects/{TEST_PROJECT_ID}/run/{TEST_THREAD_ID}/resume",
            json={
                "hitl_action": "regenerate",
                "hitl_feedback": "Be more formal",
            },
            headers=auth_headers,
        )
        # update_state must have been called with the HITL fields
        mock_graph.update_state.assert_called_once()
        update_call_args = mock_graph.update_state.call_args
        # second positional arg is the state dict
        injected_state = update_call_args[0][1]
        assert injected_state["hitl_action"] == "regenerate"
        assert injected_state["hitl_feedback"] == "Be more formal"

        # invoke must have been called with None as input (resume, not restart)
        invoke_call_args = mock_graph.invoke.call_args
        assert invoke_call_args[0][0] is None


# ─────────────────────────────────────────────────────────────────────────────
# GET /projects/{id}/run/{thread_id}/status
# ─────────────────────────────────────────────────────────────────────────────

class TestStatusEndpoint:

    def test_200_with_valid_thread(self, client, mock_repo, mock_graph, auth_headers):
        mock_graph.get_state.return_value = MagicMock(
            values=_fake_state_after_run())
        resp = client.get(
            f"/projects/{TEST_PROJECT_ID}/run/{TEST_THREAD_ID}/status",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread_id"] == TEST_THREAD_ID
        assert data["status"] in ("pending_review", "running")

    def test_404_for_nonexistent_thread(self, client, mock_repo, mock_graph, auth_headers):
        mock_graph.get_state.return_value = MagicMock(values=None)
        resp = client.get(
            f"/projects/{TEST_PROJECT_ID}/run/{uuid.uuid4()}/status",
            headers=auth_headers,
        )
        assert resp.status_code == 404
