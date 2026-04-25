"""
tests/integration/test_data_routes.py
──────────────────────────────────────
Integration tests for data-layer endpoints:
  sources, literature, suggestions, visualizations, chat
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import (
    TEST_PROJECT_ID, TEST_SECTION_ID,
    make_project, make_section, make_suggestion, make_visualization,
)

ANALYSIS_ID = str(uuid.uuid4())
SOURCE_ID = str(uuid.uuid4())
VIZ_ID = str(uuid.uuid4())
SUGGESTION_ID = str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────────────────────

class TestSourcesEndpoints:

    def test_list_sources_200(self, client, mock_repo, auth_headers):
        mock_repo.list_sources.return_value = [
            {"id": SOURCE_ID, "title": "Test Paper", "relevance_score": 0.95}
        ]
        resp = client.get(
            f"/projects/{TEST_PROJECT_ID}/sources", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["title"] == "Test Paper"

    def test_delete_source_204(self, client, mock_repo, auth_headers):
        resp = client.delete(f"/sources/{SOURCE_ID}", headers=auth_headers)
        assert resp.status_code == 204
        mock_repo.delete_source.assert_called_once_with(SOURCE_ID)


# ─────────────────────────────────────────────────────────────────────────────
# Literature
# ─────────────────────────────────────────────────────────────────────────────

class TestLiteratureEndpoints:

    def test_list_literature_200(self, client, mock_repo, auth_headers):
        mock_repo.list_literature_analyses.return_value = [
            {"id": ANALYSIS_ID, "status": "completed",
                "review_content": "Review text..."}
        ]
        resp = client.get(
            f"/projects/{TEST_PROJECT_ID}/literature", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_get_literature_with_citations(self, client, mock_repo, auth_headers):
        mock_repo.get_literature_analysis.return_value = {
            "id": ANALYSIS_ID, "project_id": TEST_PROJECT_ID,
            "review_content": "Full review text", "status": "completed",
        }
        mock_repo.get_literature_citations.return_value = [
            {"id": str(uuid.uuid4()),
             "formatted_citation": "Author, A. (2024). Title."}
        ]
        resp = client.get(f"/literature/{ANALYSIS_ID}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "citations" in data
        assert len(data["citations"]) == 1

    def test_get_literature_404_when_missing(self, client, mock_repo, auth_headers):
        mock_repo.get_literature_analysis.return_value = None
        resp = client.get(f"/literature/{uuid.uuid4()}", headers=auth_headers)
        assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Suggestions
# ─────────────────────────────────────────────────────────────────────────────

class TestSuggestionEndpoints:

    def test_list_suggestions_200(self, client, mock_repo, auth_headers):
        from tests.conftest import TEST_USER_ID
        mock_repo.get_section.return_value = make_section()
        # user_id in project MUST match what mock_jwt injects as the JWT sub
        mock_repo.get_project.return_value = make_project(user_id=TEST_USER_ID)
        mock_repo.list_suggestions.return_value = [make_suggestion()]
        resp = client.get(
            f"/sections/{TEST_SECTION_ID}/suggestions", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_update_suggestion_accept(self, client, mock_repo, auth_headers):
        accepted = {**make_suggestion(), "status": "accepted"}
        mock_repo.update_suggestion.return_value = accepted
        resp = client.patch(
            f"/suggestions/{SUGGESTION_ID}",
            json={"status": "accepted"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        mock_repo.update_suggestion.assert_called_once_with(
            suggestion_id=SUGGESTION_ID,
            status="accepted",
            feedback="",
        )

    def test_update_suggestion_reject_with_feedback(self, client, mock_repo, auth_headers):
        rejected = {**make_suggestion(), "status": "rejected",
                    "feedback": "Off topic"}
        mock_repo.update_suggestion.return_value = rejected
        resp = client.patch(
            f"/suggestions/{SUGGESTION_ID}",
            json={"status": "rejected", "feedback": "Off topic"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        mock_repo.update_suggestion.assert_called_once_with(
            suggestion_id=SUGGESTION_ID,
            status="rejected",
            feedback="Off topic",
        )

    def test_update_suggestion_invalid_status(self, client, auth_headers):
        resp = client.patch(
            f"/suggestions/{SUGGESTION_ID}",
            json={"status": "pending"},   # can't set back to pending
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_inline_suggestion(self, client, mock_repo, auth_headers):
        """
        inline_suggestion is an async endpoint that calls run_in_executor.
        We patch the executor at the asyncio level so no real LLM is called.
        run_in_executor returns a coroutine — we replace it with AsyncMock.
        """
        from unittest.mock import AsyncMock
        from tests.conftest import TEST_USER_ID

        mock_repo.get_section.return_value = make_section()
        mock_repo.get_project.return_value = make_project(user_id=TEST_USER_ID)
        mock_repo.get_preferences.return_value = {
            "writing_style": "formal", "tone": "academic",
            "target_journal": "", "language": "English",
            "citation_style": "APA", "grounded_only": False, "llm_provider": "groq",
        }

        fake_result = {
            "original": "The model works well.",
            "suggestion": "The proposed model demonstrates strong performance.",
            "mode": "improve",
            "diff": [],
        }

        # Patch run_in_executor on the event loop to return the fake result directly.
        # AsyncMock makes it awaitable so the `await loop.run_in_executor(...)` works.
        mock_loop = MagicMock()
        mock_loop.run_in_executor = AsyncMock(return_value=fake_result)

        with patch("api.routes.data.asyncio") as mock_asyncio:
            mock_asyncio.get_event_loop.return_value = mock_loop
            resp = client.post(
                f"/sections/{TEST_SECTION_ID}/suggestions/inline",
                json={
                    "document": "The model works well.",
                    "target_text": "The model works well.",
                },
                headers=auth_headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["suggestion"] == "The proposed model demonstrates strong performance."
        assert data["mode"] == "improve"


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualizationEndpoints:

    def test_list_visualizations_200(self, client, mock_repo, auth_headers):
        mock_repo.list_visualizations.return_value = [make_visualization()]
        resp = client.get(
            f"/projects/{TEST_PROJECT_ID}/visualizations", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_get_visualization_200(self, client, mock_repo, auth_headers):
        mock_repo.get_visualization.return_value = make_visualization()
        resp = client.get(f"/visualizations/{VIZ_ID}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["type"] == "chart"

    def test_get_visualization_404(self, client, mock_repo, auth_headers):
        mock_repo.get_visualization.return_value = None
        resp = client.get(
            f"/visualizations/{uuid.uuid4()}", headers=auth_headers)
        assert resp.status_code == 404

    def test_download_visualization_404_when_file_missing(self, client, mock_repo, auth_headers):
        viz = {**make_visualization(), "file_path": "/nonexistent/file.png"}
        mock_repo.get_visualization.return_value = viz
        resp = client.get(
            f"/visualizations/{VIZ_ID}/download", headers=auth_headers)
        assert resp.status_code == 404

    def test_download_visualization_serves_file(self, client, mock_repo, auth_headers, tmp_path):
        test_file = tmp_path / "chart.png"
        test_file.write_bytes(b"\x89PNG\r\n\x1a\n")   # minimal PNG header
        viz = {**make_visualization(), "file_path": str(test_file)}
        mock_repo.get_visualization.return_value = viz
        resp = client.get(
            f"/visualizations/{VIZ_ID}/download", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_delete_visualization_204(self, client, mock_repo, auth_headers):
        resp = client.delete(f"/visualizations/{VIZ_ID}", headers=auth_headers)
        assert resp.status_code == 204
        mock_repo.delete_visualization.assert_called_once_with(VIZ_ID)


# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────

class TestChatEndpoints:

    def test_get_chat_history_200(self, client, mock_repo, auth_headers):
        mock_repo.list_chat_messages.return_value = [
            {"role": "human", "content": "Hi", "created_at": "2024-01-01T00:00:00"},
            {"role": "ai",    "content": "Hello!",
                "created_at": "2024-01-01T00:00:01"},
        ]
        resp = client.get(
            f"/projects/{TEST_PROJECT_ID}/chat", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_get_chat_empty_for_new_project(self, client, mock_repo, auth_headers):
        mock_repo.list_chat_messages.return_value = []
        resp = client.get(
            f"/projects/{TEST_PROJECT_ID}/chat", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_clear_chat_204(self, client, mock_repo, auth_headers):
        resp = client.delete(
            f"/projects/{TEST_PROJECT_ID}/chat", headers=auth_headers)
        assert resp.status_code == 204
        mock_repo.clear_chat_messages.assert_called_once_with(TEST_PROJECT_ID)
