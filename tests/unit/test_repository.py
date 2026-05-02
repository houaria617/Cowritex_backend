"""
tests/unit/test_repository.py
───────────────────────────────
Unit tests for database/repository.py

The Supabase client (db) is fully mocked — these tests verify
the LOGIC of each repo function (field mapping, dedup, version numbering)
without touching any real database.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, call, patch

import pytest

PROJECT_ID = str(uuid.uuid4())
SECTION_ID = str(uuid.uuid4())
USER_ID = str(uuid.uuid4())
ANALYSIS_ID = str(uuid.uuid4())


@pytest.fixture(autouse=True)
def mock_db():
    """Replace database.client.db with a full MagicMock."""
    mock = MagicMock()
    with patch("database.client.db", mock), \
            patch("database.repository.db", mock):
        yield mock


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chain(mock_db, return_value):
    """Make mock_db.table().select()...execute() return a given value."""
    execute = MagicMock()
    execute.execute.return_value = MagicMock(data=return_value)
    mock_db.table.return_value.select.return_value = execute
    mock_db.table.return_value.select.return_value.eq.return_value = execute
    mock_db.table.return_value.select.return_value.eq.return_value.eq.return_value = execute
    mock_db.table.return_value.select.return_value.eq.return_value.single.return_value = execute
    mock_db.table.return_value.select.return_value.eq.return_value.maybe_single.return_value = execute
    mock_db.table.return_value.select.return_value.eq.return_value.order.return_value = execute
    mock_db.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value = execute
    mock_db.table.return_value.select.return_value.eq.return_value.order.return_value.desc.return_value = execute
    return execute


# ─────────────────────────────────────────────────────────────────────────────
# Projects
# ─────────────────────────────────────────────────────────────────────────────

class TestProjects:

    def test_create_project_inserts_row_and_prefs(self, mock_db):
        from database.repository import create_project
        project_row = {"id": PROJECT_ID, "user_id": USER_ID, "title": "T"}
        mock_db.table.return_value.insert.return_value.execute.return_value = \
            MagicMock(data=[project_row])
        result = create_project(USER_ID, "T", "D")
        assert result["id"] == PROJECT_ID
        # Should have called table() at least twice: projects + project_preferences
        assert mock_db.table.call_count >= 2

    def test_list_projects_filters_by_user(self, mock_db):
        from database.repository import list_projects
        rows = [{"id": PROJECT_ID, "user_id": USER_ID}]
        (mock_db.table.return_value
         .select.return_value
         .eq.return_value
         .order.return_value
         .execute.return_value) = MagicMock(data=rows)
        result = list_projects(USER_ID)
        assert result == rows

    def test_update_project_progress_clamps(self, mock_db):
        from database.repository import update_project_progress
        mock_db.table.return_value.update.return_value.eq.return_value.execute.return_value = \
            MagicMock(data=[{}])
        update_project_progress(PROJECT_ID, 75)
        mock_db.table.return_value.update.assert_called_with({"progress": 75})

    def test_save_thread_id(self, mock_db):
        from database.repository import save_thread_id
        mock_db.table.return_value.update.return_value.eq.return_value.execute.return_value = \
            MagicMock(data=[{}])
        save_thread_id(PROJECT_ID, "thread-abc")
        mock_db.table.return_value.update.assert_called_with(
            {"thread_id": "thread-abc"})


# ─────────────────────────────────────────────────────────────────────────────
# Preferences
# ─────────────────────────────────────────────────────────────────────────────

class TestPreferences:

    def test_get_preferences_returns_defaults_when_no_row(self, mock_db):
        from database.repository import get_preferences
        (mock_db.table.return_value
         .select.return_value
         .eq.return_value
         .maybe_single.return_value
         .execute.return_value) = MagicMock(data=None)
        result = get_preferences(PROJECT_ID)
        assert result["writing_style"] == "formal"
        assert result["llm_provider"] == "groq"
        assert result["grounded_only"] is False

    def test_get_preferences_maps_all_fields(self, mock_db):
        from database.repository import get_preferences
        row = {
            "writing_style": "academic", "tone": "neutral",
            "target_journal": "IEEE", "language": "French",
            "citation_style": "IEEE", "grounded_only": True,
            "llm_provider": "gemini", "assistance_level": "full",
        }
        (mock_db.table.return_value
         .select.return_value
         .eq.return_value
         .maybe_single.return_value
         .execute.return_value) = MagicMock(data=row)
        result = get_preferences(PROJECT_ID)
        assert result["language"] == "French"
        assert result["grounded_only"] is True
        assert result["llm_provider"] == "gemini"


# ─────────────────────────────────────────────────────────────────────────────
# Document versions
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentVersions:

    def test_save_new_version_increments_version_number(self, mock_db):
        from database.repository import save_new_version

        # Simulate existing max version = 3
        (mock_db.table.return_value
         .select.return_value
         .eq.return_value
         .order.return_value
         .limit.return_value
         .execute.return_value) = MagicMock(data=[{"version_number": 3}])

        inserted = {"id": str(uuid.uuid4()),
                    "version_number": 4, "is_current": True}
        mock_db.table.return_value.insert.return_value.execute.return_value = \
            MagicMock(data=[inserted])
        mock_db.table.return_value.update.return_value.eq.return_value.execute.return_value = \
            MagicMock(data=[])

        result = save_new_version(SECTION_ID, "New content", "ai")
        assert result["version_number"] == 4

    def test_save_new_version_unsets_previous_current(self, mock_db):
        from database.repository import save_new_version

        (mock_db.table.return_value
         .select.return_value
         .eq.return_value
         .order.return_value
         .limit.return_value
         .execute.return_value) = MagicMock(data=[])

        mock_db.table.return_value.insert.return_value.execute.return_value = \
            MagicMock(data=[{"id": "v1", "version_number": 1}])
        mock_db.table.return_value.update.return_value.eq.return_value.execute.return_value = \
            MagicMock(data=[])

        save_new_version(SECTION_ID, "Content", "human")
        # update({"is_current": False}) must have been called
        mock_db.table.return_value.update.assert_any_call(
            {"is_current": False})

    def test_get_current_content_returns_none_when_missing(self, mock_db):
        from database.repository import get_current_content
        (mock_db.table.return_value
         .select.return_value
         .eq.return_value
         .eq.return_value
         .maybe_single.return_value
         .execute.return_value) = MagicMock(data=None)
        result = get_current_content(SECTION_ID)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────────────────────

class TestSources:

    def test_save_sources_deduplicates_by_title(self, mock_db):
        from database.repository import save_sources

        # Pretend "existing paper" already in DB
        (mock_db.table.return_value
         .select.return_value
         .eq.return_value
         .execute.return_value) = MagicMock(data=[{"title": "existing paper"}])

        mock_db.table.return_value.insert.return_value.execute.return_value = \
            MagicMock(data=[{"id": "new-id"}])

        sources = [
            # dup — should be skipped
            {"title": "Existing Paper", "authors": "A"},
            # new — should be inserted
            {"title": "Brand New Paper", "authors": "B"},
        ]
        ids = save_sources(PROJECT_ID, sources)
        assert ids == ["new-id"]

    def test_save_sources_returns_empty_for_empty_input(self, mock_db):
        from database.repository import save_sources
        result = save_sources(PROJECT_ID, [])
        assert result == []
        mock_db.table.return_value.insert.assert_not_called()

    def test_save_sources_skips_rows_without_title(self, mock_db):
        from database.repository import save_sources
        (mock_db.table.return_value
         .select.return_value
         .eq.return_value
         .execute.return_value) = MagicMock(data=[])
        mock_db.table.return_value.insert.return_value.execute.return_value = \
            MagicMock(data=[])
        save_sources(PROJECT_ID, [{"title": "", "authors": "X"}])
        mock_db.table.return_value.insert.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Chat messages
# ─────────────────────────────────────────────────────────────────────────────

class TestChatMessages:

    def test_get_recent_messages_returns_chronological_order(self, mock_db):
        from database.repository import get_recent_messages
        # DB returns newest first (desc), repo must flip
        rows = [
            {"role": "ai",    "content": "Second",
                "created_at": "2024-01-01T00:00:02"},
            {"role": "human", "content": "First",
                "created_at": "2024-01-01T00:00:01"},
        ]
        (mock_db.table.return_value
         .select.return_value
         .eq.return_value
         .order.return_value
         .limit.return_value
         .execute.return_value) = MagicMock(data=rows)
        result = get_recent_messages(PROJECT_ID, limit=10)
        assert result[0]["content"] == "First"
        assert result[1]["content"] == "Second"

    def test_save_message_inserts_correct_role(self, mock_db):
        from database.repository import save_message
        mock_db.table.return_value.insert.return_value.execute.return_value = \
            MagicMock(data=[{"id": "msg-1", "role": "human", "content": "Hi"}])
        result = save_message(PROJECT_ID, "human", "Hi")
        assert result["role"] == "human"
