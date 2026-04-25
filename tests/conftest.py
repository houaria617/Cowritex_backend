"""
tests/conftest.py
──────────────────
Shared pytest fixtures for unit and integration tests.

Strategy
────────
- We NEVER hit real Supabase or real LLMs in unit tests.
- All external calls (repo, graph, JWT) are mocked via pytest-mock / unittest.mock.
- Integration tests spin up the real FastAPI app with TestClient but still
  mock the DB layer — so they test routing, auth, and request/response shapes
  without needing a live Supabase instance.
- A separate section at the bottom documents how to run TRUE end-to-end tests
  against a real Supabase dev project (opt-in, skipped by default).
"""

from __future__ import annotations

import uuid
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Stable test IDs — FIXED constants, never random ──────────────────────────
# uuid.uuid4() at module level regenerates on every import/reload, causing
# user_id mismatches between mock_jwt (returns TEST_USER_ID as JWT sub) and
# make_project (embeds TEST_USER_ID as user_id). Pinned strings fix this.
TEST_USER_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
TEST_PROJECT_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
TEST_SECTION_ID = "cccccccc-cccc-cccc-cccc-cccccccccccc"
TEST_THREAD_ID = "dddddddd-dddd-dddd-dddd-dddddddddddd"

# ── Fake JWT that passes our verifier ─────────────────────────────────────────
# We patch jwt.decode so the actual secret doesn't matter in tests.
FAKE_TOKEN = "fake.jwt.token"
FAKE_BEARER = f"Bearer {FAKE_TOKEN}"


# ─────────────────────────────────────────────────────────────────────────────
# Fake data factories
# ─────────────────────────────────────────────────────────────────────────────

def make_project(project_id: str = TEST_PROJECT_ID, user_id: str = TEST_USER_ID) -> dict:
    return {
        "id":          project_id,
        "user_id":     user_id,
        "title":       "Test Project",
        "description": "A test research project",
        "status":      "active",
        "thread_id":   None,
        "progress":    0,
        "created_at":  "2024-01-01T00:00:00+00:00",
        "updated_at":  "2024-01-01T00:00:00+00:00",
    }


def make_section(section_id: str = TEST_SECTION_ID) -> dict:
    return {
        "id":         section_id,
        "project_id": TEST_PROJECT_ID,
        "type":       "introduction",
        "title":      "Introduction",
        "position":   1,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }


def make_preferences() -> dict:
    return {
        "writing_style":    "formal",
        "tone":             "academic",
        "target_journal":   "",
        "language":         "English",
        "citation_style":   "APA",
        "grounded_only":    False,
        "llm_provider":     "groq",
        "assistance_level": "moderate",
    }


def make_suggestion(section_id: str = TEST_SECTION_ID) -> dict:
    return {
        "id":             str(uuid.uuid4()),
        "section_id":     section_id,
        "original_text":  "Old text",
        "suggested_text": "Improved text from AI",
        "instruction":    "Improve this paragraph",
        "status":         "pending",
        "feedback":       None,
        "created_at":     "2024-01-01T00:00:00+00:00",
        "resolved_at":    None,
    }


def make_visualization(project_id: str = TEST_PROJECT_ID) -> dict:
    return {
        "id":           str(uuid.uuid4()),
        "project_id":   project_id,
        "section_id":   None,
        "type":         "chart",
        "title":        "Test Chart",
        "raw_data":     {"x": ["A", "B"], "y": [1, 2]},
        "config":       {"type": "bar"},
        "file_path":    "/tmp/test_chart.png",
        "export_format": "png",
        "color_scheme": "default",
        "details":      "",
        "created_at":   "2024-01-01T00:00:00+00:00",
        "updated_at":   "2024-01-01T00:00:00+00:00",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_jwt(monkeypatch):
    """
    Patch jwt.decode at the exact location api/dependencies.py imports it.

    dependencies.py does:  import jwt
    So the name 'jwt' inside that module points to the jwt module object.
    monkeypatch.setattr(deps_module, "jwt", fake) replaces that exact reference.

    We MUST keep the real exception classes on the fake so that:
        except jwt.ExpiredSignatureError   →  still catches real ExpiredSignatureError
        except jwt.InvalidTokenError       →  still catches real InvalidTokenError
    """
    import api.dependencies as deps_module
    import jwt as real_jwt

    fake_jwt = MagicMock()
    # decode always succeeds and returns TEST_USER_ID as sub
    fake_jwt.decode = MagicMock(
        return_value={"sub": TEST_USER_ID, "role": "authenticated"}
    )
    # Preserve real exception classes — the except clauses in get_current_user
    # reference jwt.ExpiredSignatureError and jwt.InvalidTokenError by name
    fake_jwt.ExpiredSignatureError = real_jwt.ExpiredSignatureError
    fake_jwt.InvalidTokenError = real_jwt.InvalidTokenError

    monkeypatch.setattr(deps_module, "jwt", fake_jwt)
    return TEST_USER_ID


@pytest.fixture
def mock_repo(monkeypatch):
    """
    Replace every function in database.repository with a MagicMock.
    Tests set return values on the specific functions they need.
    """
    import database.repository as repo_module

    mock = MagicMock()

    # Set sensible defaults so tests that don't care still pass
    mock.get_project.return_value = make_project()
    mock.list_projects.return_value = [make_project()]
    mock.create_project.return_value = make_project()
    mock.update_project.return_value = make_project()
    mock.get_preferences.return_value = make_preferences()
    mock.upsert_preferences.return_value = make_preferences()
    mock.get_section.return_value = make_section()
    mock.get_project_sections.return_value = [make_section()]
    mock.create_section.return_value = make_section()
    mock.update_section.return_value = make_section()
    mock.get_current_content.return_value = "Existing section content."
    mock.get_section_versions.return_value = []
    mock.save_new_version.return_value = {
        "id": str(uuid.uuid4()), "version_number": 1}
    mock.restore_version.return_value = {
        "id": str(uuid.uuid4()), "version_number": 2}
    mock.list_suggestions.return_value = [make_suggestion()]
    mock.update_suggestion.return_value = make_suggestion()
    mock.list_sources.return_value = []
    mock.list_literature_analyses.return_value = []
    mock.get_literature_analysis.return_value = None
    mock.list_visualizations.return_value = [make_visualization()]
    mock.get_visualization.return_value = make_visualization()
    mock.list_chat_messages.return_value = []
    mock.save_thread_id.return_value = None
    mock.get_thread_id.return_value = TEST_THREAD_ID
    mock.get_user.return_value = {
        "id": TEST_USER_ID, "email": "test@example.com",
        "full_name": "Test User", "academic_position": "PhD student",
        "organization": "Test University", "field_interests": "AI",
    }

    # Patch every function that routes import from repo
    for attr in dir(repo_module):
        if not attr.startswith("_"):
            try:
                monkeypatch.setattr(repo_module, attr, getattr(mock, attr))
            except (AttributeError, TypeError):
                pass

    return mock


@pytest.fixture
def client(mock_jwt, mock_repo) -> Generator:
    """
    TestClient with mocked JWT + mocked DB.
    Use this for all integration tests.
    """
    # Patch settings so client.py doesn't fail on import
    with patch("config.settings.Settings.model_validate", return_value=MagicMock(
        SUPABASE_URL="https://fake.supabase.co",
        SUPABASE_SERVICE_KEY="fake-key",
        SUPABASE_JWT_SECRET="fake-secret",
        GROQ_API_KEY="fake-groq",
        GROQ_MODEL="llama-3.3-70b-versatile",
        GOOGLE_API_KEY="",
        GEMINI_MODEL="gemini-2.0-flash",
        SEMANTIC_SCHOLAR_API_KEY="",
        APP_ENV="test",
        cors_origins_list=["http://localhost:3000"],
    )):
        from api.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def auth_headers() -> dict:
    return {"Authorization": FAKE_BEARER}
