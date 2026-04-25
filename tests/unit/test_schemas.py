"""
tests/unit/test_schemas.py
───────────────────────────
Unit tests for api/schemas/requests.py

Tests that the Pydantic models accept valid input and reject invalid input
correctly before it reaches any route handler.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas.requests import (
    ProjectCreate,
    ProjectUpdate,
    PreferencesUpdate,
    SectionCreate,
    RunRequest,
    ResumeRequest,
    SuggestionUpdate,
)


# ─────────────────────────────────────────────────────────────────────────────
# ProjectCreate
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectCreate:

    def test_valid(self):
        p = ProjectCreate(title="My Paper")
        assert p.title == "My Paper"
        assert p.description == ""

    def test_empty_title_rejected(self):
        with pytest.raises(ValidationError):
            ProjectCreate(title="")

    def test_description_optional(self):
        p = ProjectCreate(title="T", description="Desc")
        assert p.description == "Desc"


# ─────────────────────────────────────────────────────────────────────────────
# SectionCreate
# ─────────────────────────────────────────────────────────────────────────────

class TestSectionCreate:

    def test_valid_type(self):
        s = SectionCreate(type="introduction")
        assert s.type == "introduction"
        assert s.position == 1

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            SectionCreate(type="bibliography")   # not in enum

    def test_position_must_be_positive(self):
        with pytest.raises(ValidationError):
            SectionCreate(type="abstract", position=0)

    def test_all_valid_types_accepted(self):
        valid_types = [
            "abstract", "introduction", "methodology",
            "results", "discussion", "conclusion", "other"
        ]
        for t in valid_types:
            s = SectionCreate(type=t)
            assert s.type == t


# ─────────────────────────────────────────────────────────────────────────────
# PreferencesUpdate
# ─────────────────────────────────────────────────────────────────────────────

class TestPreferencesUpdate:

    def test_all_optional(self):
        p = PreferencesUpdate()
        assert p.writing_style is None
        assert p.grounded_only is None

    def test_invalid_llm_provider_rejected(self):
        with pytest.raises(ValidationError):
            PreferencesUpdate(llm_provider="openai")   # not in enum

    def test_invalid_assistance_level_rejected(self):
        with pytest.raises(ValidationError):
            PreferencesUpdate(assistance_level="extreme")

    def test_grounded_only_bool(self):
        p = PreferencesUpdate(grounded_only=True)
        assert p.grounded_only is True


# ─────────────────────────────────────────────────────────────────────────────
# RunRequest / ResumeRequest
# ─────────────────────────────────────────────────────────────────────────────

class TestRunRequest:

    def test_valid(self):
        r = RunRequest(user_message="Write an introduction")
        assert r.user_message == "Write an introduction"
        assert r.section_id is None

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            RunRequest(user_message="")

    def test_section_id_optional(self):
        r = RunRequest(user_message="Hello", section_id="abc-123")
        assert r.section_id == "abc-123"


class TestResumeRequest:

    def test_approve(self):
        r = ResumeRequest(hitl_action="approve")
        assert r.hitl_action == "approve"
        assert r.hitl_feedback is None
        assert r.human_edited_text is None

    def test_edit(self):
        r = ResumeRequest(hitl_action="edit", human_edited_text="Fixed text")
        assert r.human_edited_text == "Fixed text"

    def test_invalid_action_rejected(self):
        with pytest.raises(ValidationError):
            ResumeRequest(hitl_action="maybe")

    def test_all_valid_actions(self):
        for action in ("approve", "edit", "reject", "regenerate"):
            r = ResumeRequest(hitl_action=action)
            assert r.hitl_action == action


# ─────────────────────────────────────────────────────────────────────────────
# SuggestionUpdate
# ─────────────────────────────────────────────────────────────────────────────

class TestSuggestionUpdate:

    def test_valid_statuses(self):
        for status in ("accepted", "rejected", "edited"):
            s = SuggestionUpdate(status=status)
            assert s.status == status

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            # can only set terminal statuses
            SuggestionUpdate(status="pending")

    def test_feedback_optional(self):
        s = SuggestionUpdate(status="rejected", feedback="Not relevant")
        assert s.feedback == "Not relevant"
