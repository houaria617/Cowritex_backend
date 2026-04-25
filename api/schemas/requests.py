"""
api/schemas/requests.py
────────────────────────
All Pydantic models for request bodies and API responses.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, EmailStr, Field


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────

class UserProfileUpdate(BaseModel):
    full_name:          Optional[str] = None
    academic_position:  Optional[str] = None
    organization:       Optional[str] = None
    field_interests:    Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Projects
# ─────────────────────────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    title:       str = Field(..., min_length=1, max_length=200)
    description: str = ""


class ProjectUpdate(BaseModel):
    title:       Optional[str] = None
    description: Optional[str] = None
    status:      Optional[Literal["active", "archived"]] = None


class PreferencesUpdate(BaseModel):
    writing_style:    Optional[str] = None
    tone:             Optional[str] = None
    target_journal:   Optional[str] = None
    language:         Optional[str] = None
    assistance_level: Optional[Literal["light", "moderate", "full"]] = None
    citation_style:   Optional[str] = None
    grounded_only:    Optional[bool] = None
    llm_provider:     Optional[Literal["groq", "gemini", "anthropic"]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Sections
# ─────────────────────────────────────────────────────────────────────────────

class SectionCreate(BaseModel):
    type:     Literal[
        "abstract", "introduction", "methodology",
        "results", "discussion", "conclusion", "other"
    ]
    title:    str = ""
    position: int = Field(default=1, ge=1)


class SectionUpdate(BaseModel):
    type:     Optional[str] = None
    title:    Optional[str] = None
    position: Optional[int] = None


class VersionRestore(BaseModel):
    version_id: str


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator  (the core: run + resume)
# ─────────────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    """
    Triggers a new graph run.

    user_message  — the researcher's raw message (classified by intent_classifier)
    section_id    — optional, scopes the run to a document section
    """
    user_message: str = Field(..., min_length=1)
    section_id:   Optional[str] = None


class RunResponse(BaseModel):
    """
    Returned immediately after the graph hits the HITL interrupt.

    thread_id    — must be sent back with the /resume call
    agent_output — the AI-generated content awaiting researcher review
    intents      — what the classifier detected (useful for frontend display)
    status       — always "pending_review" at this point
    """
    thread_id:    str
    agent_output: Optional[str]
    intents:      list[str]
    status:       str = "pending_review"


class ResumeRequest(BaseModel):
    """
    Researcher's HITL decision.

    hitl_action        — what the researcher chose
    hitl_feedback      — optional note for reject/regenerate
    human_edited_text  — required when hitl_action == "edit"
    """
    hitl_action:       Literal["approve", "edit", "reject", "regenerate"]
    hitl_feedback:     Optional[str] = None
    human_edited_text: Optional[str] = None


class ResumeResponse(BaseModel):
    thread_id:    str
    agent_output: Optional[str]
    # "completed" | "pending_review" (regenerate loops back)
    status:       str


class StatusResponse(BaseModel):
    thread_id: str
    status:    str
    intents:   list[str]
    last_agent: Optional[str]
    error:     Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Inline suggestion  (copilot-style, bypasses the main graph)
# ─────────────────────────────────────────────────────────────────────────────

class InlineSuggestionRequest(BaseModel):
    document:    str = ""
    target_text: str = ""
    section_id:  Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Suggestions (HITL artifacts)
# ─────────────────────────────────────────────────────────────────────────────

class SuggestionUpdate(BaseModel):
    status:   Literal["accepted", "rejected", "edited"]
    feedback: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Generic response wrappers
# ─────────────────────────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str


class IDResponse(BaseModel):
    id: str
