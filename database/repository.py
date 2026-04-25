"""
database/repository.py
───────────────────────
All database operations for CoWriteX.

Every function used across the orchestrator nodes is implemented here:
    search_node      → save_sources
    literature_node  → save_literature_analysis, save_sources, save_literature_citations
    writing_node     → get_current_content, get_project_sections, create_suggestion
    visualisation_node → save_visualization
    chat_node        → get_recent_messages, save_message
    persist_node     → get_pending_suggestion, save_new_version,
                       save_message, update_project_progress
    edit_node        → save_new_version
    intent_classifier → get_preferences

Plus CRUD helpers used by the FastAPI routes.

Convention:
    - All functions are synchronous (nodes run inside run_in_executor).
    - Every function catches nothing — callers wrap in try/except.
    - UUIDs are passed and returned as plain strings.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .client import db

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Users
# ─────────────────────────────────────────────────────────────────────────────

def get_user(user_id: str) -> dict | None:
    res = db.table("users").select("*").eq("id", user_id).single().execute()
    return res.data


def get_user_by_email(email: str) -> dict | None:
    res = db.table("users").select(
        "*").eq("email", email).maybe_single().execute()
    return res.data


def create_user(
    email: str,
    full_name: str,
    password_hash: str,
    academic_position: str = "",
    organization: str = "",
    field_interests: str = "",
) -> dict:
    payload = {
        "email":             email,
        "full_name":         full_name,
        "password_hash":     password_hash,
        "academic_position": academic_position,
        "organization":      organization,
        "field_interests":   field_interests,
    }
    res = db.table("users").insert(payload).execute()
    return res.data[0]


def update_user(user_id: str, updates: dict) -> dict:
    res = (
        db.table("users")
        .update(updates)
        .eq("id", user_id)
        .execute()
    )
    return res.data[0]


# ─────────────────────────────────────────────────────────────────────────────
# Projects
# ─────────────────────────────────────────────────────────────────────────────

def create_project(user_id: str, title: str, description: str = "") -> dict:
    res = db.table("projects").insert({
        "user_id":     user_id,
        "title":       title,
        "description": description,
    }).execute()
    project = res.data[0]

    # Auto-create default preferences row
    db.table("project_preferences").insert({
        "project_id": project["id"],
    }).execute()

    return project


def get_project(project_id: str) -> dict | None:
    res = (
        db.table("projects")
        .select("*")
        .eq("id", project_id)
        .single()
        .execute()
    )
    return res.data


def list_projects(user_id: str) -> list[dict]:
    res = (
        db.table("projects")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []


def update_project(project_id: str, updates: dict) -> dict:
    res = (
        db.table("projects")
        .update(updates)
        .eq("id", project_id)
        .execute()
    )
    return res.data[0]


def delete_project(project_id: str) -> None:
    db.table("projects").delete().eq("id", project_id).execute()


def update_project_progress(project_id: str, progress: int) -> None:
    db.table("projects").update({"progress": progress}).eq(
        "id", project_id).execute()


def save_thread_id(project_id: str, thread_id: str) -> None:
    """Persist the LangGraph thread_id so the frontend can resume HITL."""
    db.table("projects").update({"thread_id": thread_id}).eq(
        "id", project_id).execute()


def get_thread_id(project_id: str) -> str | None:
    res = (
        db.table("projects")
        .select("thread_id")
        .eq("id", project_id)
        .single()
        .execute()
    )
    return (res.data or {}).get("thread_id")


# ─────────────────────────────────────────────────────────────────────────────
# Project preferences
# ─────────────────────────────────────────────────────────────────────────────

def get_preferences(project_id: str) -> dict:
    """
    Returns the preferences dict expected by the orchestrator nodes.
    Falls back to defaults if no row exists yet.
    """
    res = (
        db.table("project_preferences")
        .select("*")
        .eq("project_id", project_id)
        .maybe_single()
        .execute()
    )
    if not res.data:
        return {
            "writing_style":  "formal",
            "tone":           "academic",
            "target_journal": "",
            "language":       "English",
            "citation_style": "APA",
            "grounded_only":  False,
            "llm_provider":   "groq",
        }
    row = res.data
    return {
        "writing_style":  row.get("writing_style",  "formal"),
        "tone":           row.get("tone",           "academic"),
        "target_journal": row.get("target_journal", ""),
        "language":       row.get("language",       "English"),
        "citation_style": row.get("citation_style", "APA"),
        "grounded_only":  row.get("grounded_only",  False),
        "llm_provider":   row.get("llm_provider",   "groq"),
        "assistance_level": row.get("assistance_level", "moderate"),
    }


def upsert_preferences(project_id: str, updates: dict) -> dict:
    """Create or update preferences row."""
    existing = (
        db.table("project_preferences")
        .select("id")
        .eq("project_id", project_id)
        .maybe_single()
        .execute()
    )
    if existing.data:
        res = (
            db.table("project_preferences")
            .update(updates)
            .eq("project_id", project_id)
            .execute()
        )
    else:
        res = (
            db.table("project_preferences")
            .insert({"project_id": project_id, **updates})
            .execute()
        )
    return res.data[0]


# ─────────────────────────────────────────────────────────────────────────────
# Sections
# ─────────────────────────────────────────────────────────────────────────────

def create_section(
    project_id: str,
    section_type: str,
    title: str = "",
    position: int = 1,
) -> dict:
    res = db.table("sections").insert({
        "project_id": project_id,
        "type":       section_type,
        "title":      title,
        "position":   position,
    }).execute()
    return res.data[0]


def get_section(section_id: str) -> dict | None:
    res = (
        db.table("sections")
        .select("*")
        .eq("id", section_id)
        .single()
        .execute()
    )
    return res.data


def get_project_sections(project_id: str) -> list[dict]:
    """
    Returns sections ordered by position, with current content injected
    (used by writing_node for surrounding-section context).
    """
    res = (
        db.table("sections")
        .select("*")
        .eq("project_id", project_id)
        .order("position")
        .execute()
    )
    sections = res.data or []
    for sec in sections:
        sec["content"] = get_current_content(sec["id"]) or ""
    return sections


def update_section(section_id: str, updates: dict) -> dict:
    res = (
        db.table("sections")
        .update(updates)
        .eq("id", section_id)
        .execute()
    )
    return res.data[0]


def delete_section(section_id: str) -> None:
    db.table("sections").delete().eq("id", section_id).execute()


# ─────────────────────────────────────────────────────────────────────────────
# Document versions
# ─────────────────────────────────────────────────────────────────────────────

def get_current_content(section_id: str) -> str | None:
    """Return the text of the current (is_current=True) version, or None."""
    res = (
        db.table("document_versions")
        .select("content")
        .eq("section_id", section_id)
        .eq("is_current", True)
        .maybe_single()
        .execute()
    )
    return (res.data or {}).get("content")


def get_section_versions(section_id: str) -> list[dict]:
    res = (
        db.table("document_versions")
        .select("*")
        .eq("section_id", section_id)
        .order("version_number", desc=True)
        .execute()
    )
    return res.data or []


def save_new_version(
    section_id: str,
    content: str,
    author_type: str,          # "ai" | "human"
    suggestion_id: str | None = None,
) -> dict:
    """
    Mark all existing versions as not current, then insert the new one.
    version_number auto-increments from the current max.
    """
    # Unset is_current on all previous versions
    db.table("document_versions").update({"is_current": False}).eq(
        "section_id", section_id
    ).execute()

    # Compute next version number
    res = (
        db.table("document_versions")
        .select("version_number")
        .eq("section_id", section_id)
        .order("version_number", desc=True)
        .limit(1)
        .execute()
    )
    last_num = (res.data[0]["version_number"] if res.data else 0)

    row = db.table("document_versions").insert({
        "section_id":     section_id,
        "content":        content,
        "author_type":    author_type,
        "suggestion_id":  suggestion_id,
        "version_number": last_num + 1,
        "is_current":     True,
    }).execute()
    return row.data[0]


def restore_version(version_id: str, section_id: str) -> dict:
    """Make a previous version current again."""
    res = (
        db.table("document_versions")
        .select("content, author_type")
        .eq("id", version_id)
        .single()
        .execute()
    )
    row = res.data
    return save_new_version(
        section_id=section_id,
        content=row["content"],
        author_type=row["author_type"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# AI Suggestions
# ─────────────────────────────────────────────────────────────────────────────

def create_suggestion(
    section_id: str,
    suggested_text: str,
    instruction: str = "",
    original_text: str | None = None,
) -> dict:
    res = db.table("ai_suggestions").insert({
        "section_id":     section_id,
        "suggested_text": suggested_text,
        "instruction":    instruction,
        "original_text":  original_text,
        "status":         "pending",
    }).execute()
    return res.data[0]


def get_pending_suggestion(section_id: str) -> dict | None:
    """Return the most recent pending suggestion for a section."""
    res = (
        db.table("ai_suggestions")
        .select("*")
        .eq("section_id", section_id)
        .eq("status", "pending")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def list_suggestions(section_id: str) -> list[dict]:
    res = (
        db.table("ai_suggestions")
        .select("*")
        .eq("section_id", section_id)
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []


def update_suggestion(
    suggestion_id: str,
    status: str,
    feedback: str = "",
) -> dict:
    from datetime import datetime, timezone
    updates: dict[str, Any] = {"status": status}
    if feedback:
        updates["feedback"] = feedback
    if status in ("accepted", "rejected", "edited"):
        updates["resolved_at"] = datetime.now(timezone.utc).isoformat()

    res = (
        db.table("ai_suggestions")
        .update(updates)
        .eq("id", suggestion_id)
        .execute()
    )
    return res.data[0]


# ─────────────────────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────────────────────

def save_sources(project_id: str, sources: list[dict]) -> list[str]:
    """
    Bulk-insert paper records. Returns list of inserted IDs.
    Each source dict must have at minimum: title.
    Duplicate titles for the same project are silently skipped.
    """
    if not sources:
        return []

    # Fetch existing titles to avoid duplicates
    existing_res = (
        db.table("sources")
        .select("title")
        .eq("project_id", project_id)
        .execute()
    )
    existing_titles = {r["title"].lower() for r in (existing_res.data or [])}

    rows = []
    for s in sources:
        title = (s.get("title") or "").strip()
        if not title or title.lower() in existing_titles:
            continue
        existing_titles.add(title.lower())
        rows.append({
            "project_id":      project_id,
            "title":           title,
            "authors":         s.get("authors", ""),
            "abstract":        s.get("abstract", ""),
            "url":             s.get("url", ""),
            "doi":             s.get("doi", ""),
            "citation_count":  int(s.get("citation_count", 0) or 0),
            "relevance_score": float(s.get("relevance_score", 0.0) or 0.0),
            "pdf_url":         s.get("pdf_url", ""),
        })

    if not rows:
        return []

    res = db.table("sources").insert(rows).execute()
    return [r["id"] for r in (res.data or [])]


def list_sources(project_id: str) -> list[dict]:
    res = (
        db.table("sources")
        .select("*")
        .eq("project_id", project_id)
        .order("relevance_score", desc=True)
        .execute()
    )
    return res.data or []


def delete_source(source_id: str) -> None:
    db.table("sources").delete().eq("id", source_id).execute()


# ─────────────────────────────────────────────────────────────────────────────
# Literature analysis
# ─────────────────────────────────────────────────────────────────────────────

def save_literature_analysis(
    project_id: str,
    review_content: str,
    gaps_content: str = "",
    warnings: str = "",
) -> str:
    """Insert a new literature_analysis row and return its ID."""
    res = db.table("literature_analysis").insert({
        "project_id":     project_id,
        "review_content": review_content,
        "gaps_content":   gaps_content,
        "warnings":       warnings,
        "status":         "completed",
    }).execute()
    return res.data[0]["id"]


def list_literature_analyses(project_id: str) -> list[dict]:
    res = (
        db.table("literature_analysis")
        .select("*")
        .eq("project_id", project_id)
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []


def get_literature_analysis(analysis_id: str) -> dict | None:
    res = (
        db.table("literature_analysis")
        .select("*")
        .eq("id", analysis_id)
        .single()
        .execute()
    )
    return res.data


def save_literature_citations(
    analysis_id: str,
    source_ids: list[str],
    formatted_citations: list[str],
    citation_style: str = "APA",
) -> None:
    rows = [
        {
            "literature_analysis_id": analysis_id,
            "source_id":              sid,
            "formatted_citation":     cite,
            "citation_style":         citation_style,
            "position":               i + 1,
        }
        for i, (sid, cite) in enumerate(zip(source_ids, formatted_citations))
    ]
    if rows:
        db.table("literature_citations").insert(rows).execute()


def get_literature_citations(analysis_id: str) -> list[dict]:
    res = (
        db.table("literature_citations")
        .select("*, sources(*)")
        .eq("literature_analysis_id", analysis_id)
        .order("position")
        .execute()
    )
    return res.data or []


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def save_visualization(
    project_id: str,
    section_id: str | None,
    viz_type: str,
    title: str,
    raw_data: dict,
    config: dict,
    file_path: str,
    export_format: str = "png",
    export_size: str | None = None,
    color_scheme: str = "default",
    details: str = "",
) -> dict:
    res = db.table("visualizations").insert({
        "project_id":   project_id,
        "section_id":   section_id,
        "type":         viz_type,
        "title":        title,
        "raw_data":     raw_data,
        "config":       config,
        "file_path":    file_path,
        "export_format": export_format,
        "export_size":  export_size,
        "color_scheme": color_scheme,
        "details":      details,
    }).execute()
    return res.data[0]


def list_visualizations(project_id: str) -> list[dict]:
    res = (
        db.table("visualizations")
        .select("*")
        .eq("project_id", project_id)
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []


def get_visualization(viz_id: str) -> dict | None:
    res = (
        db.table("visualizations")
        .select("*")
        .eq("id", viz_id)
        .single()
        .execute()
    )
    return res.data


def delete_visualization(viz_id: str) -> None:
    db.table("visualizations").delete().eq("id", viz_id).execute()


# ─────────────────────────────────────────────────────────────────────────────
# Chat messages
# ─────────────────────────────────────────────────────────────────────────────

def save_message(
    project_id: str,
    role: str,           # "human" | "ai"
    content: str,
    section_id: str | None = None,
) -> dict:
    res = db.table("chat_messages").insert({
        "project_id": project_id,
        "section_id": section_id,
        "role":       role,
        "content":    content,
    }).execute()
    return res.data[0]


def get_recent_messages(project_id: str, limit: int = 10) -> list[dict]:
    """Return messages in chronological order (oldest first), limited to `limit`."""
    res = (
        db.table("chat_messages")
        .select("role, content, created_at")
        .eq("project_id", project_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    messages = res.data or []
    return list(reversed(messages))   # flip back to chronological


def list_chat_messages(project_id: str) -> list[dict]:
    res = (
        db.table("chat_messages")
        .select("*")
        .eq("project_id", project_id)
        .order("created_at")
        .execute()
    )
    return res.data or []


def clear_chat_messages(project_id: str) -> None:
    db.table("chat_messages").delete().eq("project_id", project_id).execute()
