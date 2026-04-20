"""
database/repository.py
──────────────────────
High-level helpers called by orchestrator nodes.
All DB access goes through here — keeps nodes thin.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from .client import db


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────
# Projects
# ──────────────────────────────────────────────

def get_project(project_id: str) -> dict:
    res = db.table("projects").select(
        "*").eq("id", project_id).single().execute()
    return res.data


def update_project_thread(project_id: str, thread_id: str) -> None:
    db.table("projects").update({"thread_id": thread_id}).eq(
        "id", project_id).execute()


def update_project_progress(project_id: str, progress: int) -> None:
    db.table("projects").update({"progress": progress}).eq(
        "id", project_id).execute()


# ──────────────────────────────────────────────
# Preferences
# ──────────────────────────────────────────────

def get_preferences(project_id: str) -> dict:
    res = (
        db.table("project_preferences")
        .select("*")
        .eq("project_id", project_id)
        .single()
        .execute()
    )
    return res.data or {}


# ──────────────────────────────────────────────
# Sections
# ──────────────────────────────────────────────

def get_section(section_id: str) -> dict:
    res = db.table("sections").select(
        "*").eq("id", section_id).single().execute()
    return res.data


def get_current_content(section_id: str) -> Optional[str]:
    """Return the text of the is_current document_version for this section."""
    res = (
        db.table("document_versions")
        .select("content")
        .eq("section_id", section_id)
        .eq("is_current", True)
        .single()
        .execute()
    )
    return res.data["content"] if res.data else None


# ──────────────────────────────────────────────
# AI Suggestions
# ──────────────────────────────────────────────

def create_suggestion(
    section_id: str,
    suggested_text: str,
    instruction: str,
    original_text: Optional[str] = None,
) -> str:
    """Insert a pending suggestion and return its id."""
    row = {
        "id":             str(uuid.uuid4()),
        "section_id":     section_id,
        "original_text":  original_text,
        "suggested_text": suggested_text,
        "instruction":    instruction,
        "status":         "pending",
        "created_at":     _now(),
    }
    db.table("ai_suggestions").insert(row).execute()
    return row["id"]


def resolve_suggestion(
    suggestion_id: str,
    status: str,                  # accepted | rejected | edited
    feedback: Optional[str] = None,
) -> None:
    db.table("ai_suggestions").update({
        "status":      status,
        "feedback":    feedback,
        "resolved_at": _now(),
    }).eq("id", suggestion_id).execute()


def get_pending_suggestion(section_id: str) -> Optional[dict]:
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


# ──────────────────────────────────────────────
# Document Versions
# ──────────────────────────────────────────────

def save_new_version(
    section_id: str,
    content: str,
    author_type: str,              # "ai" | "human"
    suggestion_id: Optional[str] = None,
) -> None:
    """
    Mark all old versions as not current, then insert the new one.
    """
    db.table("document_versions").update({"is_current": False}).eq(
        "section_id", section_id
    ).execute()

    res = (
        db.table("document_versions")
        .select("version_number")
        .eq("section_id", section_id)
        .order("version_number", desc=True)
        .limit(1)
        .execute()
    )
    next_version = (res.data[0]["version_number"] + 1) if res.data else 1

    db.table("document_versions").insert({
        "id":             str(uuid.uuid4()),
        "section_id":     section_id,
        "suggestion_id":  suggestion_id,
        "content":        content,
        "version_number": next_version,
        "author_type":    author_type,
        "is_current":     True,
        "created_at":     _now(),
    }).execute()


# ──────────────────────────────────────────────
# Chat Messages
# ──────────────────────────────────────────────

def save_message(
    project_id: str,
    role: str,
    content: str,
    section_id: Optional[str] = None,
) -> None:
    db.table("chat_messages").insert({
        "id":         str(uuid.uuid4()),
        "project_id": project_id,
        "section_id": section_id,
        "role":       role,
        "content":    content,
        "created_at": _now(),
    }).execute()


def get_recent_messages(project_id: str, limit: int = 20) -> list[dict]:
    res = (
        db.table("chat_messages")
        .select("role, content, created_at")
        .eq("project_id", project_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return list(reversed(res.data or []))


# ──────────────────────────────────────────────
# Sources
# ──────────────────────────────────────────────

def save_sources(project_id: str, sources: list[dict]) -> list[str]:
    """Bulk-insert sources; return list of inserted IDs."""
    rows = []
    for s in sources:
        rows.append({
            "id":              str(uuid.uuid4()),
            "project_id":      project_id,
            "title":           s.get("title", ""),
            "authors":         s.get("authors", ""),
            "abstract":        s.get("abstract", ""),
            "url":             s.get("url", ""),
            "doi":             s.get("doi", ""),
            "citation_count":  s.get("citation_count", 0),
            "relevance_score": s.get("relevance_score", 0.0),
            "pdf_url":         s.get("pdf_url", ""),
            "created_at":      _now(),
        })
    if rows:
        db.table("sources").insert(rows).execute()
    return [r["id"] for r in rows]


# ──────────────────────────────────────────────
# Literature Analysis
# ──────────────────────────────────────────────

def save_literature_analysis(
    project_id: str,
    review_content: str,
    gaps_content: str,
    warnings: str,
) -> str:
    row_id = str(uuid.uuid4())
    db.table("literature_analysis").insert({
        "id":             row_id,
        "project_id":     project_id,
        "review_content": review_content,
        "gaps_content":   gaps_content,
        "warnings":       warnings,
        "status":         "pending",
        "created_at":     _now(),
        "updated_at":     _now(),
    }).execute()
    return row_id


def save_literature_citations(
    analysis_id: str,
    source_ids: list[str],
    formatted_citations: list[str],
    citation_style: str,
) -> None:
    rows = []
    for i, (src_id, formatted) in enumerate(zip(source_ids, formatted_citations), start=1):
        rows.append({
            "id":                     str(uuid.uuid4()),
            "literature_analysis_id": analysis_id,
            "source_id":              src_id,
            "formatted_citation":     formatted,
            "citation_style":         citation_style,
            "position":               i,
            "created_at":             _now(),
        })
    if rows:
        db.table("literature_citations").insert(rows).execute()


# ──────────────────────────────────────────────
# Visualizations
# ──────────────────────────────────────────────

def save_visualization(
    project_id: str,
    viz_type: str,
    title: str,
    raw_data: dict,
    config: dict,
    file_path: str,
    export_format: str,
    section_id: Optional[str] = None,
    export_size: Optional[str] = None,
    color_scheme: str = "default",
    details: Optional[str] = None,
) -> str:
    row_id = str(uuid.uuid4())
    db.table("visualizations").insert({
        "id":            row_id,
        "project_id":    project_id,
        "section_id":    section_id,
        "type":          viz_type,
        "title":         title,
        "raw_data":      raw_data,
        "config":        config,
        "file_path":     file_path,
        "export_format": export_format,
        "export_size":   export_size,
        "color_scheme":  color_scheme,
        "details":       details,
        "created_at":    _now(),
        "updated_at":    _now(),
    }).execute()
    return row_id
