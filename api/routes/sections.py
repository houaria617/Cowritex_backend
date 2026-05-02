"""
api/routes/sections.py
───────────────────────
Section CRUD and document version history.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response

from api.dependencies import get_current_user, verify_project_access
from api.schemas.requests import SectionCreate, SectionUpdate, VersionRestore, SectionContentUpdate, SectionContentUpdate
from database import repository as repo

router = APIRouter(tags=["sections"])


# ── Sections under a project ──────────────────────────────────────────────────

@router.post("/projects/{project_id}/sections", status_code=201)
def create_section(
    project_id: str,
    body: SectionCreate,
    project: dict = Depends(verify_project_access),
) -> dict:
    return repo.create_section(
        project_id=project_id,
        section_type=body.type,
        title=body.title,
        position=body.position,
    )


@router.get("/projects/{project_id}/sections")
def list_sections(
    project_id: str,
    project: dict = Depends(verify_project_access),
) -> list[dict]:
    return repo.get_project_sections(project_id)


# ── Individual section ────────────────────────────────────────────────────────

def _get_section_with_access(section_id: str, user_id: str = Depends(get_current_user)) -> dict:
    """Shared guard: fetch section and verify the user owns its project."""
    section = repo.get_section(section_id)
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")
    project = repo.get_project(section["project_id"])
    if not project or project["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return section


@router.get("/sections/{section_id}")
def get_section(section: dict = Depends(_get_section_with_access)) -> dict:
    section["current_content"] = repo.get_current_content(section["id"])
    return section


@router.patch("/sections/{section_id}")
def update_section(
    section_id: str,
    body: SectionUpdate,
    section: dict = Depends(_get_section_with_access),
) -> dict:
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    return repo.update_section(section_id, updates)


@router.delete("/sections/{section_id}", status_code=204)
def delete_section(
    section_id: str,
    section: dict = Depends(_get_section_with_access),
) -> Response:
    repo.delete_section(section_id)
    return Response(status_code=204)


# ── Direct content update (human edit outside the graph) ─────────────────────

@router.patch("/sections/{section_id}/content")
def update_section_content(
    section_id: str,
    body: SectionContentUpdate,
    section: dict = Depends(_get_section_with_access),
) -> dict:
    """
    Save new content for a section directly — no AI, no HITL.
    Creates a new document_version (author_type="human", is_current=True).

    Returns the updated section with the new content injected.
    Use this for:
      - Researcher typing directly in the editor
      - Pasting content from an external source
      - Manual corrections after AI generation
    """
    version = repo.save_new_version(
        section_id=section_id,
        content=body.content,
        author_type="human",
        suggestion_id=None,
    )
    # Return full section with updated content
    updated_section = repo.get_section(section_id)
    return updated_section


# ── Direct content save (human writing, no AI) ───────────────────────────────

@router.patch("/sections/{section_id}/content")
def save_section_content(
    section_id: str,
    body: SectionContentUpdate,
    section: dict = Depends(_get_section_with_access),
) -> dict:
    """
    Save content written directly by the researcher — no AI, no HITL.
    Creates a new document_version with author_type="human".
    Returns the section with updated content.
    """
    repo.save_new_version(
        section_id=section_id,
        content=body.content,
        author_type="human",
        suggestion_id=None,
    )
    return repo.get_section(section_id)


# ── Version history ───────────────────────────────────────────────────────────

@router.get("/sections/{section_id}/versions")
def get_versions(
    section_id: str,
    section: dict = Depends(_get_section_with_access),
) -> list[dict]:
    return repo.get_section_versions(section_id)


@router.post("/sections/{section_id}/versions/restore")
def restore_version(
    section_id: str,
    body: VersionRestore,
    section: dict = Depends(_get_section_with_access),
) -> dict:
    return repo.restore_version(
        version_id=body.version_id,
        section_id=section_id,
    )
