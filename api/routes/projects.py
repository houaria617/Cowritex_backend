"""
api/routes/projects.py
───────────────────────
Project CRUD and preferences management.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response

from api.dependencies import get_current_user, verify_project_access
from api.schemas.requests import ProjectCreate, ProjectUpdate, PreferencesUpdate
from database import repository as repo

router = APIRouter(prefix="/projects", tags=["projects"])


# ── Projects ──────────────────────────────────────────────────────────────────

@router.post("", status_code=201)
def create_project(
    body: ProjectCreate,
    user_id: str = Depends(get_current_user),
) -> dict:
    return repo.create_project(
        user_id=user_id,
        title=body.title,
        description=body.description,
    )


@router.get("")
def list_projects(user_id: str = Depends(get_current_user)) -> list[dict]:
    return repo.list_projects(user_id)


@router.get("/{project_id}")
def get_project(
    project: dict = Depends(verify_project_access),
) -> dict:
    return project


@router.patch("/{project_id}")
def update_project(
    project_id: str,
    body: ProjectUpdate,
    project: dict = Depends(verify_project_access),
) -> dict:
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    return repo.update_project(project_id, updates)


@router.delete("/{project_id}", status_code=204)
def delete_project(
    project_id: str,
    project: dict = Depends(verify_project_access),
) -> Response:
    repo.delete_project(project_id)
    return Response(status_code=204)


# ── Preferences ───────────────────────────────────────────────────────────────

@router.get("/{project_id}/preferences")
def get_preferences(
    project_id: str,
    project: dict = Depends(verify_project_access),
) -> dict:
    return repo.get_preferences(project_id)


@router.patch("/{project_id}/preferences")
def update_preferences(
    project_id: str,
    body: PreferencesUpdate,
    project: dict = Depends(verify_project_access),
) -> dict:
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    return repo.upsert_preferences(project_id, updates)
