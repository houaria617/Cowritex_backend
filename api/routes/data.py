"""
api/routes/data.py
───────────────────
All data-layer endpoints:
  - Sources & literature
  - AI suggestions (including inline copilot)
  - Visualizations (including file download)
  - Chat history
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse

from api.dependencies import get_current_user, verify_project_access
from api.schemas.requests import SuggestionUpdate, InlineSuggestionRequest
from database import repository as repo

logger = logging.getLogger(__name__)
router = APIRouter(tags=["data"])


# ── Helper: verify section belongs to user ────────────────────────────────────

def _check_section(section_id: str, user_id: str) -> dict:
    section = repo.get_section(section_id)
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")
    project = repo.get_project(section["project_id"])
    if not project or project["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return section


# ─────────────────────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/sources")
def list_sources(
    project_id: str,
    project: dict = Depends(verify_project_access),
) -> list[dict]:
    return repo.list_sources(project_id)


@router.delete("/sources/{source_id}", status_code=204)
def delete_source(
    source_id: str,
    user_id: str = Depends(get_current_user),
) -> Response:
    repo.delete_source(source_id)
    return Response(status_code=204)


# ─────────────────────────────────────────────────────────────────────────────
# Literature
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/literature")
def list_literature(
    project_id: str,
    project: dict = Depends(verify_project_access),
) -> list[dict]:
    return repo.list_literature_analyses(project_id)


@router.get("/literature/{analysis_id}")
def get_literature(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    analysis = repo.get_literature_analysis(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    analysis["citations"] = repo.get_literature_citations(analysis_id)
    return analysis


# ─────────────────────────────────────────────────────────────────────────────
# AI Suggestions
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/sections/{section_id}/suggestions")
def list_suggestions(
    section_id: str,
    user_id: str = Depends(get_current_user),
) -> list[dict]:
    _check_section(section_id, user_id)
    return repo.list_suggestions(section_id)


@router.patch("/suggestions/{suggestion_id}")
def update_suggestion(
    suggestion_id: str,
    body: SuggestionUpdate,
    user_id: str = Depends(get_current_user),
) -> dict:
    return repo.update_suggestion(
        suggestion_id=suggestion_id,
        status=body.status,
        feedback=body.feedback or "",
    )


@router.post("/sections/{section_id}/suggestions/inline")
async def inline_suggestion(
    section_id: str,
    body: InlineSuggestionRequest,
    user_id: str = Depends(get_current_user),
) -> dict:
    """
    Copilot-style inline suggestion — bypasses the main graph entirely.
    Calls run_suggestion_agent directly and returns diff for accept/reject UI.
    """
    section = _check_section(section_id, user_id)
    prefs = repo.get_preferences(section["project_id"])

    def _call_agent():
        from writing_agent.agent import run_suggestion_agent
        return run_suggestion_agent(
            document=body.document,
            target_text=body.target_text,
            context=prefs,
            target_section=section.get("type", ""),
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _call_agent)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/visualizations")
def list_visualizations(
    project_id: str,
    project: dict = Depends(verify_project_access),
) -> list[dict]:
    return repo.list_visualizations(project_id)


@router.get("/visualizations/{viz_id}")
def get_visualization(
    viz_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    viz = repo.get_visualization(viz_id)
    if not viz:
        raise HTTPException(status_code=404, detail="Visualization not found")
    return viz


@router.get("/visualizations/{viz_id}/download")
def download_visualization(
    viz_id: str,
    user_id: str = Depends(get_current_user),
) -> FileResponse:
    """Serve the generated file (PNG / PDF / LaTeX) from disk."""
    viz = repo.get_visualization(viz_id)
    if not viz:
        raise HTTPException(status_code=404, detail="Visualization not found")

    raw_path = viz.get("file_path", "")
    if not raw_path:
        raise HTTPException(
            status_code=404, detail="No file path stored for this visualization")

    file_path = Path(raw_path)

    # If the stored path is relative, resolve it against the project root.
    # The visualization module saves to visualization/outputs/ relative to CWD
    # when the graph ran — which is the Cowritex project root.
    if not file_path.is_absolute():
        # Try resolving from CWD first (works if server started from project root)
        resolved = Path.cwd() / file_path
        if not resolved.exists():
            # Fallback: try common project root locations
            for base in [Path.cwd(), Path.cwd().parent, Path("/home/djabir-houaria/Cowritex")]:
                candidate = base / file_path
                if candidate.exists():
                    resolved = candidate
                    break
        file_path = resolved

    if not file_path.exists():
        logger.error(
            "Visualization file not found. stored_path=%r resolved=%s cwd=%s",
            raw_path, file_path, Path.cwd()
        )
        raise HTTPException(
            status_code=404,
            detail=f"File not found on disk. Stored path: {raw_path}"
        )

    media_map = {
        "png":   "image/png",
        "pdf":   "application/pdf",
        "latex": "application/x-latex",
    }
    # Extension map: export_format stored in DB → actual file extension on disk
    # "latex" format → file saved as .tex by export_latex_chart()
    ext_map = {
        "png":   "png",
        "pdf":   "pdf",
        "latex": "tex",   # ← critical: format="latex" but file is .tex not .latex
    }
    fmt = viz.get("export_format", "png")
    media = media_map.get(fmt, "application/octet-stream")
    ext = ext_map.get(fmt, fmt)
    # Use the actual file's suffix if available — most reliable
    actual_suffix = file_path.suffix.lstrip(".")
    if actual_suffix:
        ext = actual_suffix
    safe_title = (viz.get("title") or "visualization").replace(" ", "_")
    filename = f"{safe_title}.{ext}"

    return FileResponse(path=str(file_path), media_type=media, filename=filename)


@router.delete("/visualizations/{viz_id}", status_code=204)
def delete_visualization(
    viz_id: str,
    user_id: str = Depends(get_current_user),
) -> Response:
    repo.delete_visualization(viz_id)
    return Response(status_code=204)


# ─────────────────────────────────────────────────────────────────────────────
# Chat history
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/chat")
def get_chat(
    project_id: str,
    project: dict = Depends(verify_project_access),
) -> list[dict]:
    return repo.list_chat_messages(project_id)


@router.delete("/projects/{project_id}/chat", status_code=204)
def clear_chat(
    project_id: str,
    project: dict = Depends(verify_project_access),
) -> Response:
    repo.clear_chat_messages(project_id)
    return Response(status_code=204)
