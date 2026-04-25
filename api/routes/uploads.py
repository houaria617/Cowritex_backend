"""
api/routes/uploads.py
──────────────────────
PDF upload management for the LiteratureAgent.

The LiteratureAgent reads PDFs from config.papers_folder (set in AgentConfig).
We store uploaded files under:
    uploads/{project_id}/

This keeps papers scoped per project so different projects don't share PDFs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, File

from api.dependencies import verify_project_access

router = APIRouter(tags=["uploads"])

# Base upload directory — matches what LiteratureAgent's AgentConfig points to.
# Override in settings if needed.
_UPLOAD_BASE = Path("uploads")


def _project_dir(project_id: str) -> Path:
    d = _UPLOAD_BASE / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/projects/{project_id}/upload/pdf", status_code=201)
async def upload_pdf(
    project_id: str,
    file: UploadFile = File(...),
    project: dict = Depends(verify_project_access),
) -> dict:
    """
    Upload a PDF for the literature agent to process.
    Accepts only .pdf files. Max size enforced by the web server (nginx/uvicorn).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are accepted")

    dest = _project_dir(project_id) / file.filename
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    return {
        "filename":   file.filename,
        "path":       str(dest),
        "project_id": project_id,
    }


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/upload/pdf")
def list_pdfs(
    project_id: str,
    project: dict = Depends(verify_project_access),
) -> list[dict]:
    """List all PDFs uploaded for this project."""
    d = _project_dir(project_id)
    return [
        {
            "filename": p.name,
            "size_bytes": p.stat().st_size,
            "path": str(p),
        }
        for p in sorted(d.glob("*.pdf"))
    ]


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/projects/{project_id}/upload/pdf/{filename}", status_code=204)
def delete_pdf(
    project_id: str,
    filename: str,
    project: dict = Depends(verify_project_access),
) -> Response:
    """Remove a specific PDF from the project's upload folder."""
    target = _project_dir(project_id) / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file() or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file")
    target.unlink()
    return Response(status_code=204)
