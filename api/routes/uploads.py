"""
api/routes/uploads.py
──────────────────────
PDF upload management for the LiteratureAgent.

Critical design note
─────────────────────
LiteratureAgent.load_documents() reads from AgentConfig.papers_folder.
We must store uploads in that exact folder so the agent finds them.

The literature_node passes web_urls extracted from search_results for
online papers. For LOCAL uploaded PDFs, the agent reads them from disk
via load_documents() → process_pdfs_with_metadata().

Upload path: uploads/{project_id}/{filename}.pdf
This path is passed to literature_node via state["preferences"]["papers_folder"]
so AgentConfig can be initialized with the correct folder per project.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, File

from api.dependencies import verify_project_access
from database import repository as repo

router = APIRouter(tags=["uploads"])

# Base upload directory — one subfolder per project
# absolute so agent can always find it
_UPLOAD_BASE = Path("uploads").resolve()


def _project_dir(project_id: str) -> Path:
    """Return and create the upload directory for a project."""
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
    File is saved to uploads/{project_id}/ with its original name.
    Duplicate filenames are overwritten (idempotent).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are accepted")

    # Sanitize filename — strip path components to prevent directory traversal
    safe_name = Path(file.filename).name
    if not safe_name or not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    dest = _project_dir(project_id) / safe_name
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not save file: {exc}")
    finally:
        await file.close()

    return {
        "filename":     safe_name,
        "path":         str(dest),          # absolute path
        "project_id":   project_id,
        "size_bytes":   dest.stat().st_size,
        # pass this to literature agent
        "papers_folder": str(_project_dir(project_id)),
    }


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/upload/pdf")
def list_pdfs(
    project_id: str,
    project: dict = Depends(verify_project_access),
) -> dict:
    """
    List all PDFs uploaded for this project.
    Also returns papers_folder so the frontend knows where to point the agent.
    """
    d = _project_dir(project_id)
    files = [
        {
            "filename":   p.name,
            "size_bytes": p.stat().st_size,
            "path":       str(p),
        }
        for p in sorted(d.glob("*.pdf"))
    ]
    return {
        "project_id":    project_id,
        "papers_folder": str(d),
        "count":         len(files),
        "files":         files,
    }


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/projects/{project_id}/upload/pdf/{filename}", status_code=204)
def delete_pdf(
    project_id: str,
    filename: str,
    project: dict = Depends(verify_project_access),
) -> Response:
    """Remove a specific PDF from the project's upload folder."""
    # Sanitize to prevent path traversal
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files can be deleted")

    target = _project_dir(project_id) / safe_name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    target.unlink()
    return Response(status_code=204)
