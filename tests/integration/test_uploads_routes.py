"""
tests/integration/test_uploads_routes.py
─────────────────────────────────────────
Integration tests for PDF upload endpoints.
Uses tmp_path to avoid touching the real filesystem.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import TEST_PROJECT_ID


@pytest.fixture(autouse=True)
def patch_upload_dir(tmp_path):
    """Redirect all uploads to tmp_path so nothing is written to disk."""
    with patch("api.routes.uploads._UPLOAD_BASE", tmp_path):
        yield tmp_path


def _pdf_file(name: str = "test.pdf") -> tuple:
    """Create a minimal in-memory PDF for upload tests."""
    content = b"%PDF-1.4 fake pdf content"
    return (name, io.BytesIO(content), "application/pdf")


class TestUploadPDF:

    def test_201_on_valid_pdf(self, client, auth_headers):
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/upload/pdf",
            files={"file": _pdf_file()},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "test.pdf"
        assert data["project_id"] == TEST_PROJECT_ID

    def test_400_on_non_pdf(self, client, auth_headers):
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/upload/pdf",
            files={"file": ("document.txt", io.BytesIO(
                b"text"), "text/plain")},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "PDF" in resp.json()["detail"]

    def test_file_is_saved_to_disk(self, client, auth_headers, tmp_path):
        client.post(
            f"/projects/{TEST_PROJECT_ID}/upload/pdf",
            files={"file": _pdf_file("my_paper.pdf")},
            headers=auth_headers,
        )
        saved = tmp_path / TEST_PROJECT_ID / "my_paper.pdf"
        assert saved.exists()


class TestListPDFs:

    def test_empty_list_for_new_project(self, client, auth_headers, tmp_path):
        resp = client.get(
            f"/projects/{TEST_PROJECT_ID}/upload/pdf",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["files"] == []
        assert "papers_folder" in data

    def test_lists_uploaded_files(self, client, auth_headers, tmp_path):
        # Pre-create a PDF in the upload dir
        project_dir = tmp_path / TEST_PROJECT_ID
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "paper1.pdf").write_bytes(b"%PDF content")
        (project_dir / "paper2.pdf").write_bytes(b"%PDF content 2")

        resp = client.get(
            f"/projects/{TEST_PROJECT_ID}/upload/pdf",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        filenames = [f["filename"] for f in data["files"]]
        assert "paper1.pdf" in filenames
        assert "paper2.pdf" in filenames


class TestDeletePDF:

    def test_204_on_valid_delete(self, client, auth_headers, tmp_path):
        project_dir = tmp_path / TEST_PROJECT_ID
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "to_delete.pdf").write_bytes(b"%PDF")

        resp = client.delete(
            f"/projects/{TEST_PROJECT_ID}/upload/pdf/to_delete.pdf",
            headers=auth_headers,
        )
        assert resp.status_code == 204
        assert not (project_dir / "to_delete.pdf").exists()

    def test_404_when_file_missing(self, client, auth_headers):
        resp = client.delete(
            f"/projects/{TEST_PROJECT_ID}/upload/pdf/nonexistent.pdf",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_400_on_non_pdf_filename(self, client, auth_headers, tmp_path):
        project_dir = tmp_path / TEST_PROJECT_ID
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "script.sh").write_bytes(b"#!/bin/bash")

        resp = client.delete(
            f"/projects/{TEST_PROJECT_ID}/upload/pdf/script.sh",
            headers=auth_headers,
        )
        assert resp.status_code == 400
