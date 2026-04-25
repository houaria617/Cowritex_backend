"""
tests/integration/test_sections_routes.py
──────────────────────────────────────────
Integration tests for /projects/{id}/sections and /sections/*
"""

from __future__ import annotations

import uuid
from tests.conftest import TEST_PROJECT_ID, TEST_SECTION_ID, make_section, make_project


class TestCreateSection:

    def test_201_on_valid_body(self, client, mock_repo, auth_headers):
        mock_repo.create_section.return_value = make_section()
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/sections",
            json={"type": "introduction",
                  "title": "Introduction", "position": 1},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["type"] == "introduction"

    def test_invalid_section_type_rejected(self, client, auth_headers):
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/sections",
            json={"type": "bibliography"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_position_must_be_gte_1(self, client, auth_headers):
        resp = client.post(
            f"/projects/{TEST_PROJECT_ID}/sections",
            json={"type": "abstract", "position": 0},
            headers=auth_headers,
        )
        assert resp.status_code == 422


class TestListSections:

    def test_returns_ordered_sections(self, client, mock_repo, auth_headers):
        sections = [
            make_section(), {**make_section(), "id": str(uuid.uuid4()), "position": 2}]
        mock_repo.get_project_sections.return_value = sections
        resp = client.get(
            f"/projects/{TEST_PROJECT_ID}/sections",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 2


class TestGetSection:

    def test_200_with_current_content(self, client, mock_repo, auth_headers):
        mock_repo.get_section.return_value = make_section()
        mock_repo.get_project.return_value = make_project()
        mock_repo.get_current_content.return_value = "Section text here"
        resp = client.get(f"/sections/{TEST_SECTION_ID}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["current_content"] == "Section text here"

    def test_404_when_section_missing(self, client, mock_repo, auth_headers):
        mock_repo.get_section.return_value = None
        resp = client.get(f"/sections/{uuid.uuid4()}", headers=auth_headers)
        assert resp.status_code == 404

    def test_403_when_section_belongs_to_other_user(self, client, mock_repo, auth_headers):
        mock_repo.get_section.return_value = make_section()
        mock_repo.get_project.return_value = {
            **make_project(), "user_id": "other-user"}
        resp = client.get(f"/sections/{TEST_SECTION_ID}", headers=auth_headers)
        assert resp.status_code == 403


class TestUpdateSection:

    def test_patch_title(self, client, mock_repo, auth_headers):
        mock_repo.update_section.return_value = {
            **make_section(), "title": "New Title"}
        resp = client.patch(
            f"/sections/{TEST_SECTION_ID}",
            json={"title": "New Title"},
            headers=auth_headers,
        )
        assert resp.status_code == 200

    def test_400_on_empty_body(self, client, auth_headers):
        resp = client.patch(
            f"/sections/{TEST_SECTION_ID}",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400


class TestDeleteSection:

    def test_204_on_delete(self, client, mock_repo, auth_headers):
        resp = client.delete(
            f"/sections/{TEST_SECTION_ID}", headers=auth_headers)
        assert resp.status_code == 204
        assert resp.content == b""


class TestVersionHistory:

    def test_get_versions_returns_list(self, client, mock_repo, auth_headers):
        mock_repo.get_section_versions.return_value = [
            {"id": str(uuid.uuid4()), "version_number": 1,
             "author_type": "ai", "is_current": True},
            {"id": str(uuid.uuid4()), "version_number": 2,
             "author_type": "human", "is_current": False},
        ]
        resp = client.get(
            f"/sections/{TEST_SECTION_ID}/versions", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_restore_version(self, client, mock_repo, auth_headers):
        version_id = str(uuid.uuid4())
        mock_repo.restore_version.return_value = {
            "id": str(uuid.uuid4()), "version_number": 3, "is_current": True
        }
        resp = client.post(
            f"/sections/{TEST_SECTION_ID}/versions/restore",
            json={"version_id": version_id},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        mock_repo.restore_version.assert_called_once_with(
            version_id=version_id,
            section_id=TEST_SECTION_ID,
        )
