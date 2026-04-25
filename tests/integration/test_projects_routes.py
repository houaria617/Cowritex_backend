"""
tests/integration/test_projects_routes.py
──────────────────────────────────────────
Integration tests for /projects/* and /projects/{id}/preferences

Uses the TestClient fixture from conftest.py which has:
  - JWT patched (any token passes, returns TEST_USER_ID)
  - repo fully mocked with sensible defaults
"""

from __future__ import annotations

import uuid
from tests.conftest import TEST_PROJECT_ID, TEST_USER_ID, FAKE_BEARER, make_project, make_preferences


class TestCreateProject:

    def test_201_on_valid_body(self, client, mock_repo, auth_headers):
        mock_repo.create_project.return_value = make_project()
        resp = client.post(
            "/projects",
            json={"title": "My Research Paper", "description": "About RAG"},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["title"] == "Test Project"
        mock_repo.create_project.assert_called_once_with(
            user_id=TEST_USER_ID,
            title="My Research Paper",
            description="About RAG",
        )

    def test_422_on_empty_title(self, client, auth_headers):
        resp = client.post(
            "/projects",
            json={"title": ""},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_401_without_token(self, client):
        resp = client.post("/projects", json={"title": "T"})
        assert resp.status_code == 403   # HTTPBearer returns 403 when no creds


class TestListProjects:

    def test_returns_list(self, client, mock_repo, auth_headers):
        mock_repo.list_projects.return_value = [make_project(), make_project()]
        resp = client.get("/projects", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 2

    def test_empty_list_is_valid(self, client, mock_repo, auth_headers):
        mock_repo.list_projects.return_value = []
        resp = client.get("/projects", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []


class TestGetProject:

    def test_200_for_own_project(self, client, mock_repo, auth_headers):
        mock_repo.get_project.return_value = make_project()
        resp = client.get(f"/projects/{TEST_PROJECT_ID}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == TEST_PROJECT_ID

    def test_404_when_not_found(self, client, mock_repo, auth_headers):
        mock_repo.get_project.return_value = None
        resp = client.get(f"/projects/{uuid.uuid4()}", headers=auth_headers)
        assert resp.status_code == 404

    def test_403_for_other_users_project(self, client, mock_repo, auth_headers):
        mock_repo.get_project.return_value = make_project(user_id="other-user")
        resp = client.get(f"/projects/{TEST_PROJECT_ID}", headers=auth_headers)
        assert resp.status_code == 403


class TestUpdateProject:

    def test_patch_title(self, client, mock_repo, auth_headers):
        updated = {**make_project(), "title": "Updated Title"}
        mock_repo.update_project.return_value = updated
        resp = client.patch(
            f"/projects/{TEST_PROJECT_ID}",
            json={"title": "Updated Title"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["title"] == "Updated Title"

    def test_400_on_empty_body(self, client, mock_repo, auth_headers):
        resp = client.patch(
            f"/projects/{TEST_PROJECT_ID}",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_invalid_status_rejected(self, client, mock_repo, auth_headers):
        resp = client.patch(
            f"/projects/{TEST_PROJECT_ID}",
            json={"status": "deleted"},   # not in enum
            headers=auth_headers,
        )
        assert resp.status_code == 422


class TestDeleteProject:

    def test_204_on_delete(self, client, mock_repo, auth_headers):
        resp = client.delete(
            f"/projects/{TEST_PROJECT_ID}",
            headers=auth_headers,
        )
        assert resp.status_code == 204
        assert resp.content == b""
        mock_repo.delete_project.assert_called_once_with(TEST_PROJECT_ID)

    def test_404_when_not_found(self, client, mock_repo, auth_headers):
        mock_repo.get_project.return_value = None
        resp = client.delete(f"/projects/{uuid.uuid4()}", headers=auth_headers)
        assert resp.status_code == 404


class TestPreferences:

    def test_get_preferences(self, client, mock_repo, auth_headers):
        mock_repo.get_preferences.return_value = make_preferences()
        resp = client.get(
            f"/projects/{TEST_PROJECT_ID}/preferences",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "writing_style" in data
        assert "llm_provider" in data

    def test_patch_preferences_valid_provider(self, client, mock_repo, auth_headers):
        mock_repo.upsert_preferences.return_value = {
            **make_preferences(), "llm_provider": "gemini"
        }
        resp = client.patch(
            f"/projects/{TEST_PROJECT_ID}/preferences",
            json={"llm_provider": "gemini"},
            headers=auth_headers,
        )
        assert resp.status_code == 200

    def test_patch_preferences_invalid_provider(self, client, auth_headers):
        resp = client.patch(
            f"/projects/{TEST_PROJECT_ID}/preferences",
            json={"llm_provider": "openai"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_patch_preferences_empty_body(self, client, auth_headers):
        resp = client.patch(
            f"/projects/{TEST_PROJECT_ID}/preferences",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400
