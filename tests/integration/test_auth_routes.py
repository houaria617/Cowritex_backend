"""
tests/integration/test_auth_routes.py
───────────────────────────────────────
Integration tests for /auth/me (GET and PATCH).
"""

from __future__ import annotations

from tests.conftest import TEST_USER_ID, make_project


class TestGetMe:

    def test_returns_user_profile(self, client, mock_repo, auth_headers):
        from tests.conftest import TEST_USER_ID
        # mock_repo.get_user already returns {"id": TEST_USER_ID, ...} by default
        # mock_jwt returns TEST_USER_ID as the JWT sub — same constant, always matches
        resp = client.get("/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == TEST_USER_ID
        assert "password_hash" not in data   # must be stripped
        assert "email" in data

    def test_401_without_token(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 403   # HTTPBearer returns 403

    def test_returns_jwt_profile_when_user_not_in_db(self, client, mock_repo, auth_headers):
        mock_repo.get_user.return_value = None
        resp = client.get("/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == TEST_USER_ID
        assert data["_source"] == "jwt_claims"


class TestUpdateMe:

    def test_patch_full_name(self, client, mock_repo, auth_headers):
        updated = {
            "id": TEST_USER_ID, "email": "test@example.com",
            "full_name": "Updated Name", "academic_position": "Professor",
            "organization": "MIT", "field_interests": "NLP",
        }
        mock_repo.update_user.return_value = updated
        resp = client.patch(
            "/auth/me",
            json={"full_name": "Updated Name"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["full_name"] == "Updated Name"
        assert "password_hash" not in resp.json()

    def test_400_on_empty_body(self, client, auth_headers):
        resp = client.patch("/auth/me", json={}, headers=auth_headers)
        assert resp.status_code == 400

    def test_patch_multiple_fields(self, client, mock_repo, auth_headers):
        mock_repo.update_user.return_value = {
            "id": TEST_USER_ID, "email": "test@example.com",
            "full_name": "New Name", "academic_position": "Researcher",
            "organization": "DeepMind", "field_interests": "RL",
        }
        resp = client.patch(
            "/auth/me",
            json={"full_name": "New Name", "organization": "DeepMind"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        call_updates = mock_repo.update_user.call_args[0][1]
        assert "full_name" in call_updates
        assert "organization" in call_updates
