"""
tests/unit/test_dependencies.py
────────────────────────────────
Unit tests for api/dependencies.py

Key patching rule
─────────────────
api/dependencies.py does:  import jwt
So we MUST patch:           api.dependencies  module attribute "jwt"
using patch.object(deps_module, "jwt", fake_jwt)

NOT:  patch("jwt.decode")           ← patches the global, deps already has its own ref
NOT:  monkeypatch.setattr(config…)  ← settings was already bound at import time
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock, patch

import jwt as real_jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

USER_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

# Must be >= 32 bytes for PyJWT HS256
SECRET = "test-secret-key-padded-to-exactly-32bytes!!"


def _make_credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _make_token(payload: dict, secret: str = SECRET) -> str:
    return real_jwt.encode(payload, secret, algorithm="HS256")


@pytest.fixture(autouse=True)
def patch_deps(monkeypatch):
    """
    Patch BOTH jwt and settings as seen by api.dependencies.
    This must happen before any test calls get_current_user().
    """
    import api.dependencies as deps_module

    # ── settings ──
    mock_settings = MagicMock()
    mock_settings.SUPABASE_JWT_SECRET = SECRET
    monkeypatch.setattr(deps_module, "settings", mock_settings)

    # ── jwt: keep real decode but expose real exception classes ──
    # We do NOT replace jwt.decode here — individual tests that need a
    # real valid token will call _make_token(). Tests that need failures
    # pass bad tokens; the real jwt.decode will raise naturally.
    # We only need to ensure the exception catch-blocks see the right types.
    # Since we're not replacing the jwt module, no patch needed for decode.
    return mock_settings


# ─────────────────────────────────────────────────────────────────────────────
# get_current_user
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCurrentUser:

    def test_valid_token_returns_user_id(self):
        from api.dependencies import get_current_user
        token = _make_token({"sub": USER_ID, "role": "authenticated"})
        result = get_current_user(_make_credentials(token))
        assert result == USER_ID

    def test_expired_token_raises_401(self):
        from api.dependencies import get_current_user
        # exp in the past → ExpiredSignatureError
        token = _make_token({"sub": USER_ID, "exp": int(time.time()) - 100})
        with pytest.raises(HTTPException) as exc:
            get_current_user(_make_credentials(token))
        assert exc.value.status_code == 401
        # PyJWT >= 2 raises ExpiredSignatureError which is a subclass of
        # InvalidTokenError — both land in the InvalidTokenError handler.
        # Accept either message.
        detail = exc.value.detail.lower()
        assert "expired" in detail or "invalid" in detail

    def test_invalid_signature_raises_401(self):
        """
        Token signed with wrong secret → HS256 raises InvalidSignatureError
        which is in _FALLTHROUGH_ERRORS → code falls through to JWKS.
        We patch _decode_with_jwks to raise InvalidTokenError (no HTTP call).
        Final result must be 401.
        """
        import api.dependencies as deps_module
        from api.dependencies import get_current_user

        token = _make_token(
            {"sub": USER_ID}, secret="completely-wrong-secret-padded-!!")

        def fake_jwks_decode(t):
            raise real_jwt.InvalidTokenError("Signature verification failed")

        with patch.object(deps_module, "_decode_with_jwks", fake_jwks_decode):
            with pytest.raises(HTTPException) as exc:
                get_current_user(_make_credentials(token))
        assert exc.value.status_code == 401

    def test_wrong_algorithm_falls_through_to_jwks(self):
        """
        'The specified alg value is not allowed' = InvalidAlgorithmError.
        This is a _FALLTHROUGH_ERRORS member — must reach JWKS, not raise 401 directly.
        We patch JWKS to succeed and confirm the user_id is returned correctly.
        """
        import api.dependencies as deps_module
        from api.dependencies import get_current_user

        # A valid HS256 token — but we'll simulate HS256 decode raising
        # InvalidAlgorithmError (as happens when token header says RS256)
        token = _make_token({"sub": USER_ID})

        def fake_hs256_decode(t, secret, **kwargs):
            raise real_jwt.InvalidAlgorithmError(
                "The specified alg value is not allowed")

        def fake_jwks_decode(t):
            return {"sub": USER_ID, "role": "authenticated"}

        with patch.object(deps_module, "_decode_with_jwks", fake_jwks_decode):
            with patch("api.dependencies.jwt") as mock_jwt:
                mock_jwt.decode.side_effect = fake_hs256_decode
                mock_jwt.ExpiredSignatureError = real_jwt.ExpiredSignatureError
                mock_jwt.InvalidAlgorithmError = real_jwt.InvalidAlgorithmError
                mock_jwt.InvalidSignatureError = real_jwt.InvalidSignatureError
                mock_jwt.DecodeError = real_jwt.DecodeError
                mock_jwt.InvalidTokenError = real_jwt.InvalidTokenError
                mock_jwt.PyJWKClientError = real_jwt.PyJWKClientError

                result = get_current_user(_make_credentials(token))
        assert result == USER_ID

    def test_missing_sub_raises_401(self):
        from api.dependencies import get_current_user
        # Valid token, correct secret, but no "sub" claim
        token = _make_token({"role": "authenticated", "uid": "something-else"})
        with pytest.raises(HTTPException) as exc:
            get_current_user(_make_credentials(token))
        assert exc.value.status_code == 401
        assert "sub" in exc.value.detail.lower()

    def test_malformed_token_raises_401(self):
        from api.dependencies import get_current_user
        with pytest.raises(HTTPException) as exc:
            get_current_user(_make_credentials("not.a.valid.jwt.at.all"))
        assert exc.value.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# verify_project_access
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyProjectAccess:

    def test_owner_gets_project(self):
        from api.dependencies import verify_project_access
        project = {"id": "p1", "user_id": USER_ID, "title": "My project"}
        with patch("api.dependencies.repo") as mock_repo:
            mock_repo.get_project.return_value = project
            result = verify_project_access("p1", USER_ID)
        assert result == project

    def test_nonexistent_project_raises_404(self):
        from api.dependencies import verify_project_access
        with patch("api.dependencies.repo") as mock_repo:
            mock_repo.get_project.return_value = None
            with pytest.raises(HTTPException) as exc:
                verify_project_access("nonexistent", USER_ID)
        assert exc.value.status_code == 404

    def test_other_user_project_raises_403(self):
        from api.dependencies import verify_project_access
        project = {"id": "p1", "user_id": "other-user-id", "title": "Not mine"}
        with patch("api.dependencies.repo") as mock_repo:
            mock_repo.get_project.return_value = project
            with pytest.raises(HTTPException) as exc:
                verify_project_access("p1", USER_ID)
        assert exc.value.status_code == 403
