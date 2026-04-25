"""
api/dependencies.py
────────────────────
Supabase JWT verification supporting both HS256 (legacy) and RS256 (new).
"""

from __future__ import annotations

import logging

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import settings
from database import repository as repo

logger = logging.getLogger(__name__)
_bearer = HTTPBearer()

_FALLTHROUGH = (
    jwt.InvalidSignatureError,
    jwt.InvalidAlgorithmError,
    jwt.DecodeError,
)


def _decode_with_rs256(token: str) -> dict:
    """Verify RS256/ES256 token using Supabase JWKS endpoint."""
    jwks_url = f"{settings.SUPABASE_URL}/auth/v1/.well-known/jwks.json"
    logger.info("RS256/ES256 path: fetching JWKS from %s", jwks_url)
    try:
        client = jwt.PyJWKClient(jwks_url)
        signing_key = client.get_signing_key_from_jwt(token)
        logger.info("JWKS signing key found kid=%s",
                    getattr(signing_key, 'key_id', 'unknown'))
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            options={"verify_aud": False, "verify_exp": True},
        )
        logger.info("JWKS decode SUCCESS alg=ES256/RS256 sub=%s",
                    payload.get("sub"))
        return payload
    except jwt.PyJWKClientError as exc:
        logger.error("RS256 JWKS client error: %s: %s",
                     type(exc).__name__, exc)
        raise jwt.InvalidTokenError(f"JWKS error: {exc}")
    except jwt.ExpiredSignatureError:
        logger.warning("JWKS path: token expired")
        raise
    except jwt.InvalidTokenError as exc:
        logger.error("RS256 decode failed: %s: %s", type(exc).__name__, exc)
        raise


def _decode_token(token: str) -> dict:
    secret = (settings.SUPABASE_JWT_SECRET or "").strip()

    if secret:
        try:
            return jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                options={"verify_aud": False, "verify_exp": True},
            )
        except jwt.ExpiredSignatureError:
            raise
        except _FALLTHROUGH as exc:
            logger.info(
                "HS256 skipped (%s) — falling back to RS256 JWKS",
                type(exc).__name__,
            )

    return _decode_with_rs256(token)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    token = credentials.credentials

    # Log token header so we can see kid and alg without exposing the secret
    try:
        unverified_header = jwt.get_unverified_header(token)
        logger.info("Token header: %s", unverified_header)
    except Exception:
        pass

    try:
        payload = _decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired — please sign in again",
        )
    except jwt.InvalidTokenError as exc:
        logger.error("Token rejected: %s: %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        )

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing sub claim",
        )

    # Auto-provision public.users row from JWT claims on first use.
    # This bridges auth.users (Supabase) → public.users (our FK target).
    try:
        email = payload.get("email", "") or ""
        user_meta = payload.get("user_metadata") or {}
        full_name = user_meta.get("full_name") or user_meta.get("name") or ""
        repo.ensure_user_exists(user_id, email=email, full_name=full_name)
    except Exception as exc:
        logger.warning("ensure_user_exists failed (non-fatal): %s", exc)

    return user_id


def verify_project_access(
    project_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    project = repo.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return project
