"""
api/dependencies.py
────────────────────
FastAPI dependency functions injected into route handlers.

Supabase JWT verification — supports BOTH signing modes automatically
──────────────────────────────────────────────────────────────────────
Old projects:  HS256  — Legacy JWT Secret (shared secret string)
New projects:  RS256  — JWT Signing Keys  (asymmetric, verified via JWKS)

Detection logic:
  1. If SUPABASE_JWT_SECRET is set → try HS256 first
  2. If HS256 fails for ANY key/algorithm reason → fall through to RS256 JWKS
  3. Only hard-stop on ExpiredSignatureError or DecodeError (truly malformed)

The error "The specified alg value is not allowed" means the token is RS256
but we tried HS256 first. The fallback to JWKS handles this automatically.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import settings
from database import repository as repo

logger = logging.getLogger(__name__)
_bearer = HTTPBearer()


# ─────────────────────────────────────────────────────────────────────────────
# RS256 / JWKS verification
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_jwks_client() -> jwt.PyJWKClient:
    """
    Build and cache a PyJWKClient pointed at Supabase's JWKS endpoint.
    The client caches keys internally and re-fetches when a new kid appears.
    """
    jwks_url = f"{settings.SUPABASE_URL}/auth/v1/.well-known/jwks.json"
    logger.info("Building JWKS client for %s", jwks_url)
    return jwt.PyJWKClient(jwks_url, cache_keys=True)


def _decode_with_jwks(token: str) -> dict:
    """
    Decode a JWT signed with Supabase's RS256 JWT Signing Keys.
    Raises jwt.InvalidTokenError on any verification failure.
    """
    try:
        client = _get_jwks_client()
        signing_key = client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False, "verify_exp": True},
        )
    except jwt.PyJWKClientError as exc:
        # JWKS fetch or key-lookup failed — treat as invalid token
        logger.error("JWKS key lookup failed: %s", exc)
        raise jwt.InvalidTokenError(f"Could not verify token signature: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main verifier — auto-detects HS256 vs RS256
# ─────────────────────────────────────────────────────────────────────────────

# Errors that mean "wrong algorithm / wrong key" — should fall through to RS256
_FALLTHROUGH_ERRORS = (
    jwt.InvalidSignatureError,    # signed with different secret
    jwt.InvalidAlgorithmError,    # token is RS256, we tried HS256
    jwt.DecodeError,              # also covers "alg not allowed" in some PyJWT versions
)


def _decode_token(token: str) -> dict:
    """
    Try HS256 with legacy secret first (if SUPABASE_JWT_SECRET is set).
    Fall through to RS256 JWKS on any algorithm/signature mismatch.
    Only hard-stop on ExpiredSignatureError (token genuinely expired).
    """
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
            # Token is genuinely expired — no point trying RS256
            raise
        except _FALLTHROUGH_ERRORS as exc:
            # Algorithm or signature mismatch — token is probably RS256
            logger.info(
                "HS256 verification skipped (%s) — trying RS256 JWKS", type(
                    exc).__name__
            )
        # Any other InvalidTokenError with secret set → fall through to JWKS too

    # RS256 path — new Supabase JWT Signing Keys
    return _decode_with_jwks(token)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependencies
# ─────────────────────────────────────────────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """
    Verifies the Supabase JWT in Authorization: Bearer <token>.
    Returns the user's UUID (sub claim) as a plain string.
    Works transparently with both Legacy HS256 and new RS256 signing keys.
    """
    token = credentials.credentials
    try:
        payload = _decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired — please sign in again",
        )
    except jwt.InvalidTokenError as exc:
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
    return user_id


def verify_project_access(
    project_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    """
    Confirms that project_id exists and belongs to user_id.
    Returns the project dict so the route doesn't need a second DB call.
    """
    project = repo.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return project
