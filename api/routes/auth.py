"""
api/routes/auth.py
───────────────────
Auth endpoints.

Important: Supabase Auth stores users in auth.users (managed by Supabase).
Our public.users table is optional extended profile storage.

GET /auth/me  — returns identity from JWT claims + public.users profile if it exists.
               Never fails with 404 just because public.users has no row yet.
PATCH /auth/me — upserts a profile row in public.users.
"""

from __future__ import annotations

import logging

import jwt as pyjwt
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.dependencies import get_current_user
from api.schemas.requests import UserProfileUpdate
from database import repository as repo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

_bearer = HTTPBearer()


def _claims_from_request(request: Request) -> dict:
    """
    Extract unverified claims from the Bearer token already validated by
    get_current_user. Safe to read without re-verifying because the
    dependency already confirmed the signature.
    """
    auth_header = request.headers.get("authorization", "")
    token = auth_header.removeprefix("Bearer ").strip()
    try:
        return pyjwt.decode(
            token,
            options={"verify_signature": False},
        )
    except Exception:
        return {}


@router.get("/me")
def get_me(
    request: Request,
    user_id: str = Depends(get_current_user),
) -> dict:
    """
    Return the current user's profile.

    Priority:
      1. public.users row (if it exists — researcher filled in their profile)
      2. JWT claims fallback (always available — email, sub)

    Never returns 404: if no public.users row exists yet, we return the
    identity from the JWT so the frontend can still bootstrap.
    """
    # Try extended profile from public.users
    db_user = repo.get_user(user_id)

    if db_user:
        db_user.pop("password_hash", None)
        return db_user

    # Fallback: build a minimal profile from JWT claims
    # Supabase stores email in the top-level claim and in user_metadata
    claims = _claims_from_request(request)
    user_metadata = claims.get("user_metadata") or {}

    return {
        "id":                user_id,
        "email":             claims.get("email") or user_metadata.get("email") or "",
        "full_name":         user_metadata.get("full_name") or user_metadata.get("name") or "",
        "academic_position": None,
        "organization":      None,
        "field_interests":   None,
        "created_at":        None,
        "updated_at":        None,
        "_source":           "jwt_claims",   # tells frontend no DB profile exists yet
    }


@router.patch("/me")
def update_me(
    body: UserProfileUpdate,
    user_id: str = Depends(get_current_user),
) -> dict:
    """
    Upsert editable profile fields into public.users.
    Creates the row if it doesn't exist yet.
    """
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    existing = repo.get_user(user_id)

    if existing:
        updated = repo.update_user(user_id, updates)
    else:
        # First time the researcher fills in their profile — create the row
        updated = repo.create_user(
            email=updates.get("email", ""),
            full_name=updates.get("full_name", ""),
            password_hash="",   # managed by Supabase Auth — we store nothing here
            academic_position=updates.get("academic_position", ""),
            organization=updates.get("organization", ""),
            field_interests=updates.get("field_interests", ""),
        )

    updated.pop("password_hash", None)
    return updated
