"""
api/routes/auth.py
───────────────────
Auth endpoints.

Since Supabase owns authentication, this layer is thin:
  - The frontend calls Supabase Auth directly to sign up / sign in / refresh.
  - Our FastAPI only needs GET /auth/me (read profile) and
    PATCH /auth/me (update profile fields we store in the users table).

If you want a pure-API flow (no Supabase JS SDK on frontend), the
POST /auth/login endpoint is included — it calls Supabase's REST auth
so the backend acts as the auth proxy.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_current_user
from api.schemas.requests import UserProfileUpdate
from config.settings import settings
from database import repository as repo

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me")
def get_me(user_id: str = Depends(get_current_user)) -> dict:
    """Return the current user's profile from the users table."""
    user = repo.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.pop("password_hash", None)   # never expose this
    return user


@router.patch("/me")
def update_me(
    body: UserProfileUpdate,
    user_id: str = Depends(get_current_user),
) -> dict:
    """Update editable profile fields."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    updated = repo.update_user(user_id, updates)
    updated.pop("password_hash", None)
    return updated
