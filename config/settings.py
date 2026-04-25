"""
config/settings.py
───────────────────
Single source of truth for all environment variables.
Loaded once at startup via pydantic-settings.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Supabase ──────────────────────────────────────────────────────────────
    SUPABASE_URL:         str
    SUPABASE_SERVICE_KEY: str   # service_role key — bypasses RLS, server-side only
    SUPABASE_JWT_SECRET:  str   # from Supabase dashboard → Settings → API → JWT secret

    # ── LLM providers ────────────────────────────────────────────────────────
    GROQ_API_KEY:   str = ""
    GROQ_MODEL:     str = "llama-3.3-70b-versatile"
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL:   str = "gemini-2.0-flash"

    # ── Search ────────────────────────────────────────────────────────────────
    SEMANTIC_SCHOLAR_API_KEY: str = ""

    # ── App ───────────────────────────────────────────────────────────────────
    APP_ENV:       str = "development"
    CORS_ORIGINS:  str = "http://localhost:3000,http://localhost:5173"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
