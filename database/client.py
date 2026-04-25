"""
database/client.py
───────────────────
Single Supabase client instance shared across the whole app.
All repository functions import `db` from here.
"""

from __future__ import annotations

from supabase import create_client, Client
from config.settings import settings

_client: Client | None = None


def get_db() -> Client:
    global _client
    if _client is None:
        _client = create_client(settings.SUPABASE_URL,
                                settings.SUPABASE_SERVICE_KEY)
    return _client


# Module-level shortcut used by repository functions and nodes
db: Client = None  # populated on first import via _init


def _init():
    global db
    db = get_db()


_init()
