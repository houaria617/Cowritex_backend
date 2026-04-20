"""
database/client.py
──────────────────
Lazy Supabase client — does NOT connect at import time.
Call get_supabase() or use the `db` proxy only when you actually need it.
"""
from __future__ import annotations

import os
from functools import lru_cache
from supabase import create_client, Client


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


class _LazyClient:
    """
    Proxy that behaves exactly like a supabase Client but only
    calls get_supabase() on first attribute access.
    This means importing database.client never touches env vars.
    """

    def __getattr__(self, name: str):
        return getattr(get_supabase(), name)


db: Client = _LazyClient()   # type: ignore[assignment]
