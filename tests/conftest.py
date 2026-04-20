"""
tests/conftest.py
─────────────────
Shared pytest configuration for all tests.
- Disables LangSmith tracing so no traces are sent during tests.
- Blocks accidental real Supabase connections.
"""
import os
import pytest


def pytest_configure(config):
    """Disable tracing before any test is collected."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_API_KEY"] = "test-fake-key"
    # Provide dummy Supabase vars so the module can be imported safely.
    # The no_real_db fixture below still blocks any actual connection.
    os.environ.setdefault("SUPABASE_URL",
                          "http://localhost:54321")
    os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-fake-key")


@pytest.fixture(autouse=True)
def no_real_db(monkeypatch):
    """
    Prevent any test from hitting a real Supabase instance.
    Patches get_supabase() so it raises immediately if called.
    Individual node tests override this by patching `database.repository`.
    """
    def _block(*a, **kw):
        raise RuntimeError(
            "Test tried to create a real Supabase client! "
            "Patch `database.repository` in your test instead."
        )

    # Clear the lru_cache so our block takes effect
    from database.client import get_supabase
    get_supabase.cache_clear()

    monkeypatch.setattr("database.client.get_supabase", _block)

    yield

    # Clean up cache after test
    try:
        from database.client import get_supabase
        get_supabase.cache_clear()
    except Exception:
        pass
