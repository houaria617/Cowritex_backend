"""
orchestrator/tracing.py
────────────────────────
LangSmith tracing setup.
Call `setup_tracing()` once at app startup (in orchestrator_api.py or main.py).

Required env vars:
    LANGCHAIN_API_KEY       — your LangSmith API key
    LANGCHAIN_PROJECT       — project name shown in LangSmith UI (e.g. "cowritex-dev")
    LANGCHAIN_TRACING_V2    — must be "true"
    LANGCHAIN_ENDPOINT      — usually https://api.smith.langchain.com (default)

Optional:
    LANGCHAIN_HIDE_INPUTS   — "true" to redact inputs in traces (GDPR)
    LANGCHAIN_HIDE_OUTPUTS  — "true" to redact outputs in traces
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def setup_tracing(
    project: str | None = None,
    enabled: bool | None = None,
) -> bool:
    """
    Configure LangSmith environment variables so every LangChain / LangGraph
    call is automatically traced — no code changes needed in nodes.

    Args:
        project : Override LANGCHAIN_PROJECT env var.
        enabled : Override LANGCHAIN_TRACING_V2 env var.
                  Defaults to True if LANGCHAIN_API_KEY is set.

    Returns:
        True if tracing is active, False if skipped.
    """
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")

    if not api_key:
        logger.warning(
            "LangSmith tracing disabled — LANGCHAIN_API_KEY not set. "
            "Set it in .env to enable traces."
        )
        return False

    # Allow caller to override
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project
    if "LANGCHAIN_PROJECT" not in os.environ:
        os.environ["LANGCHAIN_PROJECT"] = "cowritex"

    # Tracing switch
    if enabled is not None:
        os.environ["LANGCHAIN_TRACING_V2"] = "true" if enabled else "false"
    elif "LANGCHAIN_TRACING_V2" not in os.environ:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    # Endpoint (default is fine for cloud LangSmith)
    if "LANGCHAIN_ENDPOINT" not in os.environ:
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    active = os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    if active:
        logger.info(
            "LangSmith tracing ACTIVE → project=%r  endpoint=%s",
            os.environ["LANGCHAIN_PROJECT"],
            os.environ["LANGCHAIN_ENDPOINT"],
        )
    return active


def traced(name: str):
    """
    Decorator — wraps any function in a named LangSmith trace span.
    Useful for wrapping run_writing_agent / run_literature_agent calls
    so they appear as child spans inside the orchestrator trace.

    Usage:
        from orchestrator.tracing import traced

        @traced("writing_agent")
        def run_writing_agent(...): ...
    """
    from langsmith import traceable
    return traceable(name=name)
