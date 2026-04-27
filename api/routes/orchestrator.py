"""
api/routes/orchestrator.py
───────────────────────────
The heart of the backend — wraps the LangGraph graph.

POST /projects/{project_id}/run
    Triggers a new graph run. The graph pauses at interrupt_before=["hitl"].
    Returns thread_id + agent_output for the researcher to review.

POST /projects/{project_id}/run/{thread_id}/resume
    Injects the researcher's HITL decision and lets the graph finish.
    Returns the final agent_output and status.

GET /projects/{project_id}/run/{thread_id}/status
    Returns the current state of the graph thread (for polling).

How the graph pause/resume works with MemorySaver
──────────────────────────────────────────────────
1. graph.invoke(initial_state, config)
   → runs until interrupt_before=["hitl"] fires
   → state is saved in MemorySaver under thread_id
   → returns the state at the interruption point

2. graph.invoke(resume_state, config)  ← same thread_id config
   → MemorySaver replays from the checkpoint
   → graph continues from hitl node onward
   → returns final state
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from functools import partial

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import verify_project_access
from api.schemas.requests import RunRequest, RunResponse, ResumeRequest, ResumeResponse, StatusResponse
from database import repository as repo

logger = logging.getLogger(__name__)
router = APIRouter(tags=["orchestrator"])

# ── Graph singleton (built once at startup) ───────────────────────────────────
# Imported lazily inside the route functions so the graph is not built
# at import time (avoids circular imports with database.client).

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from langgraph.checkpoint.memory import MemorySaver
        from orchestrator.graph import build_graph
        _graph = build_graph(checkpointer=MemorySaver())
    return _graph


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config(thread_id: str) -> dict:
    """LangGraph config dict that scopes the checkpoint to a single thread."""
    return {"configurable": {"thread_id": thread_id}}


def _run_graph_sync(initial_state: dict, config: dict) -> dict:
    """
    Invoke the graph synchronously.
    Called via run_in_executor so it doesn't block the event loop.
    """
    graph = _get_graph()
    result = graph.invoke(initial_state, config)
    return result


def _resume_graph_sync(resume_state: dict, config: dict) -> dict:
    """
    Resume after HITL using the correct LangGraph interrupt/resume pattern.

    The graph paused at interrupt_before=["hitl"].
    Correct sequence:
      1. graph.update_state(config, hitl_fields) — merge HITL decision into checkpoint
      2. graph.invoke(None, config)              — resume from where it paused
         (None input = continue from checkpoint, don't restart)

    Passing a partial state dict to invoke() directly causes KeyError on
    required fields like 'project_id' because LangGraph treats it as a new run.
    """
    graph = _get_graph()
    # Step 1: inject the HITL decision into the saved checkpoint
    graph.update_state(config, resume_state)
    # Step 2: resume execution — None input means "continue from checkpoint"
    result = graph.invoke(None, config)
    return result


def _get_state_sync(config: dict) -> dict | None:
    """Read the current saved state without advancing the graph."""
    graph = _get_graph()
    snapshot = graph.get_state(config)
    return snapshot.values if snapshot else None


# ─────────────────────────────────────────────────────────────────────────────
# POST /projects/{project_id}/run
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/projects/{project_id}/run", response_model=RunResponse)
async def run_graph(
    project_id: str,
    body: RunRequest,
    project: dict = Depends(verify_project_access),
) -> RunResponse:
    """
    Start a new graph run for the given project.

    A fresh thread_id is generated per run so each conversation turn
    has its own checkpoint. The thread_id is persisted to the project
    so /resume and /status can look it up.
    """
    thread_id = str(uuid.uuid4())
    config = _make_config(thread_id)

    # Load preferences — intent_classifier will also do this, but pre-loading
    # here means the graph has prefs from the first node.
    prefs = repo.get_preferences(project_id)

    initial_state = {
        "project_id":      project_id,
        "section_id":      body.section_id,
        "user_id":         project["user_id"],
        "user_message":    body.user_message,
        "preferences":     prefs,
        "messages":        [],
        "agent_outputs":   {},
        "search_results":  [],
        "intents":         [],
        "intent":          "unknown",
        "last_agent":      None,
        "agent_output":    None,
        "search_summary":  None,
        "hitl_action":     None,
        "hitl_feedback":   None,
        "human_edited_text": None,
        "document_context": None,
        "instruction":     None,
        "error":           None,
    }

    loop = asyncio.get_event_loop()
    try:
        final_state = await loop.run_in_executor(
            None, partial(_run_graph_sync, initial_state, config)
        )
    except Exception as exc:
        logger.error("Graph run failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Graph error: {exc}")

    # Persist thread_id so /resume and /status can use it
    repo.save_thread_id(project_id, thread_id)

    return RunResponse(
        thread_id=thread_id,
        agent_output=final_state.get("agent_output"),
        intents=final_state.get("intents", []),
        status="pending_review",
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /projects/{project_id}/run/{thread_id}/resume
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/projects/{project_id}/run/{thread_id}/resume",
    response_model=ResumeResponse,
)
async def resume_graph(
    project_id: str,
    thread_id: str,
    body: ResumeRequest,
    project: dict = Depends(verify_project_access),
) -> ResumeResponse:
    """
    Resume the graph after the researcher makes a HITL decision.

    The graph was paused at interrupt_before=["hitl"].
    We inject the decision into the saved state and let it continue.

    For "approve" / "edit": graph runs to END, status = "completed".
    For "reject" / "regenerate": graph loops back to the agent, then
    pauses again at hitl — status = "pending_review".
    """
    config = _make_config(thread_id)

    if body.hitl_action == "edit" and not body.human_edited_text:
        raise HTTPException(
            status_code=400,
            detail="human_edited_text is required for edit action",
        )

    # State update injected into the checkpoint before resuming.
    # LangGraph merges this into the saved state.
    resume_state = {
        "hitl_action":       body.hitl_action,
        "hitl_feedback":     body.hitl_feedback,
        "human_edited_text": body.human_edited_text,
    }

    loop = asyncio.get_event_loop()
    try:
        final_state = await loop.run_in_executor(
            None, partial(_resume_graph_sync, resume_state, config)
        )
    except Exception as exc:
        logger.error("Graph resume failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Resume error: {exc}")

    # "reject" / "regenerate" loop back — the graph will pause at hitl again
    action = body.hitl_action
    status = "completed" if action in ("approve", "edit") else "pending_review"

    return ResumeResponse(
        thread_id=thread_id,
        agent_output=final_state.get("agent_output"),
        status=status,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /projects/{project_id}/run/{thread_id}/status
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/projects/{project_id}/run/{thread_id}/status",
    response_model=StatusResponse,
)
async def get_run_status(
    project_id: str,
    thread_id: str,
    project: dict = Depends(verify_project_access),
) -> StatusResponse:
    """
    Read the current state of a graph thread without advancing it.
    Useful for the frontend to poll while a long-running search is in progress.
    """
    config = _make_config(thread_id)
    loop = asyncio.get_event_loop()

    state = await loop.run_in_executor(
        None, partial(_get_state_sync, config)
    )

    if not state:
        raise HTTPException(status_code=404, detail="Thread not found")

    return StatusResponse(
        thread_id=thread_id,
        status="pending_review" if state.get("agent_output") else "running",
        intents=state.get("intents", []),
        last_agent=state.get("last_agent"),
        error=state.get("error"),
    )
