"""
api/main.py
────────────
FastAPI application entry point.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from api.routes import auth, projects, sections, orchestrator, data, uploads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

app = FastAPI(
    title="CoWriteX API",
    description="Human-in-the-Loop AI Research Writing Assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(sections.router)
app.include_router(orchestrator.router)
app.include_router(data.router)
app.include_router(uploads.router)


# ── Root endpoint ─────────────────────────────────────────────────────────────
@app.get("/")
def root() -> dict:
    return {"message": "Welcome to CoWriteX API. See /docs for API documentation."}


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok", "env": settings.APP_ENV}
