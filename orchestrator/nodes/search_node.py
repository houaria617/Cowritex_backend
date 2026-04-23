"""
orchestrator/nodes/search_node.py
───────────────────────────────────
Dedicated academic paper search node.

Triggered when:
  • intent == "search"   (user explicitly wants papers)
  • intent == "literature" AND grounded_only == False

Scoring
───────
search_node calls core.py functions directly, which do NOT call scoring.py.
So we call calculate_relevance_scores() ourselves after fetching and
deduplication, before sorting and persisting.

The scoring function writes 'relevance_score' onto each paper dict.
The engine wrapper (used elsewhere) renames that to 'score' in its JSON
output — here we're pre-wrapper so we always read 'relevance_score'.

Field-name variants (core.py vs wrapper):
  url / paper_url        → resolved by _get_url()
  pdf_link / download_link → resolved by _get_pdf()
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..state import GraphState
from ._llm import get_llm
from database import repository as repo

from search_engine.core import (
    search_semantic_scholar,
    search_google_scholar,
    search_paper_abstract,
)
from search_engine.scoring import calculate_relevance_scores

logger = logging.getLogger(__name__)

_MAX_SEMANTIC = 8
_MAX_GOOGLE = 4

_QUERY_SYSTEM = """You are a query builder for an academic search engine.

Given the user's instruction, output ONLY a JSON object:
{
  "query":    "<concise academic search query, 3-8 words>",
  "keywords": ["<kw1>", "<kw2>", "<kw3>"]
}

Rules:
- Strip conversational filler ("find me", "can you search for", etc.)
- Keep domain terminology exact (e.g. "RAG", "transformer", "LLM fine-tuning")
- keywords: 2-5 individual terms useful for filtering
Respond with raw JSON only — no markdown fences."""


# ── Field-name helpers ────────────────────────────────────────────────────────

def _get_url(p: dict) -> str:
    """Handles both 'url' (core.py) and 'paper_url' (wrapper) field names."""
    return (p.get("url") or p.get("paper_url") or "").strip()


def _get_pdf(p: dict) -> str:
    """Handles both 'pdf_link' (core.py) and 'download_link' (wrapper) field names."""
    return (p.get("pdf_link") or p.get("download_link") or "").strip()


def _get_score(p: dict) -> float:
    """
    Reads the score written by calculate_relevance_scores() — field name is
    'relevance_score' when coming from core.py + scoring.py directly.
    Falls back to 'score' in case the wrapper has already been applied.
    """
    raw = p.get("relevance_score") if p.get("relevance_score") is not None \
        else p.get("score")
    try:
        return round(float(raw), 4) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_query(instruction: str, provider: str) -> str:
    import json
    try:
        llm = get_llm(provider, temperature=0.0)
        response = llm.invoke([
            SystemMessage(content=_QUERY_SYSTEM),
            HumanMessage(content=instruction),
        ])
        raw = (response.content.strip()
               .lstrip("```json").lstrip("```").rstrip("```").strip())
        parsed = json.loads(raw)
        return parsed.get("query") or instruction
    except Exception as exc:
        logger.warning(
            "Query extraction failed (%s) — using raw instruction", exc)
        return instruction


def _deduplicate(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers by normalised title."""
    seen: set[str] = set()
    unique = []
    for p in papers:
        key = (p.get("title") or "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _enrich_abstracts(papers: list[dict], verbose: bool = False) -> list[dict]:
    """Fill in missing abstracts via ArXiv lookup (best-effort)."""
    for p in papers:
        if not p.get("abstract"):
            fetched = search_paper_abstract(p, verbose=verbose)
            if fetched:
                p["abstract"] = fetched
    return papers


def _score_and_sort(papers: list[dict], query: str) -> list[dict]:
    """
    Call the engine's own scoring function then sort descending.

    calculate_relevance_scores() writes 'relevance_score' onto each dict
    in-place (TF-IDF when sklearn is available, keyword fallback otherwise).
    We then sort by that field — no custom heuristic needed.
    """
    if not papers:
        return papers

    papers = calculate_relevance_scores(papers, query)
    papers.sort(key=lambda p: _get_score(p), reverse=True)

    logger.info(
        "search_node: scored %d papers — top=%.4f  bottom=%.4f",
        len(papers),
        _get_score(papers[0]),
        _get_score(papers[-1]),
    )
    return papers


def _format_summary(papers: list[dict], query: str) -> str:
    """
    Markdown summary for HITL and literature_node.

    Per paper:
      title, authors, year, source, citations, relevance score,
      PDF link, Google Scholar URL, abstract preview (300 chars).
    """
    if not papers:
        return f"No papers found for query: **{query}**"

    lines = [f"## 🔍 Search Results for: *{query}*\n"]
    lines.append(f"Found **{len(papers)}** paper(s), sorted by relevance.\n")

    for i, p in enumerate(papers, 1):
        title = p.get("title", "Untitled")
        authors = ", ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        year = p.get("year") or "n.d."
        source = p.get("source", "")
        cites = p.get("citations", 0)
        score = _get_score(p)
        pdf = _get_pdf(p)
        gs_url = p.get("google_scholar_url") or ""
        abstract = (p.get("abstract") or "").strip()
        abstract_preview = (
            abstract[:300] + "…") if len(abstract) > 300 else abstract

        lines.append(f"### {i}. {title}")
        lines.append(
            f"**Authors:** {authors}  |  **Year:** {year}  |  "
            f"**Source:** {source}  |  **Citations:** {cites}  |  "
            f"**Relevance:** {score:.4f}"
        )

        link_parts = []
        if pdf:
            link_parts.append(f"[PDF]({pdf})")
        if gs_url:
            link_parts.append(f"[Google Scholar]({gs_url})")
        if link_parts:
            lines.append("**Links:** " + "  ·  ".join(link_parts))

        if abstract_preview:
            lines.append(f"\n> {abstract_preview}")

        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────

def search_node(state: GraphState) -> dict:
    """
    State reads:
        instruction / user_message  — what to search for
        preferences.grounded_only   — if True, skip entirely
        preferences.llm_provider    — LLM for query extraction
        project_id                  — for DB persistence

    State writes:
        search_results   — List[dict] scored and sorted by relevance_score,
                           ready for literature_node and frontend display
        search_summary   — str markdown formatted summary
        agent_outputs["search"] — same as search_summary
        last_agent       — "search"
        error            — None on success
    """
    prefs = state.get("preferences", {})
    project_id = state["project_id"]
    instruction = state.get("instruction") or state["user_message"]
    provider = prefs.get("llm_provider", "groq")

    # ── Grounded-only guard ───────────────────────────────────────────────────
    if prefs.get("grounded_only", False):
        logger.info("search_node: grounded_only=True — skipping web search")
        existing = state.get("agent_outputs", {})
        return {
            "search_results": [],
            "search_summary": (
                "ℹ️ Web search disabled (grounded-only mode). "
                "Literature node will use local knowledge base only."
            ),
            "agent_outputs":  {**existing, "search": ""},
            "last_agent":     "search",
            "error":          None,
        }

    # ── Build focused query ───────────────────────────────────────────────────
    query = _build_query(instruction, provider)
    logger.info("search_node: query=%r", query)

    # ── Fetch from both sources ───────────────────────────────────────────────
    papers: list[dict] = []

    semantic_results = search_semantic_scholar(
        query, max_results=_MAX_SEMANTIC, verbose=False
    )
    papers.extend(semantic_results)
    logger.info("search_node: Semantic Scholar → %d results",
                len(semantic_results))

    google_results = search_google_scholar(
        query, max_results=_MAX_GOOGLE, verbose=False
    )
    papers.extend(google_results)
    logger.info("search_node: Google Scholar → %d results",
                len(google_results))

    # ── Post-process ──────────────────────────────────────────────────────────
    papers = _deduplicate(papers)
    papers = _enrich_abstracts(papers, verbose=False)

    # Score with engine's own TF-IDF scorer, then sort descending
    papers = _score_and_sort(papers, query)

    # ── Format markdown summary ───────────────────────────────────────────────
    summary = _format_summary(papers, query)

    # ── Persist to DB ─────────────────────────────────────────────────────────
    if papers:
        try:
            repo.save_sources(project_id, [
                {
                    "title":           p.get("title", ""),
                    "authors":         ", ".join(p.get("authors", [])),
                    "abstract":        p.get("abstract", ""),
                    "url":             _get_url(p),
                    "doi":             "",
                    "citation_count":  p.get("citations", 0) or 0,
                    "relevance_score": _get_score(p),
                    "pdf_url":         _get_pdf(p),
                }
                for p in papers
            ])
        except Exception as exc:
            logger.warning("search_node: DB persistence failed: %s", exc)

    existing = state.get("agent_outputs", {})

    return {
        "search_results":  papers,
        "search_summary":  summary,
        "agent_outputs":   {**existing, "search": summary},
        "last_agent":      "search",
        "error":           None,
        "hitl_action":     None,
        "hitl_feedback":   None,
    }
