"""
orchestrator/nodes/search_node.py
───────────────────────────────────
Dedicated academic paper search node.

Triggered when:
  • intent == "search"   (user explicitly wants papers)

Outputs
───────
search_node produces TWO output representations of the same data:

  1. search_results  — List[dict]  raw paper records (for downstream nodes /
                       frontend API consumption), sorted by relevance_score
  2. search_summary  — dict  structured JSON-serialisable summary
                       (replaces the old markdown string) so the frontend
                       can render cards, sort, filter without parsing markdown

Auto-approve
────────────
Search results never go through HITL — they are factual retrieval output,
not AI-generated text. search_node sets hitl_action="approve" directly so
the graph wires straight to persist → output → END.

Scoring
───────
search_node calls core.py functions directly, which do NOT call scoring.py.
So we call calculate_relevance_scores() ourselves after fetching and
deduplication, before sorting and persisting.

Field-name variants (core.py vs wrapper):
  url / paper_url        → resolved by _get_url()
  pdf_link / download_link → resolved by _get_pdf()
"""

from __future__ import annotations

import json
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

def _build_query(instruction: str, provider: str) -> tuple[str, list[str]]:
    """
    Use LLM to distill the instruction into a clean academic query.
    Returns (query_string, keywords_list).
    """
    try:
        llm = get_llm(provider, temperature=0.0)
        response = llm.invoke([
            SystemMessage(content=_QUERY_SYSTEM),
            HumanMessage(content=instruction),
        ])
        raw = (response.content.strip()
               .lstrip("```json").lstrip("```").rstrip("```").strip())
        parsed = json.loads(raw)
        query = parsed.get("query") or instruction
        keywords = parsed.get("keywords") or []
        return query, keywords
    except Exception as exc:
        logger.warning(
            "Query extraction failed (%s) — using raw instruction", exc)
        return instruction, []


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
    Call the engine's own TF-IDF scoring function then sort descending.
    calculate_relevance_scores() writes 'relevance_score' onto each dict
    in-place. Falls back to keyword scoring if sklearn is unavailable.
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


def _build_json_summary(
    papers: list[dict],
    query: str,
    keywords: list[str],
) -> dict:
    """
    Build a structured JSON-serialisable summary of search results.

    Schema
    ──────
    {
      "query":         str,
      "keywords":      list[str],
      "total_results": int,
      "results": [
        {
          "rank":               int,        // 1-based, sorted by relevance
          "title":              str,
          "authors":            list[str],
          "year":               int | null,
          "source":             str,        // "Semantic Scholar" | "Google Scholar"
          "citations":          int,
          "relevance_score":    float,      // 0.0–1.0 from scoring.py
          "abstract":           str,        // full abstract (not truncated)
          "url":                str,        // landing page
          "pdf_url":            str,        // direct PDF or ""
          "google_scholar_url": str         // GS search link or ""
        },
        ...
      ]
    }
    """
    results = []
    for i, p in enumerate(papers, 1):
        results.append({
            "rank":               i,
            "title":              p.get("title", "Untitled"),
            "authors":            p.get("authors", []),
            "year":               p.get("year"),
            "source":             p.get("source", ""),
            "citations":          p.get("citations", 0) or 0,
            "relevance_score":    _get_score(p),
            "abstract":           (p.get("abstract") or "").strip(),
            "url":                _get_url(p),
            "pdf_url":            _get_pdf(p),
            "google_scholar_url": p.get("google_scholar_url") or "",
        })

    return {
        "query":         query,
        "keywords":      keywords,
        "total_results": len(results),
        "results":       results,
    }


# ─────────────────────────────────────────────────────────────────────────────

def search_node(state: GraphState) -> dict:
    """
    Academic paper search node.

    State reads:
        instruction / user_message  — what to search for
        preferences.grounded_only   — if True, skip entirely
        preferences.llm_provider    — LLM for query extraction
        project_id                  — for DB persistence

    State writes:
        search_results          — List[dict] raw paper records, each carrying
                                  relevance_score, sorted highest-first
        search_summary          — dict  structured JSON summary (frontend-ready)
        agent_outputs["search"] — JSON string of search_summary (for merge_node
                                  display / output_node passthrough)
        last_agent              — "search"
        hitl_action             — "approve"  (search always auto-approves)
        error                   — None on success
    """
    prefs = state.get("preferences", {})
    project_id = state["project_id"]
    instruction = state.get("instruction") or state["user_message"]
    provider = prefs.get("llm_provider", "groq")

    # ── Grounded-only guard ───────────────────────────────────────────────────
    if prefs.get("grounded_only", False):
        logger.info("search_node: grounded_only=True — skipping web search")
        empty_summary = {
            "query":         instruction,
            "keywords":      [],
            "total_results": 0,
            "results":       [],
            "note":          "Web search disabled (grounded-only mode).",
        }
        existing = state.get("agent_outputs", {})
        return {
            "search_results": [],
            "search_summary": empty_summary,
            "agent_outputs":  {**existing, "search": json.dumps(empty_summary, ensure_ascii=False)},
            "last_agent":     "search",
            "hitl_action":    "approve",   # auto-approve even on skip
            "error":          None,
        }

    # ── Build focused academic query ──────────────────────────────────────────
    query, keywords = _build_query(instruction, provider)
    logger.info("search_node: query=%r  keywords=%s", query, keywords)

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
    papers = _score_and_sort(papers, query)

    # ── Build structured JSON summary ─────────────────────────────────────────
    summary = _build_json_summary(papers, query, keywords)

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

    # ── Return — hitl_action="approve" skips HITL entirely ───────────────────
    existing = state.get("agent_outputs", {})

    return {
        "search_results":  papers,
        "search_summary":  summary,
        "agent_outputs":   {**existing, "search": json.dumps(summary, ensure_ascii=False, indent=2)},
        "last_agent":      "search",
        "hitl_action":     "approve",    # search auto-approves — no human review needed
        "hitl_feedback":   None,
        "error":           None,
    }
