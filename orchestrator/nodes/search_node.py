"""
orchestrator/nodes/search_node.py
───────────────────────────────────
Dedicated academic paper search node.

Triggered when:
  • intent == "search"   (user explicitly wants papers)
  • intent == "literature" AND grounded_only == False
      (literature synthesis needs fresh web papers)

When grounded_only == True  →  skips entirely (pass-through).
When grounded_only == False →  queries Semantic Scholar + Google Scholar,
                               stores raw paper list AND a formatted markdown
                               summary into state for downstream nodes.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..state import GraphState
from ._llm import get_llm
from database import repository as repo

# Re-use your existing search core
from search_engine.core import (
    search_semantic_scholar,
    search_google_scholar,
    search_paper_abstract,
)

logger = logging.getLogger(__name__)

# ── How many results to fetch from each source ──────────────────────────────
_MAX_SEMANTIC = 8
_MAX_GOOGLE = 4   # Google Scholar is slower; keep low

# ── LLM prompt: extract a focused search query from user instruction ─────────
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


def _build_query(instruction: str, provider: str) -> str:
    """Use LLM to distill the instruction into a clean academic query."""
    import json
    try:
        llm = get_llm(provider, temperature=0.0)
        response = llm.invoke([
            SystemMessage(content=_QUERY_SYSTEM),
            HumanMessage(content=instruction),
        ])
        raw = response.content.strip().lstrip(
            "```json").lstrip("```").rstrip("```").strip()
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


def _format_summary(papers: list[dict], query: str) -> str:
    """Render a clean markdown summary for HITL / literature_node consumption."""
    if not papers:
        return f"No papers found for query: **{query}**"

    lines = [f"## 🔍 Search Results for: *{query}*\n"]
    lines.append(f"Found **{len(papers)}** paper(s).\n")

    for i, p in enumerate(papers, 1):
        title = p.get("title", "Untitled")
        authors = ", ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        year = p.get("year") or "n.d."
        source = p.get("source", "")
        cites = p.get("citations", 0)
        pdf = p.get("pdf_link")
        abstract = (p.get("abstract") or "").strip()
        abstract_preview = (
            abstract[:300] + "…") if len(abstract) > 300 else abstract

        lines.append(f"### {i}. {title}")
        lines.append(
            f"**Authors:** {authors}  |  **Year:** {year}  |  **Source:** {source}  |  **Citations:** {cites}")
        if pdf:
            lines.append(f"**PDF:** [{pdf}]({pdf})")
        if abstract_preview:
            lines.append(f"\n> {abstract_preview}")
        lines.append("")   # blank line between entries

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────

def search_node(state: GraphState) -> dict:
    """
    Academic paper search node.

    State reads:
        instruction / user_message  — what to search for
        preferences.grounded_only   — if True, skip entirely
        preferences.llm_provider    — which LLM to use for query extraction
        project_id                  — for DB persistence

    State writes:
        search_results              — List[dict]  raw paper records
        search_summary              — str         markdown formatted summary
        agent_outputs["search"]     — same as search_summary (for merge_node)
        last_agent                  — "search"
        error                       — None on success
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
            "search_results":  [],
            "search_summary":  "ℹ️ Web search disabled (grounded-only mode). "
                               "Literature node will use local knowledge base only.",
            "agent_outputs":   {**existing, "search": ""},
            "last_agent":      "search",
            "error":           None,
        }

    # ── Extract a focused academic query ─────────────────────────────────────
    query = _build_query(instruction, provider)
    logger.info("search_node: query=%r", query)

    # ── Fetch from both sources ───────────────────────────────────────────────
    papers: list[dict] = []

    semantic_results = search_semantic_scholar(
        query, max_results=_MAX_SEMANTIC, verbose=False
    )
    papers.extend(semantic_results)
    logger.info("search_node: Semantic Scholar returned %d results",
                len(semantic_results))

    google_results = search_google_scholar(
        query, max_results=_MAX_GOOGLE, verbose=False
    )
    papers.extend(google_results)
    logger.info("search_node: Google Scholar returned %d results",
                len(google_results))

    # ── Post-process ──────────────────────────────────────────────────────────
    papers = _deduplicate(papers)
    papers = _enrich_abstracts(papers, verbose=False)

    # Sort by citation count descending
    papers.sort(key=lambda p: p.get("citations", 0) or 0, reverse=True)

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
                    "url":             p.get("url", ""),
                    "doi":             "",
                    "citation_count":  p.get("citations", 0) or 0,
                    "relevance_score": 0.0,
                    "pdf_url":         p.get("pdf_link", ""),
                }
                for p in papers
            ])
        except Exception as exc:
            logger.warning("search_node: DB persistence failed: %s", exc)

    # ── Merge into agent_outputs ──────────────────────────────────────────────
    existing = state.get("agent_outputs", {})

    return {
        "search_results":  papers,          # raw list for literature_node
        "search_summary":  summary,         # formatted text for HITL / display
        "agent_outputs":   {**existing, "search": summary},
        "last_agent":      "search",
        "error":           None,
        "hitl_action":     None,
        "hitl_feedback":   None,
    }
