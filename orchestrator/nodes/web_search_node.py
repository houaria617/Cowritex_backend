"""
orchestrator/nodes/web_search_node.py
──────────────────────────────────────
Runs the CoWriteX Search Engine to download PDFs into a project-specific folder.
"""

from __future__ import annotations
import json
import logging
import os
from langsmith import traceable

from ..state import GraphState
from search_engine.run_search import run_search

logger = logging.getLogger(__name__)


@traceable(name="web_search_call")
def web_search_node(state: GraphState) -> dict:
    instruction = state.get("instruction") or state["user_message"]
    project_id = state["project_id"]

    # Create a dedicated workspace for this project's downloaded PDFs
    download_dir = f"./workspace/pdfs/{project_id}"
    os.makedirs(download_dir, exist_ok=True)

    papers = []
    try:
        # Run the full search engine
        json_path = run_search(
            query=instruction,
            max_results=5,          # Keep low to prevent LLM context overflow
            use_scholar=True,
            require_pdf=True,       # Force it to find readable papers
            download_pdfs=True,     # Download them to disk!
            download_folder=download_dir,
            verbose=False
        )

        # Read the output from the generated JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            search_data = json.load(f)
            papers = search_data.get("results", [])

    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        return {
            "research_papers": [],
            "error": f"Search failed: {exc}",
        }

    # Format a readable summary for the researcher's Chat UI
    if papers:
        lines = [f"🔍 **Found and downloaded {len(papers)} papers:**\n"]
        for i, p in enumerate(papers, 1):
            authors = ", ".join(p.get("authors", [])[:2])
            year = p.get("year", "n.d.")
            title = p.get("title", "Untitled")
            lines.append(f"{i}. **{title}** ({authors}, {year})")
        summary = "\n".join(lines)
    else:
        summary = "⚠️ No readable PDFs could be downloaded for this query."

    logger.info("web_search_node downloaded %d papers", len(papers))
    existing = state.get("agent_outputs", {})

    return {
        "research_papers": papers,
        "agent_outputs":   {**existing, "web_search": summary},
        "error":           None,
    }
