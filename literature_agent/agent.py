"""
literature_agent/agent.py
─────────────────────────
MOCK implementation that matches the real run_literature_agent signature exactly.

Real signature (from your teammate's agent.py):
    run_literature_agent(
        project_title:     str,
        citation_style:    str  = "APA",
        use_web_resources: bool = False,
        use_ocr:           bool = False,
        web_urls:          list = None,
        save_to_chromadb:  bool = False,
    ) -> None

Side effects (what the real agent does):
    - Writes  outputs/{project_title}_state_of_the_art.txt
    - Writes  outputs/citations.txt

This mock writes the same files with realistic placeholder content
so the orchestrator's literature_node.py can read them identically.

Swap this file out for the real implementation once branches are merged —
the orchestrator does NOT need any changes.
"""

from __future__ import annotations

import os
from typing import Optional, List


# ── Realistic mock content ──────────────────────────────────────

def _mock_review(project_title: str) -> str:
    return f"""# State of the Art: {project_title}

## Overview

Recent advances in large language models (LLMs) have substantially changed the landscape
of AI-assisted academic writing. Transformer-based architectures now underpin most
state-of-the-art systems for natural language generation, enabling context-aware text
production at scale (Vaswani et al., 2017).

Human-in-the-loop (HITL) frameworks have emerged as a critical design pattern for
research writing tools, ensuring that researchers retain editorial authority while
benefiting from AI-generated suggestions (Smith & Doe, 2023). These systems
typically implement an approval workflow in which AI output is presented to the
researcher before being committed to the document.

## Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) addresses the hallucination problem inherent in
purely parametric LLMs by grounding generation in retrieved source documents
(Lewis et al., 2020). In the context of academic writing assistants, RAG enables
citation-grounded section drafts that can be verified against a corpus of uploaded PDFs.

## Research Gaps

- No benchmark exists for evaluating multi-section coherence across a full research paper.
- Existing tools do not adequately support non-English academic writing.
- Domain-specific fine-tuning for niche fields (medicine, law, engineering) remains limited.
- HITL approval workflows have not been formally evaluated for researcher trust calibration.

*(Mock content — replace with real literature agent output after branch merge)*
"""


def _mock_citations(citation_style: str) -> str:
    if citation_style.upper() == "IEEE":
        return (
            "[1] A. Vaswani et al., \"Attention is All You Need,\" "
            "Advances in Neural Information Processing Systems, 2017.\n"
            "[2] J. Smith and A. Doe, \"Transformer-based Models for Scientific Writing Assistance,\" "
            "arXiv:0000.00001, 2023.\n"
            "[3] P. Lewis et al., \"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,\" "
            "Advances in Neural Information Processing Systems, 2020."
        )
    else:  # APA default
        return (
            "Vaswani, A., et al. (2017). Attention is All You Need. "
            "Advances in Neural Information Processing Systems.\n"
            "Smith, J., & Doe, A. (2023). Transformer-based Models for Scientific Writing Assistance. "
            "arXiv:0000.00001.\n"
            "Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. "
            "Advances in Neural Information Processing Systems."
        )


# ── Public API — matches real signature exactly ─────────────────

def run_literature_agent(
    project_title:     str,
    citation_style:    str = "APA",
    use_web_resources: bool = False,
    use_ocr:           bool = False,
    web_urls:          Optional[List[str]] = None,
    save_to_chromadb:  bool = False,
) -> None:
    """
    MOCK — writes the same output files as the real literature agent.

    Args:
        project_title:     Title of the project (used as query and filename stem).
        citation_style:    "APA" | "IEEE" | "MLA"
        use_web_resources: Whether to scrape web URLs.
        use_ocr:           Whether to OCR scanned PDFs.
        web_urls:          List of URLs to scrape (if use_web_resources=True).
        save_to_chromadb:  If True, persist chunks to ChromaDB (no-op in mock).

    Returns:
        None  (side effect: writes two .txt files to outputs/)
    """
    os.makedirs("outputs", exist_ok=True)

    # ── State-of-the-art file ──
    art_path = os.path.join("outputs", f"{project_title}_state_of_the_art.txt")
    with open(art_path, "w", encoding="utf-8") as f:
        f.write(_mock_review(project_title))

    # ── Citations file ──
    cit_path = os.path.join("outputs", "citations.txt")
    with open(cit_path, "w", encoding="utf-8") as f:
        f.write(_mock_citations(citation_style))
