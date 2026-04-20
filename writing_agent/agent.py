"""
writing_agent/agent.py
──────────────────────
MOCK implementation that matches the real run_writing_agent signature exactly.

Real signature (from your teammate's agent.py):
    run_writing_agent(
        document:    str,
        instruction: str,
        context:     dict,
        operation:   str = None,   # "generate" | "rephrase" | "improve" | None (auto-detect)
    ) -> str

Real context dict keys (from real agent's extract_writing_preferences + build_prompt_values):
    writing_style:  str   — "academic" | "formal" | "semi-formal" | "technical"
    tone:           str   — "formal" | "academic" | "neutral" | "concise"
    target_journal: str   — e.g. "IEEE Transactions", "Nature", ""
    language:       str   — e.g. "English", "French"
    citation_style: str   — "APA" | "IEEE" | "MLA"
    grounded_only:  bool  — if True, only use provided sources
    sources:        list  — pre-fetched sources from literature agent (may be empty)

Real operation detection logic (from _detect_operation):
    "rephrase"  ← rephrase / rewrite / reword / paraphrase
    "improve"   ← improve / enhance / fix / correct / polish / proofread / make formal / fix grammar
    "generate"  ← write / generate / create / draft / compose / produce / expand / add
    fallback    ← "improve" if document exists, "generate" if empty

Real output contract:
    - Returns plain string
    - Returns "[ERROR] ..." string on failure (never raises)
    - May contain inline citations if sources were passed via context["sources"]

This mock reproduces the same operation detection logic and returns realistic
placeholder text so the orchestrator works end-to-end before the branch merge.
"""

from __future__ import annotations

from typing import Optional


# ── Mirrors _detect_operation from the real agent exactly ───────

def _detect_operation(instruction: str, document: str) -> str:
    lower = instruction.lower()

    if any(kw in lower for kw in ["rephrase", "rewrite", "reword", "paraphrase"]):
        return "rephrase"

    if any(kw in lower for kw in ["improve", "enhance", "fix", "correct",
                                  "polish", "proofread", "make formal",
                                  "make academic", "fix grammar"]):
        return "improve"

    if any(kw in lower for kw in ["write", "generate", "create", "draft",
                                  "compose", "produce", "expand", "add"]):
        return "generate"

    return "improve" if document.strip() else "generate"


# ── Mirrors format_prefetched_sources from the real agent ───────

def _format_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = ["Referenced sources:"]
    for i, s in enumerate(sources, 1):
        authors = s.get("authors", s.get("author", "Unknown"))
        year = s.get("year", "n.d.")
        title = s.get("title", "Untitled")
        lines.append(f"  [{i}] {authors} ({year}). {title}.")
    return "\n".join(lines)


# ── Mirrors inject_source_citations from the real agent ─────────

def _inject_citations(text: str, sources: list, style: str) -> str:
    """Append a references block the same way the real agent does."""
    if not sources:
        return text
    refs = ["\n\n## References"]
    for i, s in enumerate(sources, 1):
        authors = s.get("authors", s.get("author", "Unknown"))
        year = s.get("year", "n.d.")
        title = s.get("title", "Untitled")
        if style.upper() == "IEEE":
            refs.append(f"[{i}] {authors}, \"{title},\" {year}.")
        else:
            refs.append(f"[{i}] {authors} ({year}). {title}.")
    return text + "\n".join(refs)


# ── Mock output per operation ────────────────────────────────────

def _mock_generate(instruction: str, context: dict) -> str:
    style = context.get("writing_style", "formal")
    tone = context.get("tone", "academic")
    journal = context.get("target_journal", "")
    lang = context.get("language", "English")
    journal_note = f" targeting {journal}" if journal else ""

    return (
        f"[MOCK — generate | {style}/{tone}{journal_note} | {lang}]\n\n"
        "This paper presents a novel framework for AI-assisted academic writing that "
        "integrates large language models with a human-in-the-loop approval workflow. "
        "The proposed system enables researchers to draft, refine, and validate "
        "individual paper sections while retaining full editorial control over the "
        "final content. Experimental results demonstrate a significant reduction in "
        "drafting time without compromising the quality or originality of the output.\n\n"
        f"*(Mock output for instruction: \"{instruction[:80]}\" — "
        "replace with real writing agent after branch merge)*"
    )


def _mock_rephrase(document: str, instruction: str, context: dict) -> str:
    style = context.get("writing_style", "formal")
    tone = context.get("tone", "academic")
    preview = document[:120].rstrip() + ("…" if len(document) > 120 else "")

    return (
        f"[MOCK — rephrase | {style}/{tone}]\n\n"
        f"Original: {preview}\n\n"
        "Rephrased: The proposed methodology leverages transformer-based language "
        "models to iteratively refine researcher-provided drafts, producing stylistically "
        "consistent and academically rigorous text aligned with the target journal's "
        "formatting conventions.\n\n"
        f"*(Mock rephrase for instruction: \"{instruction[:80]}\" — "
        "replace with real writing agent after branch merge)*"
    )


def _mock_improve(document: str, instruction: str, context: dict) -> str:
    style = context.get("writing_style", "formal")
    tone = context.get("tone", "academic")

    return (
        f"[MOCK — improve | {style}/{tone}]\n\n"
        + document
        + "\n\n[AI improvement]: The paragraph has been restructured for clarity. "
        "Passive constructions have been replaced with active voice where appropriate, "
        "and technical terminology is now consistent with IEEE style guidelines. "
        "Transitions between sentences have been strengthened to improve logical flow.\n\n"
        f"*(Mock improve for instruction: \"{instruction[:80]}\" — "
        "replace with real writing agent after branch merge)*"
    )


# ── Public API — matches real signature exactly ──────────────────

def run_writing_agent(
    document:    str,
    instruction: str,
    context:     dict,
    operation:   Optional[str] = None,
) -> str:
    """
    MOCK — returns realistic placeholder text matching the real agent's contract.

    Args:
        document:    Current section text. Empty string = write from scratch.
        instruction: What to do, e.g. "Write an introduction about transformers".
        context:     Writing preferences + optional sources dict.
        operation:   Optional override ("generate" | "rephrase" | "improve").
                     If None, auto-detected from instruction keywords (same logic
                     as the real _detect_operation function).

    Returns:
        Plain string. Returns "[ERROR] ..." on failure (never raises).
    """
    # 1. Validate operation (mirror real agent's validation)
    valid_ops = {"generate", "rephrase", "improve"}
    op = operation if operation in valid_ops else _detect_operation(
        instruction, document)

    # 2. Extract sources from context (mirrors real _get_literature_context Flow B)
    sources = context.get("sources", [])

    # 3. Generate output based on operation
    if op == "generate":
        output = _mock_generate(instruction, context)
    elif op == "rephrase":
        output = _mock_rephrase(document, instruction, context)
    else:  # improve
        output = _mock_improve(document, instruction, context)

    # 4. Inject citations if orchestrator passed sources (mirrors real inject_source_citations)
    if sources:
        output = _inject_citations(
            output, sources, context.get("citation_style", "APA"))

    return output
