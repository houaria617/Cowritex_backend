"""
agent.py — Writing Agent  (LangChain + Groq)

PUBLIC INTERFACE:
    from writing_agent.agent import run_writing_agent, run_suggestion_agent

    # ── Full section generation / improvement ──────────────────────────────
    result = run_writing_agent(
        document    = "existing section text, or empty string",
        instruction = "generate an introduction about RAG systems",
        context     = {
            "writing_style":  "academic",
            "tone":           "formal",
            "target_journal": "IEEE Transactions",
            "language":       "English",
            "citation_style": "APA",
            "grounded_only":  False,
            "sources":        [],
        },
        target_section     = "introduction",   # NEW — explicit section name
        preceding_sections = {"abstract": "...text..."},  # NEW — for redundancy avoidance
        next_sections      = {"methodology": "...text..."}, # NEW
    )
    # Returns: plain string

    # ── Copilot-style suggestion (accept/reject) ───────────────────────────
    suggestion = run_suggestion_agent(
        document      = "full document so far",
        target_text   = "The model was fine-tuned on",   # selected text, or ""
        context       = {...},
        # suggestion_mode is now AUTO-EXTRACTED from target_text + document — no need to pass it
        target_section     = "methodology",   # NEW
        preceding_sections = {"introduction": "..."},  # NEW
        next_sections      = {"results": "..."},       # NEW
    )
    # Returns: {"original": str, "suggestion": str, "mode": str, "diff": list}
    # or:      {"error": str}
"""

import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from writing_agent.prompts import (
    WRITING_PROMPT,
    SUGGESTION_PROMPT,
    build_prompt_values,
    build_suggestion_prompt_values,
)
from writing_agent.tools import (
    query_literature_context,
    format_prefetched_sources,
    clean_llm_output,
    inject_source_citations,
    extract_writing_preferences,
    validate_output,
    build_suggestion_diff,
)

load_dotenv()

# ─────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────


def _make_llm(temperature: float = 0.3, max_tokens: int = 1500) -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY", ""),
        temperature=temperature,
        max_tokens=max_tokens,
    )


_writing_chain = WRITING_PROMPT | _make_llm(
    temperature=0.3) | StrOutputParser()
_suggestion_chain = SUGGESTION_PROMPT | _make_llm(
    temperature=0.2, max_tokens=400) | StrOutputParser()


# ─────────────────────────────────────────────
# OPERATION DETECTION  (auto-extracted from prompt)
# ─────────────────────────────────────────────

# Intent patterns — ordered from most specific to least specific
_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("rephrase", [
        r"\b(rephrase|rewrite|reword|paraphrase|reformulate|restructure)\b",
        r"\b(say it differently|express differently|another way to say)\b",
    ]),
    ("improve", [
        r"\b(improve|enhance|fix|correct|polish|proofread|refine|revise|edit|clean up)\b",
        r"\b(make (it |more )?(formal|academic|clearer|concise|precise|fluent|coherent))\b",
        r"\b(fix (the )?(grammar|style|tone|errors|mistakes|language))\b",
        r"\b(strengthen|tighten|sharpen|elevate|upgrade)\b",
        r"\b(check|review|adjust|optimise|optimize)\b",
    ]),
    ("generate", [
        r"\b(write|generate|create|draft|compose|produce|expand|add|build|develop|suggest)\b",
        r"\b(give me|provide|come up with|formulate)\b",
        r"\b(write (a|an|the)|draft (a|an|the)|create (a|an|the))\b",
    ]),
]


def _detect_operation(instruction: str, document: str) -> str:
    """
    Auto-extract the intended operation from a free-text instruction.
    Uses ranked regex patterns instead of keyword lists so partial matches
    (e.g. 'formal' inside 'informally') are avoided.

    Falls back to 'improve' if document exists, 'generate' if it doesn't.
    """
    text = instruction.lower().strip()
    for op, patterns in _INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text):
                return op
    return "improve" if document.strip() else "generate"


def _detect_suggestion_mode(selected_text: str, document: str) -> str:
    """
    Auto-detect suggestion mode from the selected text.
    - No selection           → complete
    - Short selection (≤40w) → improve
    - Long selection (>40w)  → rephrase
    """
    if not selected_text or not selected_text.strip():
        return "complete"
    word_count = len(selected_text.split())
    return "rephrase" if word_count > 40 else "improve"


# ─────────────────────────────────────────────
# SECTION NAME NORMALISER
# ─────────────────────────────────────────────

_SECTION_ALIASES: dict[str, str] = {
    # canonical name : list of aliases (checked as substrings, lowercased)
    "abstract":         ["abstract"],
    "introduction":     ["introduction", "intro"],
    "related work":     ["related work", "related works", "literature survey",
                         "prior work", "prior art", "background work"],
    "literature review": ["literature review", "review of literature"],
    "methodology":      ["methodology", "methods", "method", "approach",
                         "proposed method", "proposed approach", "framework"],
    "results":          ["results", "experimental results", "findings", "experiments"],
    "discussion":       ["discussion", "analysis", "interpretation"],
    "conclusion":       ["conclusion", "conclusions", "summary", "closing remarks"],
}


def _normalise_section(raw: str) -> str:
    """Map an arbitrary section label to its canonical form, or return it lowercased."""
    if not raw:
        return ""
    lower = raw.lower().strip()
    for canonical, aliases in _SECTION_ALIASES.items():
        if any(alias in lower for alias in aliases):
            return canonical
    return lower


# ─────────────────────────────────────────────
# SURROUNDING SECTION SUMMARISER
# ─────────────────────────────────────────────

def _build_surrounding_context(
    preceding_sections: dict[str, str] | None,
    next_sections:      dict[str, str] | None,
    max_chars_per_section: int = 400,
) -> str:
    """
    Build a compact textual block summarising which topics are already
    covered in the surrounding sections, so the agent avoids repetition.

    Returns an empty string if neither dict is provided.
    """
    parts: list[str] = []

    if preceding_sections:
        for sec_name, text in preceding_sections.items():
            if not text or not text.strip():
                continue
            canonical = _normalise_section(sec_name)
            snippet = text.strip()[:max_chars_per_section]
            parts.append(
                f"PRECEDING — {canonical.upper()}:\n"
                f"(Do NOT repeat topics already covered here)\n"
                f"{snippet}{'...' if len(text.strip()) > max_chars_per_section else ''}"
            )

    if next_sections:
        for sec_name, text in next_sections.items():
            if not text or not text.strip():
                continue
            canonical = _normalise_section(sec_name)
            snippet = text.strip()[:max_chars_per_section]
            parts.append(
                f"UPCOMING — {canonical.upper()}:\n"
                f"(The following section will cover this — do NOT pre-empt it)\n"
                f"{snippet}{'...' if len(text.strip()) > max_chars_per_section else ''}"
            )

    if not parts:
        return ""

    return (
        "SURROUNDING SECTIONS — avoid redundancy with these:\n"
        + "\n\n".join(parts)
    )


# ─────────────────────────────────────────────
# LITERATURE CONTEXT BUILDER  (shared)
# ─────────────────────────────────────────────

def _get_literature_context(query: str, context: dict) -> tuple[str, list]:
    pre_fetched = context.get("sources", [])
    if pre_fetched:
        return format_prefetched_sources(pre_fetched), pre_fetched

    search_query = query
    if context.get("target_journal"):
        search_query += f" {context['target_journal']}"
    return query_literature_context(search_query, top_k=5), []


# ─────────────────────────────────────────────
# PUBLIC API — FULL SECTION WRITER
# ─────────────────────────────────────────────

def run_writing_agent(
    document:           str,
    instruction:        str,
    context:            dict,
    operation:          str = None,
    target_section:     str = None,   # NEW — explicit section name
    preceding_sections: dict = None,   # NEW — {section_name: text, ...}
    next_sections:      dict = None,   # NEW — {section_name: text, ...}
) -> str:
    """
    Main entry point for full section generation, improvement, or rephrasing.

    Args:
        document           : Current section text. Empty string = write from scratch.
        instruction        : Free-text instruction — the operation is AUTO-EXTRACTED
                             from it; no need to pass `operation` explicitly.
        context            : Writing preferences + optional sources.
        operation          : Optional hard override ("generate"|"rephrase"|"improve").
                             Ignored if None (auto-detected).
        target_section     : Name of the section being written/edited
                             (e.g. "introduction", "methodology"). Auto-detected from
                             instruction/document if not provided.
        preceding_sections : Dict mapping section names to their text.
                             Used to avoid repeating what came before.
        next_sections      : Dict mapping section names to their text.
                             Used to avoid pre-empting what comes next.

    Returns:
        Plain string — publication-ready prose.
        Returns "[ERROR] ..." on any failure.
    """
    prefs = extract_writing_preferences(context)

    # Operation: explicit override > auto-detect from instruction
    op = (
        operation
        if operation in ("generate", "rephrase", "improve")
        else _detect_operation(instruction, document)
    )

    # Section: explicit > auto-detect from instruction + document
    section_name = _normalise_section(target_section) if target_section else ""

    # Surrounding context block
    surrounding = _build_surrounding_context(preceding_sections, next_sections)

    lit_context, raw_sources = _get_literature_context(instruction, prefs)

    values = build_prompt_values(
        document=document,
        instruction=instruction,
        context=prefs,
        operation=op,
        literature_context=lit_context,
        target_section=section_name,      # NEW
        surrounding_context=surrounding,       # NEW
    )

    try:
        raw_output = _writing_chain.invoke(values)
    except Exception as e:
        return f"[ERROR] LLM call failed: {e}. Check GROQ_API_KEY in your .env file."

    cleaned = clean_llm_output(raw_output)

    if not validate_output(cleaned):
        return f"[ERROR] Agent returned an unusable response. Raw: {raw_output[:200]}"

    if raw_sources:
        cleaned = inject_source_citations(
            text=cleaned,
            sources=raw_sources,
            style=prefs.get("citation_style", "APA"),
        )

    return cleaned


# ─────────────────────────────────────────────
# PUBLIC API — COPILOT-STYLE SUGGESTION
# ─────────────────────────────────────────────

def run_suggestion_agent(
    document:           str,
    context:            dict,
    target_text:        str = "",
    suggestion_mode:    str = None,   # auto-detected if None
    target_section:     str = None,   # NEW — explicit section name
    preceding_sections: dict = None,   # NEW — {section_name: text, ...}
    next_sections:      dict = None,   # NEW — {section_name: text, ...}
) -> dict:
    """
    Generate a single inline AI suggestion for the researcher to accept or reject.

    Args:
        document        : Full document text so far.
        context         : Writing preferences dict.
        target_text     : Selected text to work on. Empty = complete from cursor.
        suggestion_mode : "complete"|"improve"|"rephrase"|None.
                          AUTO-DETECTED from target_text if None — no manual selection needed.
        target_section  : Name of the section (e.g. "introduction").
                          Auto-detected from document if not provided.
        preceding_sections : Dict {section_name: text} — preceding content to avoid repeating.
        next_sections      : Dict {section_name: text} — upcoming content to avoid pre-empting.

    Returns:
        On success:
            {
              "original":   str,
              "suggestion": str,
              "mode":       str,
              "diff":       list,
            }
        On failure:
            {"error": str}
    """
    prefs = extract_writing_preferences(context)

    # Mode: explicit override > auto-detect
    mode = (
        suggestion_mode
        if suggestion_mode in ("complete", "improve", "rephrase")
        else _detect_suggestion_mode(target_text, document)
    )

    # Section
    section_name = _normalise_section(target_section) if target_section else ""

    # Surrounding context (kept compact for suggestions — speed matters)
    surrounding = _build_surrounding_context(
        preceding_sections, next_sections, max_chars_per_section=250
    )

    # Literature context — limited for suggestions
    lit_context, _ = _get_literature_context(
        target_text or document[-300:], prefs)
    lit_lines = lit_context.split("\n\n")[:2]
    lit_context = "\n\n".join(lit_lines)

    values = build_suggestion_prompt_values(
        document=document,
        target_text=target_text,
        context=prefs,
        suggestion_mode=mode,
        literature_context=lit_context,
        target_section=section_name,      # NEW
        surrounding_context=surrounding,       # NEW
    )

    try:
        raw_output = _suggestion_chain.invoke(values)
    except Exception as e:
        return {"error": f"LLM call failed: {e}"}

    try:
        cleaned = clean_llm_output(raw_output)
        cleaned = cleaned.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(cleaned)

        original = parsed.get("original",   "")
        suggestion = parsed.get("suggestion", "")

        if not suggestion or not suggestion.strip():
            return {"error": f"Empty suggestion. Raw output: {raw_output[:200]}"}

        return {
            "original":   original,
            "suggestion": suggestion,
            "mode":       mode,
            "diff":       build_suggestion_diff(original, suggestion),
        }

    except (json.JSONDecodeError, KeyError) as e:
        return {"error": f"Could not parse suggestion JSON: {e}. Raw: {raw_output[:300]}"}


# ─────────────────────────────────────────────
# LOCAL TEST  —  python agent.py
# ─────────────────────────────────────────────

if __name__ == "__main__":

    ctx = {
        "writing_style":  "academic",
        "tone":           "formal",
        "target_journal": "IEEE Transactions on Neural Networks",
        "language":       "English",
        "citation_style": "IEEE",
        "grounded_only":  False,
        "sources":        [],
    }

    print("\n" + "=" * 60)
    print("TEST — GENERATE Introduction (explicit section + surrounding)")
    print("=" * 60)
    result = run_writing_agent(
        document="",
        instruction="Write an introduction for a paper about human-in-the-loop AI writing assistants for researchers.",
        context=ctx,
        target_section="introduction",
        preceding_sections={
            "abstract": (
                "This paper presents CoWriteX, a human-in-the-loop AI writing assistant "
                "designed to help researchers draft and refine research paper sections. "
                "We demonstrate significant improvements in writing quality and speed."
            )
        },
        next_sections={
            "related work": (
                "Prior systems such as GPT-based editors and Grammarly focus on surface-level "
                "corrections. Unlike these, CoWriteX provides structured section-level assistance."
            )
        },
    )
    print(result)

    print("\n" + "=" * 60)
    print("TEST — SUGGESTION: auto-detect mode (no mode passed)")
    print("=" * 60)
    s = run_suggestion_agent(
        document="This paper presents a new method for academic writing assistance using LLMs.",
        target_text="The method works well on all datasets.",
        context=ctx,
        target_section="results",
        preceding_sections={
            "methodology": "We fine-tuned LLaMA-3 on 50k academic paper sections."
        },
    )
    print(s)

    print("\n" + "=" * 60)
    print("TEST — SUGGESTION: complete (no text selected)")
    print("=" * 60)
    s = run_suggestion_agent(
        document="In recent years, large language models have",
        target_text="",
        context=ctx,
        target_section="introduction",
    )
    print(s)

    print("\n" + "=" * 60)
    print("TEST — Operation auto-extract from free-text instruction")
    print("=" * 60)
    for instr in [
        "Can you make this paragraph more formal and fix the grammar?",
        "Please reword the following sentence to be clearer.",
        "Draft a methodology section for my paper on federated learning.",
        "Polish this abstract and tighten the language.",
    ]:
        op = _detect_operation(instr, "some existing text")
        print(f"  '{instr[:60]}' → {op}")
