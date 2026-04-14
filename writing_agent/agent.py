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
        }
    )
    # Returns: plain string

    # ── Copilot-style suggestion (accept/reject) ───────────────────────────
    suggestion = run_suggestion_agent(
        document      = "full document so far",
        target_text   = "The model was fine-tuned on",   # selected text, or ""
        context       = {...},
        suggestion_mode = "complete",    # "complete" | "improve" | "rephrase"
    )
    # Returns: {"original": str, "suggestion": str, "mode": str}
    # or:      {"error": str}
"""

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from prompts import (
    WRITING_PROMPT,
    SUGGESTION_PROMPT,
    build_prompt_values,
    build_suggestion_prompt_values,
)
from tools import (
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

# Shared writing chain (generation / rephrase / improve)
_writing_chain = WRITING_PROMPT | _make_llm(temperature=0.3) | StrOutputParser()

# Suggestion chain uses lower temperature for tighter, more focused completions
# and lower max_tokens because we only need one suggestion, not a full section.
_suggestion_chain = SUGGESTION_PROMPT | _make_llm(temperature=0.2, max_tokens=400) | StrOutputParser()


# ─────────────────────────────────────────────
# OPERATION DETECTION
# ─────────────────────────────────────────────

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


def _detect_suggestion_mode(selected_text: str, document: str) -> str:
    """
    Auto-detect the right suggestion mode when the caller passes mode=None.

    Rules:
      - No selected text + document ends mid-sentence  → "complete"
      - Text selected (a word or sentence)              → "improve"
      - Text selected (full paragraph or more)          → "rephrase"
    """
    if not selected_text or not selected_text.strip():
        return "complete"
    word_count = len(selected_text.split())
    return "rephrase" if word_count > 40 else "improve"


# ─────────────────────────────────────────────
# LITERATURE CONTEXT BUILDER  (shared)
# ─────────────────────────────────────────────

def _get_literature_context(query: str, context: dict) -> tuple[str, list]:
    """
    Returns (formatted_string, raw_sources_list).
    Prefers pre-fetched sources from the orchestrator over ChromaDB.
    """
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
    document:    str,
    instruction: str,
    context:     dict,
    operation:   str = None,
) -> str:
    """
    Main entry point for full section generation, improvement, or rephrasing.

    Args:
        document    : Current section text. Empty string = write from scratch.
        instruction : What to do, e.g. "Write an introduction about transformers"
        context     : Writing preferences + optional sources.
        operation   : Optional override ("generate"|"rephrase"|"improve").
                      Auto-detected from instruction keywords if None.

    Returns:
        Plain string — publication-ready prose.
        Returns "[ERROR] ..." on any failure.
    """
    prefs = extract_writing_preferences(context)

    op = operation if operation in ("generate", "rephrase", "improve") \
         else _detect_operation(instruction, document)

    lit_context, raw_sources = _get_literature_context(instruction, prefs)

    values = build_prompt_values(
        document=document,
        instruction=instruction,
        context=prefs,
        operation=op,
        literature_context=lit_context,
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
    document:        str,
    context:         dict,
    target_text:     str  = "",
    suggestion_mode: str  = None,
) -> dict:
    """
    Generate a single inline AI suggestion for the researcher to accept or reject.

    This is the backend function for the copilot-like feature. The frontend calls this
    after a debounce period and displays the result as ghost text or a diff overlay.

    Args:
        document        : Full document text so far (provides coherence context).
        context         : Writing preferences dict (same shape as run_writing_agent).
        target_text     : The specific text to work on.
                            - Empty string → complete the document from where it ends.
                            - A sentence or two → improve/rephrase inline.
                            - A selected paragraph → rephrase.
        suggestion_mode : "complete" | "improve" | "rephrase" | None (auto-detect).

    Returns:
        On success:
            {
              "original":   str,   # the text that was changed (empty for completions)
              "suggestion": str,   # the AI-generated replacement or continuation
              "mode":       str,   # which mode was used
              "diff":       list,  # [{"type": "equal"|"remove"|"add", "text": str}, ...]
            }
        On failure:
            {"error": str}
    """
    prefs = extract_writing_preferences(context)

    mode = suggestion_mode if suggestion_mode in ("complete", "improve", "rephrase") \
           else _detect_suggestion_mode(target_text, document)

    # For suggestions we only pull a small context window, not full top_k=5
    lit_context, _ = _get_literature_context(target_text or document[-300:], prefs)
    # Limit literature context to 2 sources max — suggestions should be quick
    lit_lines  = lit_context.split("\n\n")[:2]
    lit_context = "\n\n".join(lit_lines)

    values = build_suggestion_prompt_values(
        document=document,
        target_text=target_text,
        context=prefs,
        suggestion_mode=mode,
        literature_context=lit_context,
    )

    try:
        raw_output = _suggestion_chain.invoke(values)
    except Exception as e:
        return {"error": f"LLM call failed: {e}"}

    # Parse the JSON the LLM was instructed to return
    try:
        cleaned = clean_llm_output(raw_output)
        # Strip any accidental markdown fences (model sometimes adds them anyway)
        cleaned = cleaned.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed  = json.loads(cleaned)

        original   = parsed.get("original",   "")
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
    print("TEST — GENERATE a structured Introduction")
    print("=" * 60)
    result = run_writing_agent(
        document="",
        instruction="Write an introduction for a paper about human-in-the-loop AI writing assistants for researchers.",
        context=ctx,
    )
    print(result)

    print("\n" + "=" * 60)
    print("TEST — SUGGESTION: improve a weak sentence")
    print("=" * 60)
    s = run_suggestion_agent(
        document="This paper presents a new method for academic writing assistance using LLMs.",
        target_text="The method works well on all datasets.",
        context=ctx,
        suggestion_mode="improve",
    )
    print(s)

    print("\n" + "=" * 60)
    print("TEST — SUGGESTION: complete a half-sentence")
    print("=" * 60)
    s = run_suggestion_agent(
        document="In recent years, large language models have",
        target_text="",  # cursor at end, no selection
        context=ctx,
        suggestion_mode="complete",
    )
    print(s)