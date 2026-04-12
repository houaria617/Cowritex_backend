"""
agent.py — Writing Agent  (LangChain + Groq)

PUBLIC INTERFACE:
    from writing_agent.agent import run_writing_agent

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
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from prompts import WRITING_PROMPT, build_prompt_values
from tools import (
    query_literature_context,
    format_prefetched_sources,
    clean_llm_output,
    inject_source_citations,
    extract_writing_preferences,
    validate_output,
)

load_dotenv()

# ─────────────────────────────────────────────
# LANGCHAIN CHAIN SETUP
# ─────────────────────────────────────────────

_llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    api_key=os.getenv("GROQ_API_KEY", ""),
    temperature=0.3,
    max_tokens=1500,
)

_chain = WRITING_PROMPT | _llm | StrOutputParser()


# ─────────────────────────────────────────────
# OPERATION DETECTION
# ─────────────────────────────────────────────

def _detect_operation(instruction: str, document: str) -> str:
    """Auto-detect operation from instruction keywords."""
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


# ─────────────────────────────────────────────
# LITERATURE CONTEXT BUILDER
# ─────────────────────────────────────────────

def _get_literature_context(instruction: str, context: dict) -> tuple[str, list]:
    """
    Build the literature context string and raw sources list.
    Returns: (literature_context_string, raw_sources_list)
    """
    # Flow B: orchestrator already passed sources
    pre_fetched = context.get("sources", [])
    if pre_fetched:
        return format_prefetched_sources(pre_fetched), pre_fetched

    # Flow A: query ChromaDB
    search_query = instruction
    if context.get("target_journal"):
        search_query += f" {context['target_journal']}"

    lit_context = query_literature_context(search_query, top_k=5)
    return lit_context, []


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def run_writing_agent(
    document:    str,
    instruction: str,
    context:     dict,
    operation:   str = None,
) -> str:
    """
    Main entry point called by the orchestrator.

    Args:
        document    : Current section text. Empty string = write from scratch.
        instruction : What to do, e.g. "Write an introduction about transformers"
        context     : Writing preferences + optional sources.
        operation   : Optional override ("generate"|"rephrase"|"improve").
                      If None, auto-detected from instruction.

    Returns:
        Plain string. Never raises — returns "[ERROR] ..." on failure.
    """

    # 1. Normalise context
    prefs = extract_writing_preferences(context)

    # 2. Detect operation
    op = operation if operation in ("generate", "rephrase", "improve") \
         else _detect_operation(instruction, document)

    # 3. Get literature context
    lit_context, raw_sources = _get_literature_context(instruction, prefs)

    # 4. Build prompt values
    values = build_prompt_values(
        document=document,
        instruction=instruction,
        context=prefs,
        operation=op,
        literature_context=lit_context,
    )

    # 5. Run LangChain chain
    try:
        raw_output = _chain.invoke(values)
    except Exception as e:
        return f"[ERROR] LLM call failed: {e}. Check GROQ_API_KEY in your .env file."

    # 6. Clean output
    cleaned = clean_llm_output(raw_output)

    # 7. Validate
    if not validate_output(cleaned):
        return f"[ERROR] Agent returned an unusable response. Raw: {raw_output[:200]}"

    # 8. Inject citations (only when orchestrator passed raw sources)
    if raw_sources:
        cleaned = inject_source_citations(
            text=cleaned,
            sources=raw_sources,
            style=prefs.get("citation_style", "APA"),
        )

    return cleaned


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

    tests = [
        {
            "name":        "TEST 1 — GENERATE from scratch",
            "document":    "",
            "instruction": "Write an introduction for a paper about human-in-the-loop AI writing assistants for researchers.",
        },
        {
            "name":        "TEST 2 — REPHRASE existing text",
            "document":    "AI is useful. It helps people write papers. The system is good.",
            "instruction": "Rephrase this text to make it more formal and academic.",
        },
        {
            "name":        "TEST 3 — IMPROVE a methodology paragraph",
            "document":    "The methodology uses a transformer model fine-tuned on academic data.",
            "instruction": "Improve this methodology paragraph with more technical detail.",
        },
    ]

    for t in tests:
        print("\n" + "=" * 60)
        print(t["name"])
        print("=" * 60)
        result = run_writing_agent(
            document=t["document"],
            instruction=t["instruction"],
            context=ctx,
        )
        print(result)
# ─────────────────────────────────────────────
# EXTRA TESTS — different user contexts
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 4 — French student, APA style, no journal")
print("=" * 60)
result = run_writing_agent(
    document="",
    instruction="Write an introduction about deep learning in medical imaging.",
    context={
        "writing_style":  "academic",
        "tone":           "formal",
        "language":       "French",       # ← different language
        "citation_style": "APA",          # ← different citation style
        "grounded_only":  False,
        "sources":        [],
    }
)
print(result)

print("\n" + "=" * 60)
print("TEST 5 — Orchestrator passes sources from Literature Agent")
print("=" * 60)
result = run_writing_agent(
    document="",
    instruction="Write a related work section about transformer models.",
    context={
        "writing_style":  "academic",
        "tone":           "formal",
        "language":       "English",
        "citation_style": "IEEE",
        "grounded_only":  True,           # ← STRICT: only use provided sources
        "sources": [                      # ← literature agent passed these
            {
                "title":       "Attention is All You Need",
                "authors":     "Vaswani et al.",
                "year":        "2017",
                "abstract":    "We propose the Transformer, a model architecture "
                               "based entirely on attention mechanisms, dispensing "
                               "with recurrence and convolutions entirely.",
                "source_file": "vaswani2017.pdf",
                "page":        1,
            },
            {
                "title":       "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors":     "Devlin et al.",
                "year":        "2019",
                "abstract":    "We introduce BERT, which stands for Bidirectional "
                               "Encoder Representations from Transformers, designed "
                               "to pre-train deep bidirectional representations.",
                "source_file": "devlin2019.pdf",
                "page":        1,
            },
        ],
    }
)
print(result)

print("\n" + "=" * 60)
print("TEST 6 — Force operation override (ignore instruction keywords)")
print("=" * 60)
result = run_writing_agent(
    document="The results show that the model performed well on all datasets.",
    instruction="Look at this and do something with it.",  # ambiguous instruction
    context={"grounded_only": False},
    operation="improve",   # ← force it explicitly instead of auto-detect
)
print(result)