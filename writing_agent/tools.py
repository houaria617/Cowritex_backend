"""
tools.py — Writing Agent Utilities (LangChain)

Handles:
  - Querying ChromaDB via LangChain's Chroma wrapper
  - Cleaning LLM output
  - Citation injection
  - Context dict helpers
"""

import re
from typing import List

# ── ChromaDB via LangChain ────────────────────────────────────────────────────
try:
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    CHROMA_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        CHROMA_AVAILABLE = True
    except ImportError:
        CHROMA_AVAILABLE = False

# ── Shared constants (must match literature agent config exactly) ─────────────
CHROMA_DB_PATH       = "./chroma_db"
CHROMA_COLLECTION    = "literature_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level cache — connect once, reuse forever
_vectorstore = None


# ─────────────────────────────────────────────
# 1. LITERATURE CONTEXT  (queries ChromaDB)
# ─────────────────────────────────────────────

def _get_vectorstore():
    """
    Connect to ChromaDB using LangChain's Chroma wrapper.
    Cached at module level — only connects once per process.
    Returns None gracefully if ChromaDB isn't ready yet.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if not CHROMA_AVAILABLE:
        print("[tools] langchain_chroma not installed — literature context disabled.")
        return None

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        _vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH,
        )
        count = _vectorstore._collection.count()
        print(f"[tools] Connected to ChromaDB — {count} chunks available.")
        return _vectorstore
    except Exception as e:
        print(f"[tools] ChromaDB unavailable ({e}). Running without literature context.")
        return None


def query_literature_context(query: str, top_k: int = 5) -> str:
    """
    Retrieve the most relevant literature chunks for a query.

    Returns formatted string like:
        [SOURCE 1] Author et al. (2023) — paper.pdf, p.3
        Content of the chunk...

    Returns empty string if ChromaDB is unavailable.
    """
    vs = _get_vectorstore()
    if vs is None:
        return ""

    try:
        docs = vs.similarity_search(query, k=top_k)
    except Exception as e:
        print(f"[tools] ChromaDB query failed: {e}")
        return ""

    if not docs:
        return ""

    chunks = []
    for i, doc in enumerate(docs, start=1):
        meta    = doc.metadata
        authors = meta.get("authors",     "Unknown")
        year    = meta.get("year",        "n.d.")
        source  = meta.get("source_file", "unknown")
        page    = meta.get("page_number", "?")
        title   = meta.get("title",       "")

        header = f"[SOURCE {i}] {authors} ({year})"
        if title:
            header += f" — {title}"
        header += f" — {source}, p.{page}"

        chunks.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n".join(chunks)


# ─────────────────────────────────────────────
# 2. OUTPUT CLEANING
# ─────────────────────────────────────────────

def clean_llm_output(raw: str) -> str:
    """
    Strip common LLM artefacts:
      - Markdown code fences
      - "Assistant:" / "AI:" prefixes
      - Runs of 3+ blank lines collapsed to 2
    """
    if not raw:
        return ""

    text = re.sub(r"```[\w]*\n?", "", raw)
    text = re.sub(r"```", "", text)
    text = re.sub(r"^(Assistant|AI)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ─────────────────────────────────────────────
# 3. CITATION INJECTION
# ─────────────────────────────────────────────

def inject_source_citations(text: str, sources: list, style: str = "APA") -> str:
    """
    Replace [SOURCE N] markers with real inline citations and append References.

    Args:
        text    : Generated text containing [SOURCE 1], [SOURCE 2] markers.
        sources : List of dicts with keys: authors, year, title, source_file, page.
        style   : "APA" or "IEEE".

    Returns:
        Text with markers replaced + References block appended.
    """
    if not sources:
        return text

    references = []

    for i, meta in enumerate(sources, start=1):
        authors = meta.get("authors",     "Unknown")
        year    = meta.get("year",        "n.d.")
        title   = meta.get("title",       "Untitled")
        src     = meta.get("source_file", "")
        page    = meta.get("page_number", "")

        if style.upper() == "IEEE":
            inline = f"[{i}]"
            ref    = f'[{i}] {authors}, "{title}," {year}. ({src}, p.{page})'
        else:
            first  = authors.split(",")[0].strip()
            suffix = " et al." if "et al." in authors or "," in authors else ""
            inline = f"({first}{suffix}, {year})"
            ref    = f"{authors} ({year}). {title}. {src}, p.{page}."

        text = text.replace(f"[SOURCE {i}]", inline)
        references.append(ref)

    if references:
        text += "\n\n## References\n" + "\n".join(references)

    return text


# ─────────────────────────────────────────────
# 4. CONTEXT HELPERS
# ─────────────────────────────────────────────

def extract_writing_preferences(context: dict) -> dict:
    """
    Safely read and normalise the context dict from the orchestrator.
    Provides sensible defaults so the agent never crashes on a missing key.
    """
    return {
        "writing_style":    context.get("writing_style",    "academic"),
        "tone":             context.get("tone",             "formal"),
        "target_journal":   context.get("target_journal",   ""),
        "language":         context.get("language",         "English"),
        "citation_style":   context.get("citation_style",   "APA"),
        "assistance_level": context.get("assistance_level", "medium"),
        "grounded_only":    context.get("grounded_only",    True),
        "sources":          context.get("sources",          []),
    }


def validate_output(text: str) -> bool:
    """
    Basic sanity check on generated text.
    Returns False if output is empty, too short, or looks like a refusal.
    """
    if not text or not text.strip():
        return False
    if len(text.split()) < 20:
        return False
    refusals = ["i cannot", "i can't", "i am unable", "as an ai", "i don't have access"]
    lower = text.lower()
    if any(r in lower[:150] for r in refusals):
        return False
    return True


def format_prefetched_sources(sources: List[dict]) -> str:
    """
    Format source dicts (passed by orchestrator via context["sources"])
    into the same [SOURCE N] format that query_literature_context() produces.
    """
    if not sources:
        return ""

    chunks = []
    for i, src in enumerate(sources, start=1):
        authors = src.get("authors", "Unknown")
        year    = src.get("year",    "n.d.")
        title   = src.get("title",   "")
        source  = src.get("source_file", src.get("url", "unknown"))
        page    = src.get("page",    "")
        text    = src.get("abstract", src.get("text", ""))

        header = f"[SOURCE {i}] {authors} ({year})"
        if title:
            header += f" — {title}"
        if source:
            header += f" — {source}"
        if page:
            header += f", p.{page}"

        chunks.append(f"{header}\n{text}")

    return "\n\n".join(chunks)