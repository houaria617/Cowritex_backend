"""
tools.py — Writing Agent Utilities (LangChain)

Handles:
  - Querying ChromaDB via LangChain's Chroma wrapper
  - Cleaning LLM output
  - Citation injection
  - Context dict helpers
  - Suggestion diff builder  (new — for copilot accept/reject feature)
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

CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION = "literature_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_vectorstore = None


# ─────────────────────────────────────────────
# 1. LITERATURE CONTEXT  (queries ChromaDB)
# ─────────────────────────────────────────────

def _get_vectorstore():
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
        print(
            f"[tools] ChromaDB unavailable ({e}). Running without literature context.")
        return None


def query_literature_context(query: str, top_k: int = 5) -> str:
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
        meta = doc.metadata
        authors = meta.get("authors",     "Unknown")
        year = meta.get("year",        "n.d.")
        source = meta.get("source_file", "unknown")
        page = meta.get("page_number", "?")
        title = meta.get("title",       "")
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
    if not sources:
        return text
    references = []
    for i, meta in enumerate(sources, start=1):
        authors = meta.get("authors",     "Unknown")
        year = meta.get("year",        "n.d.")
        title = meta.get("title",       "Untitled")
        src = meta.get("source_file", "")
        page = meta.get("page_number", "")
        if style.upper() == "IEEE":
            inline = f"[{i}]"
            ref = f'[{i}] {authors}, "{title}," {year}. ({src}, p.{page})'
        else:
            first = authors.split(",")[0].strip()
            suffix = " et al." if "et al." in authors or "," in authors else ""
            inline = f"({first}{suffix}, {year})"
            ref = f"{authors} ({year}). {title}. {src}, p.{page}."
        text = text.replace(f"[SOURCE {i}]", inline)
        references.append(ref)
    if references:
        text += "\n\n## References\n" + "\n".join(references)
    return text


# ─────────────────────────────────────────────
# 4. CONTEXT HELPERS
# ─────────────────────────────────────────────

def extract_writing_preferences(context: dict) -> dict:
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
    if not text or not text.strip():
        return False
    if len(text.split()) < 20:
        return False
    refusals = ["i cannot", "i can't", "i am unable",
                "as an ai", "i don't have access"]
    lower = text.lower()
    if any(r in lower[:150] for r in refusals):
        return False
    return True


def format_prefetched_sources(sources: List[dict]) -> str:
    if not sources:
        return ""
    chunks = []
    for i, src in enumerate(sources, start=1):
        authors = src.get("authors", "Unknown")
        year = src.get("year",    "n.d.")
        title = src.get("title",   "")
        source = src.get("source_file", src.get("url", "unknown"))
        page = src.get("page",    "")
        text = src.get("abstract", src.get("text", ""))
        header = f"[SOURCE {i}] {authors} ({year})"
        if title:
            header += f" — {title}"
        if source:
            header += f" — {source}"
        if page:
            header += f", p.{page}"
        chunks.append(f"{header}\n{text}")
    return "\n\n".join(chunks)


# ─────────────────────────────────────────────
# 5. SUGGESTION DIFF BUILDER  (new)
# ─────────────────────────────────────────────

def build_suggestion_diff(original: str, suggestion: str) -> list[dict]:
    """
    Build a word-level diff between original and suggestion.

    Returns a list of operations that the frontend can use to render
    a highlighted diff (e.g. strike-through removed words in red,
    new words in green):

        [
          {"type": "equal",  "text": "The model"},
          {"type": "remove", "text": "works well"},
          {"type": "add",    "text": "demonstrates strong performance"},
          {"type": "equal",  "text": "on all datasets."},
        ]

    For completions (original is empty), the whole suggestion is "add".
    """
    if not original or not original.strip():
        # Pure completion — nothing was removed
        return [{"type": "add", "text": suggestion}]

    # Word-level longest-common-subsequence diff
    orig_words = original.split()
    sugg_words = suggestion.split()

    # Build LCS table
    m, n = len(orig_words), len(sugg_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if orig_words[i - 1].lower() == sugg_words[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to build edit sequence
    edits = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and orig_words[i - 1].lower() == sugg_words[j - 1].lower():
            edits.append(("equal", sugg_words[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            edits.append(("add", sugg_words[j - 1]))
            j -= 1
        else:
            edits.append(("remove", orig_words[i - 1]))
            i -= 1

    edits.reverse()

    # Merge consecutive operations of the same type into single tokens
    result: list[dict] = []
    for op_type, word in edits:
        if result and result[-1]["type"] == op_type:
            result[-1]["text"] += " " + word
        else:
            result.append({"type": op_type, "text": word})

    return result
