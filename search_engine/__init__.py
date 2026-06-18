"""CoWriteX Search Engine - Academic paper search with PDF discovery."""

from .run_search import run_search
from .core import search_semantic_scholar, search_google_scholar, search_paper_pdf
from .scoring import calculate_relevance_scores

__version__ = "1.0.0"
__all__ = [
    'run_search',
    'search_semantic_scholar',
    'search_google_scholar',
    'search_paper_pdf',
    'calculate_relevance_scores'
]