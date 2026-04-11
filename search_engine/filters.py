"""
Filter and deduplication functions for paper search.
"""

import re
from typing import List, Dict, Optional
from urllib.parse import quote


def build_google_scholar_url(title: str, authors: Optional[List[str]] = None) -> str:
    """Build a Google Scholar search URL for a paper."""
    if not title:
        return "https://scholar.google.com"
    
    search_terms = [title]
    if authors:
        author_names = [a.split()[-1] if ' ' in a else a for a in authors[:2]]
        search_terms.extend(author_names)
    
    query = " ".join(search_terms)
    encoded_query = quote(query)
    return f"https://scholar.google.com/scholar?q={encoded_query}"


def apply_filters(
    results: List[Dict],
    min_year: Optional[int] = None,
    min_citations: Optional[int] = None,
    max_year: Optional[int] = None,
    require_pdf: bool = False
) -> List[Dict]:
    """Apply filters to papers."""
    filtered = []
    
    for paper in results:
        year = paper.get('year')
        if year and isinstance(year, str):
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = None
        
        if min_year is not None and year is not None:
            if year < min_year:
                continue
        if max_year is not None and year is not None:
            if year > max_year:
                continue
        
        citations = paper.get('citations', 0)
        if isinstance(citations, str):
            try:
                citations = int(citations)
            except (ValueError, TypeError):
                citations = 0
        
        if min_citations is not None and citations < min_citations:
            continue
        
        if require_pdf and not paper.get('pdf_link'):
            continue
        
        filtered.append(paper)
    
    return filtered


def normalize_title(title: str) -> str:
    """Normalize title for deduplication."""
    if not title:
        return ""
    
    normalized = title.lower()
    stop_words = ['the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'and', 'or']
    words = normalized.split()
    words = [w for w in words if w not in stop_words]
    normalized = ' '.join(words)
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    
    return normalized


def deduplicate(results: List[Dict]) -> List[Dict]:
    """Remove duplicate papers by normalized title."""
    seen = set()
    deduped = []
    
    for r in results:
        if not r.get("title"):
            continue
        
        key = normalize_title(r["title"])
        year = r.get("year", 'unknown')
        unique_key = f"{key}_{year}"
        
        if key and unique_key not in seen:
            seen.add(unique_key)
            deduped.append(r)
    
    return deduped