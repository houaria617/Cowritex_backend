"""
Core search functions for Semantic Scholar and Google Scholar.
"""

import os
import re
import time
import requests
from typing import List, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False

from .filters import build_google_scholar_url
from .arxiv_utils import build_arxiv_pdf_url, search_arxiv_for_pdf, get_arxiv_abstract

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"

# Load API key from environment variable
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

HEADERS = {
    "User-Agent": "CoWriteX/1.0 (https://github.com/yourorg/cowritex)"
}
if API_KEY:
    HEADERS["x-api-key"] = API_KEY


def create_session_with_backoff() -> requests.Session:
    """Create requests session with exponential backoff for rate limits."""
    session = requests.Session()
    
    retries = Retry(
        total=5,
        backoff_factor=5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True
    )
    
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


def search_semantic_scholar(
    query: str, 
    max_results: int = 10, 
    verbose: bool = False
) -> List[Dict]:
    """
    Search Semantic Scholar API for papers.
    """
    if not API_KEY and verbose:
        print("[Semantic Scholar] WARNING: No API key. Rate limits strict.")
        print("  Get key at: https://www.semanticscholar.org/product/api")

    session = create_session_with_backoff()
    
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,year,abstract,citationCount,externalIds,openAccessPdf,url"
    }

    delay = 0.3 if API_KEY else 5.0
    if verbose:
        print(f"[Semantic Scholar] Waiting {delay}s...")
    time.sleep(delay)

    max_attempts = 3
    data = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            if verbose:
                print(f"[Semantic Scholar] Attempt {attempt}/{max_attempts}...")
            
            resp = session.get(
                SEMANTIC_SCHOLAR_API, 
                params=params, 
                headers=HEADERS, 
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            break
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = 30 * attempt
                if verbose:
                    print(f"[Semantic Scholar] Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                if attempt == max_attempts:
                    if verbose:
                        print("[Semantic Scholar] Max retries exceeded.")
                    return []
            else:
                if verbose:
                    print(f"[Semantic Scholar] HTTP error: {e}")
                return []
                
        except Exception as e:
            if verbose:
                print(f"[Semantic Scholar] Error: {e}")
            return []
    
    if data is None:
        return []

    results = []
    for paper in data.get("data", []):
        title = paper.get("title")
        if not title:
            continue
            
        authors = []
        for a in paper.get("authors", []) or []:
            name = a.get("name")
            if name:
                authors.append(name)

        year = paper.get("year")
        if year and isinstance(year, str):
            try:
                year = int(year)
            except ValueError:
                year = None

        # PDF link extraction
        pdf_link = None
        oa = paper.get("openAccessPdf") or {}
        if oa and oa.get("url"):
            pdf_link = oa["url"]
        
        if not pdf_link:
            arxiv_id = paper.get("externalIds", {}).get("ArXiv")
            if arxiv_id:
                pdf_link = build_arxiv_pdf_url(arxiv_id)

        gs_url = build_google_scholar_url(title, authors) if title else None

        results.append({
            "source": "Semantic Scholar",
            "title": title,
            "authors": authors,
            "url": paper.get("url"),
            "google_scholar_url": gs_url,
            "pdf_link": pdf_link,
            "abstract": paper.get("abstract") or "",
            "year": year,
            "citations": paper.get("citationCount", 0) or 0,
        })

    if verbose:
        print(f"[Semantic Scholar] Found {len(results)} results")
        
    return results


def search_google_scholar(
    query: str, 
    max_results: int = 10, 
    verbose: bool = False
) -> List[Dict]:
    """
    Search Google Scholar (requires `scholarly` package).
    """
    if not SCHOLARLY_AVAILABLE:
        if verbose:
            print("[Google Scholar] Skipped - install 'scholarly'")
        return []

    if verbose:
        print(f"[Google Scholar] Searching: {query}")

    results = []
    try:
        search_query = scholarly.search_pubs(query)
        
        for i, pub in enumerate(search_query):
            if i >= max_results:
                break

            bib = pub.get("bib", {}) or {}
            title = bib.get("title")
            if not title:
                continue
                
            authors = bib.get("author", [])
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(" and ")]
            elif not isinstance(authors, list):
                authors = []

            pdf_link = pub.get("eprint_url") or pub.get("pdf_url")
            
            if not pdf_link and pub.get("pub_url"):
                try:
                    filled = scholarly.fill(pub)
                    pdf_link = filled.get("eprint_url") or filled.get("pdf_url")
                except Exception:
                    pass

            year = bib.get("pub_year")
            if year and isinstance(year, str):
                try:
                    year = int(year)
                except ValueError:
                    year = None

            gs_url = build_google_scholar_url(title, authors) if title else None

            results.append({
                "source": "Google Scholar",
                "title": title,
                "authors": authors,
                "url": pub.get("pub_url"),
                "google_scholar_url": gs_url,
                "pdf_link": pdf_link,
                "abstract": bib.get("abstract") or "",
                "year": year,
                "citations": pub.get("num_citations", 0) or 0,
            })
            
            time.sleep(3)
            
    except Exception as e:
        if verbose:
            print(f"[Google Scholar] Error: {e}")

    if verbose:
        print(f"[Google Scholar] Found {len(results)} results")
        
    return results


def search_paper_pdf(paper: Dict, verbose: bool = False) -> Optional[str]:
    """
    Search for PDF link for a specific paper using multiple strategies.
    """
    title = paper.get('title', '')
    authors = paper.get('authors', [])
    
    if not title:
        return None
    
    if verbose:
        print(f"      [PDF Search] {title[:50]}...")
    
    # Strategy 1: ArXiv with full metadata
    arxiv_pdf = search_arxiv_for_pdf(title, authors, verbose=False)
    if arxiv_pdf:
        return arxiv_pdf
    
    # Strategy 2: Simplified title
    simplified = re.sub(r'[^\w\s]', ' ', title).strip()
    if simplified != title:
        arxiv_pdf = search_arxiv_for_pdf(simplified, authors[:1], verbose=False)
        if arxiv_pdf:
            return arxiv_pdf
    
    # Strategy 3: First author only
    if len(authors) > 1:
        arxiv_pdf = search_arxiv_for_pdf(title, [authors[0]], verbose=False)
        if arxiv_pdf:
            return arxiv_pdf
    
    # Strategy 4: Keywords only
    keywords = ' '.join(title.split()[:5])
    if len(keywords) > 10:
        arxiv_pdf = search_arxiv_for_pdf(keywords, [], verbose=False)
        if arxiv_pdf:
            return arxiv_pdf
    
    return None


def search_paper_abstract(paper: Dict, verbose: bool = False) -> Optional[str]:
    """
    Search for paper abstract using ArXiv when missing.
    """
    title = paper.get('title', '')
    authors = paper.get('authors', [])
    
    if not title:
        return None
    
    if verbose:
        print(f"      [Abstract Search] {title[:50]}...")
    
    arxiv_data = get_arxiv_abstract(title, authors, verbose=False)
    
    if arxiv_data and arxiv_data.get('abstract'):
        if verbose:
            print(f"      [Abstract Search] ✅ Found ({len(arxiv_data['abstract'])} chars)")
        return arxiv_data['abstract']
    
    if verbose:
        print(f"      [Abstract Search] ❌ Not found")
    
    return None