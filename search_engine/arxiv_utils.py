"""
arXiv API utilities for paper search and abstract retrieval.
"""

import re
import time
import requests
from typing import Optional, List, Dict
from xml.etree import ElementTree as ET

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
ARXIV_API = "http://export.arxiv.org/api/query"


def build_arxiv_pdf_url(arxiv_id: str) -> str:
    """Build direct PDF URL from an arXiv ID."""
    arxiv_id = re.sub(r'v\d+$', '', arxiv_id.strip())
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def extract_arxiv_id(url: str) -> Optional[str]:
    """Extract an arXiv ID from a URL."""
    if not url:
        return None
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s/?#]+)", url, re.IGNORECASE)
    if match:
        return match.group(1).replace(".pdf", "")
    return None


def search_arxiv_for_pdf(title: str, authors: List[str], verbose: bool = False) -> Optional[str]:
    """
    Query the arXiv API for a PDF link.
    """
    if not title:
        return None
    
    clean_title = re.sub(r'[^\w\s]', ' ', title).strip()
    
    query_parts = []
    if clean_title:
        title_words = clean_title.split()[:10]
        query_parts.append(f'ti:{" ".join(title_words)}')
    
    if authors and len(authors) > 0:
        first_author = authors[0].strip()
        if first_author:
            name_parts = first_author.split()
            last_name = name_parts[-1] if name_parts else first_author
            last_name = re.sub(r'[^a-zA-Z]', '', last_name)
            if last_name:
                query_parts.append(f'au:{last_name}')
    
    if not query_parts:
        return None
    
    params = {
        "search_query": " AND ".join(query_parts),
        "max_results": 3,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    try:
        time.sleep(0.5)
        resp = requests.get(ARXIV_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    # Parse XML
    try:
        root = ET.fromstring(resp.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', ns):
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'pdf' or link.get('type') == 'application/pdf':
                    pdf_url = link.get('href')
                    if pdf_url:
                        return pdf_url
            
            id_elem = entry.find('atom:id', ns)
            if id_elem is not None:
                arxiv_id = extract_arxiv_id(id_elem.text)
                if arxiv_id:
                    return build_arxiv_pdf_url(arxiv_id)
                    
    except ET.ParseError:
        pdf_links = re.findall(r'<link[^>]+title="pdf"[^>]+href="([^"]+)"', resp.text)
        if pdf_links:
            return pdf_links[0]
        
        ids = re.findall(r'<id>([^<]+)</id>', resp.text)
        for url in ids:
            arxiv_id = extract_arxiv_id(url)
            if arxiv_id:
                return build_arxiv_pdf_url(arxiv_id)

    return None


def get_arxiv_abstract(title: str, authors: List[str], verbose: bool = False) -> Optional[Dict]:
    """
    Search ArXiv and return abstract + PDF link for a paper.
    
    Returns:
        Dict with 'abstract', 'pdf_link', 'arxiv_id' or None if not found
    """
    if not title:
        return None
    
    clean_title = re.sub(r'[^\w\s]', ' ', title).strip()
    
    query_parts = []
    if clean_title:
        title_words = clean_title.split()[:10]
        query_parts.append(f'ti:{" ".join(title_words)}')
    
    if authors and len(authors) > 0:
        first_author = authors[0].strip()
        if first_author:
            name_parts = first_author.split()
            last_name = name_parts[-1] if name_parts else first_author
            last_name = re.sub(r'[^a-zA-Z]', '', last_name)
            if last_name:
                query_parts.append(f'au:{last_name}')
    
    if not query_parts:
        return None
    
    params = {
        "search_query": " AND ".join(query_parts),
        "max_results": 1,  # Just need the best match
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    try:
        time.sleep(0.3)
        resp = requests.get(ARXIV_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        if verbose:
            print(f"      [ArXiv Abstract] Request failed: {exc}")
        return None

    # Parse XML response
    try:
        root = ET.fromstring(resp.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entry = root.find('atom:entry', ns)
        if entry is None:
            return None
        
        # Get abstract
        summary_elem = entry.find('atom:summary', ns)
        abstract = summary_elem.text.strip() if summary_elem is not None else None
        
        # Get PDF link
        pdf_link = None
        for link in entry.findall('atom:link', ns):
            if link.get('title') == 'pdf' or link.get('type') == 'application/pdf':
                pdf_link = link.get('href')
                break
        
        # Get arXiv ID
        arxiv_id = None
        id_elem = entry.find('atom:id', ns)
        if id_elem is not None:
            arxiv_id = extract_arxiv_id(id_elem.text)
        
        if not pdf_link and arxiv_id:
            pdf_link = build_arxiv_pdf_url(arxiv_id)
        
        # Get published year
        published_elem = entry.find('atom:published', ns)
        year = None
        if published_elem is not None:
            year_match = re.search(r'(\d{4})', published_elem.text)
            if year_match:
                year = int(year_match.group(1))
        
        if abstract or pdf_link:
            return {
                'abstract': abstract,
                'pdf_link': pdf_link,
                'arxiv_id': arxiv_id,
                'year': year
            }
        
    except ET.ParseError as e:
        if verbose:
            print(f"      [ArXiv Abstract] XML parse error: {e}")
        
        # Fallback regex
        abstract_match = re.search(r'<summary>(.*?)</summary>', resp.text, re.DOTALL)
        pdf_match = re.search(r'<link[^>]+title="pdf"[^>]+href="([^"]+)"', resp.text)
        
        if abstract_match or pdf_match:
            return {
                'abstract': abstract_match.group(1).strip() if abstract_match else None,
                'pdf_link': pdf_match.group(1) if pdf_match else None,
                'arxiv_id': None,
                'year': None
            }
    
    return None


def enrich_with_arxiv(results: List[Dict], verbose: bool = False) -> List[Dict]:
    """Add ArXiv PDF links AND abstracts to papers missing them."""
    enriched_pdf = 0
    enriched_abstract = 0
    
    for i, paper in enumerate(results):
        needs_pdf = not paper.get("pdf_link")
        needs_abstract = not paper.get("abstract") or len(paper.get("abstract", "")) < 50
        
        if not needs_pdf and not needs_abstract:
            continue
            
        if verbose:
            missing = []
            if needs_pdf:
                missing.append("PDF")
            if needs_abstract:
                missing.append("abstract")
            print(f"    [ArXiv] [{i+1}/{len(results)}] Missing {', '.join(missing)}: {paper['title'][:45]}...")
        
        try:
            arxiv_data = get_arxiv_abstract(paper["title"], paper.get("authors", []), verbose=False)
            
            if arxiv_data:
                # Add PDF if missing
                if needs_pdf and arxiv_data.get('pdf_link'):
                    paper["pdf_link"] = arxiv_data['pdf_link']
                    enriched_pdf += 1
                    if verbose:
                        print(f"    [ArXiv]   ✅ Found PDF")
                
                # Add abstract if missing or too short
                if needs_abstract and arxiv_data.get('abstract'):
                    paper["abstract"] = arxiv_data['abstract']
                    enriched_abstract += 1
                    if verbose:
                        print(f"    [ArXiv]   ✅ Found abstract ({len(arxiv_data['abstract'])} chars)")
                
                # Update year if missing
                if not paper.get('year') and arxiv_data.get('year'):
                    paper['year'] = arxiv_data['year']
            else:
                if verbose:
                    print(f"    [ArXiv]   ❌ Not found")
                    
        except Exception as e:
            if verbose:
                print(f"    [ArXiv]   ⚠️ Error: {e}")
                
        time.sleep(0.3)
    
    if verbose:
        print(f"    [ArXiv] Enriched {enriched_pdf} PDFs, {enriched_abstract} abstracts")
    
    return results