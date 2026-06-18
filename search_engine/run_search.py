"""
CoWriteX Search Engine – Main Interface
Exposes: run_search(query, min_year=None, min_citations=None, ...) -> json_path
"""

import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional

from .core import search_semantic_scholar, search_google_scholar, search_paper_pdf, search_paper_abstract
from .filters import apply_filters, deduplicate
from .scoring import calculate_relevance_scores
from .arxiv_utils import enrich_with_arxiv
from .download_utils import download_papers_pdfs


def run_search(
    query: str,
    min_year: Optional[int] = None,
    min_citations: Optional[int] = None,
    max_results: int = 20,
    use_scholar: bool = True,
    require_pdf: bool = False,
    download_pdfs: bool = False,
    download_folder: str = "downloaded_papers",
    output_file: Optional[str] = None,
    verbose: bool = False
) -> str:
    """
    Search for academic papers and save results to a JSON file.
    Includes fallback searches for both PDFs and abstracts from ArXiv.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"🔍 SEARCH: '{query}'")
        print(f"{'='*70}")

    # 1. Collect results from all sources
    results = []
    
    if verbose:
        print("\n[1/5] Searching Semantic Scholar...")
    ss_results = search_semantic_scholar(query, max_results, verbose)
    results.extend(ss_results)
    
    if use_scholar:
        if verbose:
            print("\n[2/5] Searching Google Scholar...")
        gs_results = search_google_scholar(query, max_results, verbose)
        results = _merge_results(results, gs_results, verbose)
    elif verbose:
        print("\n[⚠] Google Scholar skipped")

    # 3. Enrich with ArXiv (PDFs + Abstracts)
    if verbose:
        print("\n[3/5] Enriching with ArXiv (PDFs + abstracts)...")
    results = enrich_with_arxiv(results, verbose)

    # 4. Fallback searches for missing data
    if verbose:
        print("\n[4/5] Fallback searches for missing PDFs/abstracts...")
    
    for paper in results:
        # Search for PDF if missing
        if not paper.get('pdf_link'):
            pdf = search_paper_pdf(paper, verbose)
            if pdf:
                paper['pdf_link'] = pdf
        
        # Search for abstract if missing or too short (< 100 chars)
        current_abstract = paper.get('abstract') or ""
        if len(current_abstract) < 100:
            abstract = search_paper_abstract(paper, verbose)
            if abstract:
                paper['abstract'] = abstract

    # 5. Deduplicate and filter
    if verbose:
        print("\n[5/5] Deduplicating, filtering, and scoring...")
    results = deduplicate(results)
    results = apply_filters(results, min_year, min_citations, require_pdf=require_pdf)

    # 6. Score and sort
    if results:
        results = calculate_relevance_scores(results, query)
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

    # 7. Download PDFs if requested
    if download_pdfs:
        results = download_papers_pdfs(results, download_folder, verbose)

    # 8. Build clean output
    clean_results = []
    for paper in results:
        clean_results.append({
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "citations": paper.get("citations", 0),
            "year": paper.get("year"),
            "abstract": paper.get("abstract") or "",
            "score": paper.get("relevance_score", 0.0),
            "paper_url": paper.get("url", ""),
            "google_scholar_url": paper.get("google_scholar_url", ""),
            "download_link": paper.get("pdf_link") or ""
        })

    # 9. Save to JSON
    json_path = _save_to_json(query, clean_results, {
        "min_year": min_year,
        "min_citations": min_citations,
        "require_pdf": require_pdf,
        "max_results": max_results
    }, output_file, verbose)

    if verbose:
        with_pdf = sum(1 for r in clean_results if r['download_link'])
        with_abstract = sum(1 for r in clean_results if len(r['abstract']) > 100)
        print(f"\n{'='*70}")
        print(f"✅ SEARCH COMPLETE")
        print(f"   Total papers: {len(clean_results)}")
        print(f"   With PDF: {with_pdf}")
        print(f"   With abstract: {with_abstract}")
        print(f"   Saved to: {json_path}")
        print(f"{'='*70}")

    return json_path


def _merge_results(
    semantic_results: List[Dict], 
    scholar_results: List[Dict],
    verbose: bool
) -> List[Dict]:
    """
    Merge Semantic Scholar and Google Scholar results.
    Prioritize Google Scholar PDF links, keep Semantic Scholar metadata.
    """
    ss_by_title = {}
    for r in semantic_results:
        title = r.get('title', '').lower().strip()
        if title:
            ss_by_title[title] = r
    
    for gs in scholar_results:
        title = gs.get('title', '').lower().strip()
        if not title:
            continue
            
        pdf_link = gs.get('pdf_link')
        abstract = gs.get('abstract')
        
        if title in ss_by_title:
            # Merge: Keep SS metadata, add GS data if better
            if pdf_link and not ss_by_title[title].get('pdf_link'):
                ss_by_title[title]['pdf_link'] = pdf_link
                ss_by_title[title]['source'] = 'SS + GS'
            # Add abstract if SS doesn't have one
            if abstract and not ss_by_title[title].get('abstract'):
                ss_by_title[title]['abstract'] = abstract
                if verbose:
                    print(f"   Added GS abstract: {title[:50]}...")
        else:
            semantic_results.append(gs)
    
    return semantic_results


def _save_to_json(
    query: str,
    results: List[Dict],
    filters: Dict,
    output_file: Optional[str],
    verbose: bool
) -> str:
    """Save results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = re.sub(r'[^\w\s-]', '', query)
    safe_query = re.sub(r'[-\s]+', '_', safe_query)[:50]

    if output_file:
        filename = output_file if output_file.endswith('.json') else output_file + '.json'
    else:
        filename = f"search_{safe_query}_{timestamp}.json"

    output_data = {
        "search_metadata": {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "total_results": len(results),
            "filters_applied": filters
        },
        "results": results
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n💾 Saved to: {os.path.abspath(filename)}")

    return os.path.abspath(filename)