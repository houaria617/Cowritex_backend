#!/usr/bin/env python3
"""
Full search example with JSON save and PDF link verification.
Run with: python -m search_engine.example_basic
"""

try:
    from .run_search import run_search
except ImportError:
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from search_engine.run_search import run_search


def main():
    print("=" * 70)
    print("🔍 FULL SEARCH WITH PDF DISCOVERY")
    print("=" * 70)
    
    query = "artificial intelligence medical imaging"
    
    json_path = run_search(
        query=query,
        max_results=10,
        use_scholar=True,
        require_pdf=False,
        download_pdfs=False,
        verbose=True
    )
    
    # Display results
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    with_pdf = sum(1 for r in results if r.get('download_link'))
    
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY: {len(results)} papers, {with_pdf} with PDF ({with_pdf/len(results)*100:.1f}%)")
    print(f"{'='*70}")
    
    for i, paper in enumerate(results[:5], 1):
        has_pdf = "✅" if paper.get('download_link') else "❌"
        print(f"\n{i}. {has_pdf} {paper['title'][:65]}...")
        print(f"   Score: {paper['score']:.3f} | Year: {paper['year']} | Citations: {paper['citations']}")
        print(f"   Authors: {', '.join(paper['authors'][:2])}{' et al.' if len(paper['authors']) > 2 else ''}")
        if paper.get('download_link'):
            print(f"   PDF: {paper['download_link'][:65]}...")
        print(f"   GS: {paper.get('google_scholar_url', 'N/A')[:50]}...")
    
    print(f"\n{'='*70}")
    print(f"💾 Full results: {json_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()