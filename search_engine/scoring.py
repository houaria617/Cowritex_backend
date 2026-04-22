"""
Relevance scoring functions for paper search.
"""

import re
from typing import List, Dict, Set
from datetime import datetime

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def calculate_relevance_scores(results: List[Dict], query: str) -> List[Dict]:
    """Calculate relevance scores for papers."""
    if not results:
        return results
    
    if len(results) == 1:
        results[0]['relevance_score'] = 1.0
        return results
    
    query_terms = set(query.lower().split()) if query else set()
    
    if SKLEARN_AVAILABLE and len(results) > 1:
        try:
            return calculate_tfidf_scores(results, query)
        except Exception as e:
            print(f"[WARN] TF-IDF failed: {e}")
            return calculate_basic_scores(results, query_terms)
    else:
        return calculate_basic_scores(results, query_terms)


def calculate_tfidf_scores(results: List[Dict], query: str) -> List[Dict]:
    """Calculate scores using TF-IDF."""
    documents = []
    for paper in results:
        title = (paper.get('title') or '') * 3  # Weight title 3x
        abstract = paper.get('abstract') or ''
        documents.append(f"{title} {abstract}".lower())
    
    documents_with_query = documents + [query.lower()]
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents_with_query)
        
        if tfidf_matrix.shape[0] < 2:
            raise ValueError("Insufficient vectors")
        
        query_vector = tfidf_matrix[-1]
        paper_vectors = tfidf_matrix[:-1]
        
        similarities = cosine_similarity(paper_vectors, query_vector).flatten()
        
        # Normalize citations
        citations = []
        for p in results:
            cit = p.get('citations', 0) or 0
            if isinstance(cit, str):
                try:
                    cit = int(cit)
                except:
                    cit = 0
            citations.append(max(0, cit))
        
        max_citations = max(citations) if citations else 1
        citation_scores = [c / max_citations if max_citations > 0 else 0 for c in citations]
        
        # Normalize years
        current_year = datetime.now().year
        years = []
        for p in results:
            year = p.get('year')
            if year and isinstance(year, (int, float)):
                years.append(int(year))
            else:
                years.append(current_year - 2)
        
        max_year = max(years) if years else current_year
        min_year = min(years) if years else current_year - 10
        
        year_scores = []
        for year in years:
            if max_year > min_year:
                normalized = (year - min_year) / (max_year - min_year)
                year_scores.append(0.3 + 0.7 * normalized)
            else:
                year_scores.append(0.5)
        
        # Combine: 60% text, 25% citations, 15% recency
        for i, paper in enumerate(results):
            score = (
                similarities[i] * 0.60 + 
                citation_scores[i] * 0.25 + 
                year_scores[i] * 0.15
            )
            paper['relevance_score'] = round(max(0.0, min(1.0, score)), 4)
            
    except Exception as e:
        print(f"[WARN] TF-IDF error: {e}")
        return calculate_basic_scores(results, set(query.lower().split()))
    
    return results


def calculate_basic_scores(results: List[Dict], query_terms: Set[str]) -> List[Dict]:
    """Fallback basic scoring using keyword matching."""
    if not results:
        return results
    
    if not query_terms:
        for paper in results:
            paper['relevance_score'] = 0.5
        return results
    
    # Normalize citations
    citations = []
    for p in results:
        cit = p.get('citations', 0) or 0
        if isinstance(cit, str):
            try:
                cit = int(cit)
            except:
                cit = 0
        citations.append(max(0, cit))
    
    max_citations = max(citations) if citations else 1
    
    # Normalize years
    current_year = datetime.now().year
    years = []
    for p in results:
        year = p.get('year')
        if year and isinstance(year, (int, float)):
            years.append(int(year))
        else:
            years.append(current_year - 2)
    
    max_year = max(years) if years else current_year
    min_year = min(years) if years else current_year - 10
    
    for i, paper in enumerate(results):
        score = 0.0
        
        title = (paper.get('title') or '').lower()
        abstract = (paper.get('abstract') or '').lower()
        
        # Title match (45%)
        if query_terms and title:
            title_terms = set(title.split())
            if len(query_terms) > 0:
                title_match = len(query_terms & title_terms) / len(query_terms)
                if query.lower() in title:
                    title_match = min(1.0, title_match + 0.3)
                score += title_match * 0.45
        
        # Abstract match (30%)
        if abstract and query_terms:
            abstract_terms = set(abstract.split())
            if len(query_terms) > 0:
                abstract_match = len(query_terms & abstract_terms) / len(query_terms)
                score += abstract_match * 0.30
        
        # Citations (15%)
        citation_score = min(citations[i] / max_citations, 1.0) if max_citations > 0 else 0
        score += citation_score * 0.15
        
        # Recency (10%)
        year = years[i]
        if max_year > min_year:
            recency_score = (year - min_year) / (max_year - min_year)
        else:
            recency_score = 0.5
        score += recency_score * 0.10
        
        paper['relevance_score'] = round(max(0.0, min(1.0, score)), 4)
    
    return results