"""
PDF download utilities for academic papers.
"""

import os
import re
import time
import hashlib
import requests
from typing import List, Dict

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def generate_unique_filename(title: str, url: str) -> str:
    """Generate unique filename to avoid collisions."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    safe_title = re.sub(r'[^\w\s-]', '', title) if title else "paper"
    safe_title = re.sub(r'[-\s]+', '_', safe_title).strip('_')
    return f"{safe_title[:80]}_{url_hash}.pdf"


def download_single_pdf(
    pdf_url: str,
    title: str,
    download_folder: str = "downloaded_papers",
    verbose: bool = False,
    skip_existing: bool = True
) -> Dict:
    """Download a single PDF file."""
    result = {
        "success": False,
        "filepath": None,
        "error": None,
        "skipped": False
    }
    
    if not pdf_url:
        result["error"] = "No PDF URL provided"
        return result
    
    filename = generate_unique_filename(title, pdf_url)
    filepath = os.path.join(download_folder, filename)
    
    if skip_existing and os.path.exists(filepath):
        result["success"] = True
        result["filepath"] = filepath
        result["skipped"] = True
        if verbose:
            print(f"   ⏭️  Exists: {filename}")
        return result
    
    try:
        os.makedirs(download_folder, exist_ok=True)
        
        if verbose:
            print(f"   ⬇️  Downloading: {title[:50]}...")
        
        response = requests.get(pdf_url, headers=HEADERS, timeout=30, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(filepath)
        if file_size < 1024:
            os.remove(filepath)
            result["error"] = f"File too small ({file_size} bytes)"
            return result
        
        result["success"] = True
        result["filepath"] = filepath
        
        if verbose:
            print(f"   ✅ Downloaded: {filename} ({file_size/1024:.1f} KB)")
        
    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"   ❌ Failed: {e}")
    
    return result


def download_papers_pdfs(
    results: List[Dict],
    download_folder: str = "downloaded_papers",
    verbose: bool = False,
    skip_existing: bool = True
) -> List[Dict]:
    """Download PDFs for all papers with pdf_link."""
    for paper in results:
        pdf_link = paper.get("pdf_link")
        title = paper.get("title", "untitled")
        
        if not pdf_link:
            paper["pdf_downloaded"] = False
            paper["pdf_local_path"] = None
            continue
        
        result = download_single_pdf(
            pdf_url=pdf_link,
            title=title,
            download_folder=download_folder,
            verbose=verbose,
            skip_existing=skip_existing
        )
        
        paper["pdf_downloaded"] = result["success"]
        paper["pdf_local_path"] = result["filepath"]
        
        time.sleep(0.5)
    
    return results