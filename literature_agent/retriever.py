"""Vector store and retrieval logic"""

import faiss
import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

from literature_agent.config import DocumentChunk
from literature_agent.utils import split_text_with_overlap


class VectorStore:
    """FAISS vector store with MMR diversity retrieval"""

    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.chunk_embeddings: np.ndarray | None = None
        self.dimension: int | None = None

    def build_index(self, chunks: List[DocumentChunk], use_hnsw: bool = True):
        """Build FAISS index with L2-normalised embeddings for cosine similarity."""
        if not chunks:
            print("No chunks to index")
            return

        self.chunks = chunks

        print(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk.text for chunk in chunks]

        self.chunk_embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True,
        ).astype("float32")

        faiss.normalize_L2(self.chunk_embeddings)
        self.dimension = self.chunk_embeddings.shape[1]

        if use_hnsw and len(chunks) > 1000:
            print("Using HNSW index for faster retrieval")
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64
        else:
            print("Using flat index for exact cosine similarity")
            self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(self.chunk_embeddings)
        print(
            f"Index built with {self.index.ntotal} vectors of dimension {self.dimension}")

    def retrieve_mmr(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.5,
        candidate_multiplier: int = 4,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Maximum Marginal Relevance retrieval."""

        if self.index is None or len(self.chunks) == 0:
            return []

        query_embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        candidate_k = min(k * candidate_multiplier, len(self.chunks))
        scores, indices = self.index.search(query_embedding, candidate_k)

        valid_mask = indices[0] >= 0
        candidate_indices = indices[0][valid_mask].tolist()
        candidate_scores = scores[0][valid_mask].tolist()

        if not candidate_indices:
            return []

        selected: List[int] = []
        remaining_indices = list(candidate_indices)
        remaining_scores = list(candidate_scores)

        for _ in range(min(k, len(remaining_indices))):

            best_pos = -1
            best_mmr = float("-inf")

            for pos, (idx, relevance) in enumerate(zip(remaining_indices, remaining_scores)):

                if selected:
                    selected_embeddings = self.chunk_embeddings[selected]
                    candidate_embedding = self.chunk_embeddings[idx]

                    similarities = selected_embeddings @ candidate_embedding
                    max_sim = float(np.max(similarities))
                else:
                    max_sim = 0.0

                mmr = alpha * relevance - (1 - alpha) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_pos = pos

            if best_pos >= 0:
                chosen_idx = remaining_indices.pop(best_pos)
                remaining_scores.pop(best_pos)
                selected.append(chosen_idx)

        results = []

        for idx in selected:
            score = float(query_embedding[0] @ self.chunk_embeddings[idx])
            results.append((self.chunks[idx], score))

        return results

    def validate_query_relevance(
        self,
        query: str,
        threshold: float = 0.2,
        top_n: int = 10,
    ) -> Tuple[bool, float]:

        if self.index is None or len(self.chunks) == 0:
            return False, 0.0

        query_embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        n = min(top_n, len(self.chunks))
        scores, _ = self.index.search(query_embedding, n)

        valid_scores = scores[0][scores[0] >= 0]
        max_sim = float(np.max(valid_scores)) if len(valid_scores) > 0 else 0.0

        return max_sim >= threshold, max_sim


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------

def process_pdfs_with_metadata(
    folder_path: str,
    chunk_size: int,
    chunk_overlap: int,
    use_ocr: bool = False,
    llm_client: Optional[OpenAI] = None,
    llm_model: Optional[str] = None,
) -> List[DocumentChunk]:
    """
    Process all PDFs in folder, extract text and metadata.

    Args:
        folder_path: Path to folder containing PDFs
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        use_ocr: Whether to use OCR for scanned PDFs
        llm_client: OpenAI client for LLM-based metadata extraction (optional)
        llm_model: Model name for LLM-based metadata extraction (optional)
    """
    from literature_agent.tools import PDFExtractor
    import hashlib

    all_chunks: List[DocumentChunk] = []
    extractor = PDFExtractor()

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Folder does not exist: '{folder_path}'")
        return all_chunks

    pdf_files = [f for f in os.listdir(
        folder_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"❌ No PDF files found in '{folder_path}'")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Files in folder: {os.listdir(folder_path)[:10]}")
        return all_chunks

    print(f"✅ Found {len(pdf_files)} PDF file(s) to process")

    for pdf_file in sorted(pdf_files):
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"\nProcessing: {pdf_file}")

        # Extract text and metadata (pass LLM client if available)
        pages_data, metadata = extractor.extract_full_text(
            pdf_path,
            use_ocr=use_ocr,
            llm_client=llm_client,
            llm_model=llm_model
        )

        if not pages_data:
            print(f"  Warning: no text extracted from {pdf_file}")
            continue

        global_chunk_idx = 0
        doc_chunk_count = 0

        for page_num in sorted(pages_data.keys()):
            page_info = pages_data[page_num]
            page_text = page_info["text"].strip()
            extraction_method = page_info["source"]

            if not page_text:
                continue

            chunks = split_text_with_overlap(
                page_text, chunk_size, chunk_overlap)

            for chunk_text in chunks:
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                chunk_id = hashlib.md5(
                    f"{pdf_file}_{page_num}_{global_chunk_idx}".encode()
                ).hexdigest()[:8]

                doc_chunk = DocumentChunk(
                    text=chunk_text,
                    source_file=pdf_file,
                    source_metadata=metadata,
                    page_number=page_num,
                    chunk_id=chunk_id,
                    extraction_method=extraction_method,
                    chunk_index=global_chunk_idx,
                )

                all_chunks.append(doc_chunk)
                global_chunk_idx += 1
                doc_chunk_count += 1

        print(
            f"  Extracted {len(pages_data)} pages → {doc_chunk_count} chunks")

    return all_chunks


# ---------------------------------------------------------------------------
# Web ingestion
# ---------------------------------------------------------------------------

def process_web_resources(
    urls: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[DocumentChunk]:

    from config import PDFMetadata
    import hashlib
    from datetime import datetime

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        print(f"Missing dependency for web fetching: {e}")
        return []

    all_chunks: List[DocumentChunk] = []

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            " AppleWebKit/537.36 (KHTML, like Gecko)"
            " Chrome/120.0.0.0 Safari/537.36"
        )
    }

    for url in urls:
        print(f"Fetching: {url}")

        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()

            paragraphs = [p.get_text(separator=" ", strip=True)
                          for p in soup.find_all("p")]
            paragraphs = [p for p in paragraphs if len(p) > 40]

            if not paragraphs:
                body_text = soup.get_text(separator="\n", strip=True)
                paragraphs = [body_text]

            full_text = "\n\n".join(paragraphs)

            if len(full_text) < 200:
                print(
                    f"  Skipping — too little text extracted ({len(full_text)} chars)")
                continue

            metadata = PDFMetadata(
                title=soup.title.string[:100] if soup.title else url.split(
                    "/")[-1][:100],
                authors=["Web Source"],
                year=datetime.now().year,
                filename=url[:100],
            )

            chunks = split_text_with_overlap(
                full_text, chunk_size, chunk_overlap)

            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                chunk_id = hashlib.md5(
                    f"{url}_{chunk_idx}".encode()).hexdigest()[:8]

                doc_chunk = DocumentChunk(
                    text=chunk_text,
                    source_file=url,
                    source_metadata=metadata,
                    page_number=chunk_idx + 1,
                    chunk_id=chunk_id,
                    extraction_method="Web",
                    chunk_index=chunk_idx,
                )

                all_chunks.append(doc_chunk)

            print(f"  Created {len(chunks)} chunks from {url}")

        except Exception as e:
            print(f"  Failed to fetch {url}: {e}")

    return all_chunks
