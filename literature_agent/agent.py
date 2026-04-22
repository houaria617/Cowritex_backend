"""Main literature agent - modular RAG system"""

import os
import re
import json
import pickle
from typing import Optional, List, Tuple
from openai import OpenAI

import chromadb
from chromadb.utils import embedding_functions

from literature_agent.config import AgentConfig, AgentInput, AgentOutput, DocumentChunk, PDFMetadata
from literature_agent.retriever import VectorStore, process_pdfs_with_metadata, process_web_resources
from literature_agent.prompts import get_literature_review_prompt, SYSTEM_PROMPT
from literature_agent.utils import ensure_folder_exists
from sentence_transformers import SentenceTransformer
import numpy as np


class LiteratureAgent:

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.vector_store = None
        self.embedding_model = None
        self.chunks = []

    def initialize(self):
        self.embedding_model = SentenceTransformer(
            self.config.embedding_model_name)
        self.vector_store = VectorStore(self.embedding_model)

    def load_documents(self, input_data: AgentInput) -> bool:
        ensure_folder_exists(self.config.papers_folder)

        if not os.path.exists(self.config.papers_folder):
            return False

        # Create LLM client for accurate metadata extraction
        try:
            llm_client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=os.environ.get("HF_TOKEN", "")
            )
            llm_model = self.config.llm_model
        except Exception:
            llm_client = None
            llm_model = None

        from literature_agent.retriever import process_pdfs_with_metadata
        pdf_chunks = process_pdfs_with_metadata(
            self.config.papers_folder,
            self.config.chunk_size,
            self.config.chunk_overlap,
            use_ocr=input_data.use_ocr,
            llm_client=llm_client,
            llm_model=llm_model
        )

        self.chunks = pdf_chunks.copy()

        if input_data.use_web_resources and input_data.web_urls:
            from literature_agent.retriever import process_web_resources
            web_chunks = process_web_resources(
                input_data.web_urls,
                self.config.chunk_size,
                self.config.chunk_overlap
            )
            self.chunks.extend(web_chunks)

        if not self.chunks:
            return False

        self.vector_store.build_index(
            self.chunks, use_hnsw=len(self.chunks) > 1000)
        return True

    def format_citations(self, review: str, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        for i, (chunk, _) in enumerate(retrieved_chunks, 1):
            meta = chunk.source_metadata
            author = meta.get_author_string()
            year = meta.year if meta.year != "n.d." else "n.d."
            citation = f"({author}, {year})"
            review = re.sub(rf'\[CHUNK\s*{i}\]', citation, review)
            review = re.sub(rf'\[CHUNK{i}\]', citation, review)
        return review

    def semantic_verification(self, claim: str, source_chunks: List[str], threshold_weak: float = 0.55, threshold_unverified: float = 0.45) -> Tuple[str, float, dict]:
        if not claim or not source_chunks:
            return 'unverified', 0.0, {}

        claim_emb = self.embedding_model.encode([claim])[0]
        chunk_embs = self.embedding_model.encode(source_chunks)
        similarities = np.dot(chunk_embs, claim_emb) / \
            (np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(claim_emb))
        max_sim = float(np.max(similarities))
        best_idx = int(np.argmax(similarities))

        if max_sim >= 0.7:
            status = 'verified'
        elif max_sim >= threshold_weak:
            status = 'partial'
        elif max_sim >= threshold_unverified:
            status = 'weak'
        else:
            status = 'unverified'

        best_source = {
            'document': source_chunks[best_idx][:200] if source_chunks else '',
            'similarity': max_sim
        }
        return status, max_sim, best_source

    def generate_review(self, input_data: AgentInput) -> AgentOutput:
        try:
            if self.embedding_model is None:
                self.initialize()

            if not self.chunks:
                if not self.load_documents(input_data):
                    return AgentOutput(
                        literature_review="",
                        verification_report={},
                        retrieved_sources=[],
                        success=False,
                        error_message="No documents found"
                    )

            retrieved = self.vector_store.retrieve_mmr(
                input_data.query,
                k=self.config.top_k,
                alpha=self.config.diversity_alpha
            )

            if not retrieved:
                return AgentOutput(
                    literature_review="",
                    verification_report={},
                    retrieved_sources=[],
                    success=False,
                    error_message="No relevant chunks retrieved"
                )

            context = "\n\n".join(
                f"[CHUNK {i+1}]\n{chunk.text}"
                for i, (chunk, _) in enumerate(retrieved)
            )

            prompt = get_literature_review_prompt(
                input_data.query,
                context,
                input_data.citation_style
            )

            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=os.environ.get("HF_TOKEN", "")
            )

            completion = client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            review = ""
            if completion and completion.choices:
                review = completion.choices[0].message.content or ""

            if not review.strip():
                review = "⚠️ Model returned empty output. Check HF_TOKEN or model name."

            # Remove any existing "## References" section that the LLM might have added
            if "## References" in review:
                review = review.split("## References")[0].strip()

            # -------- Remove references section before verification --------
            review_for_verification = review
            if "## References" in review_for_verification:
                review_for_verification = review_for_verification.split("## References")[
                    0]
            if "References" in review_for_verification:
                review_for_verification = review_for_verification.split("References")[
                    0]

            # -------- Semantic Verification --------
            raw_sentences = [s.strip() for s in review_for_verification.replace(
                '\n', ' ').split('.') if len(s.strip()) > 30]
            claim_sentences = [
                s for s in raw_sentences if not re.match(r'^\d+(\.\d+)*\s*$', s)]
            source_texts = [chunk.text for chunk, _ in retrieved]

            all_results = []
            verified = []
            partial = []
            weak = []
            unverified = []

            for sent in claim_sentences[:25]:
                status, conf, best = self.semantic_verification(
                    sent, source_texts)
                result = {
                    'claim': sent[:150] + ('...' if len(sent) > 150 else ''),
                    'status': status,
                    'confidence': conf,
                    'best_source': best
                }
                all_results.append(result)
                if status == 'verified':
                    verified.append(result)
                elif status == 'partial':
                    partial.append(result)
                elif status == 'weak':
                    weak.append(result)
                else:
                    unverified.append(result)

            verification_summary = {
                'total_claims': len(claim_sentences[:25]),
                'verified': len(verified),
                'partial': len(partial),
                'weak': len(weak),
                'unverified': len(unverified),
                'verification_rate': len(verified) / max(len(claim_sentences[:25]), 1),
                'supported_rate': (len(verified) + len(partial)) / max(len(claim_sentences[:25]), 1),
                'overall_status': 'PASS' if len(unverified) == 0 else 'REVIEW' if len(verified) > 0 else 'FAIL',
                'verified_claims': [{'claim': r['claim'], 'confidence': r['confidence']} for r in verified[:5]],
                'unverified_claims': [{'claim': r['claim'], 'confidence': r['confidence']} for r in unverified[:5]],
                'all_results': all_results
            }

            # Format citations
            review = self.format_citations(review, retrieved)

            # Store the full review with references (for citations file)
            review_with_refs = review + "\n\n---\n## References\n"
            for i, (chunk, _) in enumerate(retrieved, 1):
                meta = chunk.source_metadata
                author = meta.get_author_string()
                year = meta.year
                title = meta.title if meta.title else "Untitled"
                review_with_refs += f"[{i}] {author} ({year}). {title}. {chunk.source_file}, p.{chunk.page_number}\n"

            return AgentOutput(
                literature_review=review_with_refs,  # full with references
                verification_report=verification_summary,
                retrieved_sources=[
                    {
                        "file": c.source_file,
                        "page": c.page_number,
                        "score": s,
                        "author": c.source_metadata.get_author_string(),
                        "year": c.source_metadata.year
                    }
                    for c, s in retrieved
                ],
                success=True,
                error_message=""
            )

        except Exception as e:
            return AgentOutput(
                literature_review="",
                verification_report={},
                retrieved_sources=[],
                success=False,
                error_message=str(e)
            )

    # ========== CHROMA DB INTEGRATION ==========

    def save_to_chromadb(self, db_path: str = None, collection_name: str = None) -> None:
        """
        Store all chunks in ChromaDB for reuse by other agents.
        """
        if not self.chunks:
            raise ValueError("No chunks to save. Run load_documents() first.")

        db_path = db_path or self.config.chroma_db_path
        collection_name = collection_name or self.config.chroma_collection_name

        # Create persistent ChromaDB client
        client = chromadb.PersistentClient(path=db_path)

        # Use the same embedding model
        chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model_name
        )

        # Delete existing collection if it exists (optional)
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=collection_name,
            embedding_function=chroma_ef,
            metadata={"hnsw:space": "cosine"}
        )

        # Prepare data
        ids = [chunk.chunk_id for chunk in self.chunks]
        texts = [chunk.text for chunk in self.chunks]
        metadatas = [
            {
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "title": chunk.source_metadata.title,
                "authors": ", ".join(chunk.source_metadata.authors),
                "year": chunk.source_metadata.year,
                "chunk_index": chunk.chunk_index,
                "extraction_method": chunk.extraction_method
            }
            for chunk in self.chunks
        ]

        # Add in batches (ChromaDB handles large sets)
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        # No print, but you could log if needed

    # ========== JSON SERIALIZATION (optional) ==========

    def save_chunks_to_json(self, filepath: str) -> None:
        """Save all chunks as a JSON file."""
        data = []
        for chunk in self.chunks:
            data.append({
                "text": chunk.text,
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "extraction_method": chunk.extraction_method,
                "chunk_index": chunk.chunk_index,
                "metadata": {
                    "title": chunk.source_metadata.title,
                    "authors": chunk.source_metadata.authors,
                    "year": chunk.source_metadata.year,
                    "filename": chunk.source_metadata.filename,
                }
            })
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_chunks_from_json(filepath: str) -> List[DocumentChunk]:
        """Load chunks from a previously saved JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = []
        for item in data:
            meta = PDFMetadata(
                title=item["metadata"]["title"],
                authors=item["metadata"]["authors"],
                year=item["metadata"]["year"],
                filename=item["metadata"]["filename"]
            )
            chunk = DocumentChunk(
                text=item["text"],
                source_file=item["source_file"],
                source_metadata=meta,
                page_number=item["page_number"],
                chunk_id=item["chunk_id"],
                extraction_method=item.get("extraction_method", ""),
                chunk_index=item.get("chunk_index", 0)
            )
            chunks.append(chunk)
        return chunks


# ========== PUBLIC API ==========

def run_literature_agent(
    project_title: str,
    citation_style: str = "APA",
    use_web_resources: bool = False,
    use_ocr: bool = False,
    web_urls: Optional[List[str]] = None,
    save_to_chromadb: bool = False
) -> None:
    """
    Run the literature agent and save output to files.

    Args:
        project_title: Title of the project (used as query and for output filenames)
        citation_style: "APA" or "IEEE"
        use_web_resources: Whether to fetch web content
        use_ocr: Whether to use OCR for scanned PDFs
        web_urls: List of URLs to scrape (if use_web_resources=True)
        save_to_chromadb: If True, also save chunks to ChromaDB
    """
    input_data = AgentInput(
        query=project_title,
        citation_style=citation_style,
        use_web_resources=use_web_resources,
        use_ocr=use_ocr,
        web_urls=web_urls or []
    )
    agent = LiteratureAgent()
    output = agent.generate_review(input_data)

    os.makedirs("outputs", exist_ok=True)

    # ---- State‑of‑the‑art file (no references) ----
    review_full = output.literature_review
    if "## References" in review_full:
        review_for_art = review_full.split("## References")[0].strip()
    else:
        review_for_art = review_full

    art_path = os.path.join("outputs", f"{project_title}_state_of_the_art.txt")
    with open(art_path, "w", encoding="utf-8") as f:
        f.write(review_for_art)

    # ---- Citations file (always citations.txt) ----
    citations_path = os.path.join("outputs", "citations.txt")
    if output.success:
        citations = ""
        if "## References" in review_full:
            citations = review_full.split("## References")[-1].strip()
        elif "References" in review_full:
            citations = review_full.split("References")[-1].strip()
        with open(citations_path, "w", encoding="utf-8") as f:
            f.write(citations if citations else "No citations found.")
    else:
        with open(citations_path, "w", encoding="utf-8") as f:
            f.write("Citations unavailable due to error.")

    # ---- Optionally save to ChromaDB ----
    if save_to_chromadb and output.success and agent.chunks:
        agent.save_to_chromadb()


if __name__ == "__main__":
    import sys
    title = sys.argv[1] if len(sys.argv) > 1 else input(
        "Enter project title: ")
    run_literature_agent(project_title=title, save_to_chromadb=False)
