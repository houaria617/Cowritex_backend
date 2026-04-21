"""Configuration classes for the literature agent"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


@dataclass
class PDFMetadata:
    """Metadata extracted from PDF files"""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: str = "n.d."
    filename: str = ""

    def get_author_string(self) -> str:
        if self.authors:
            if len(self.authors) == 1:
                return self.authors[0]
            elif len(self.authors) == 2:
                return f"{self.authors[0]} and {self.authors[1]}"
            elif len(self.authors) <= 3:
                return ", ".join(self.authors)
            else:
                return f"{self.authors[0]} et al."
        name = self.filename.replace('.pdf', '').replace('_', ' ').title()
        if len(name.split()) > 3:
            name = ' '.join(name.split()[:3])
        return name

    def get_citation_key(self) -> str:
        author = self.get_author_string().split()[0] if self.authors else self.filename[:8]
        return f"{author}_{self.year}"


@dataclass
class DocumentChunk:
    """A chunk of text from a document with metadata"""
    text: str
    source_file: str
    source_metadata: PDFMetadata
    page_number: int
    chunk_id: str
    extraction_method: str = ""
    chunk_index: int = 0


@dataclass
class AgentConfig:
    """Configuration for the literature agent (reads from environment)"""

    papers_folder: str = os.getenv("PAPERS_FOLDER", "papers")

    chunk_size: int = 200
    chunk_overlap: int = 50

    top_k: int = 10
    diversity_alpha: float = 0.5

    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")
    llm_model: str = os.getenv("LLM_MODEL", "moonshotai/Kimi-K2-Instruct-0905")

    temperature: float = 0.2
    max_tokens: int = 1500

    confidence_threshold: float = 0.3

    # ChromaDB settings (optional)
    chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "literature_chunks")


@dataclass
class AgentInput:
    """Input parameters for the literature agent"""
    query: str
    citation_style: str = "APA"
    use_web_resources: bool = False
    use_ocr: bool = False
    web_urls: List[str] = field(default_factory=list)


@dataclass
class AgentOutput:
    """Output from the literature agent"""
    literature_review: str
    verification_report: dict
    retrieved_sources: List[dict]
    success: bool
    error_message: str = ""