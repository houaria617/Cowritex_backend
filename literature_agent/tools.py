# External sources the agent can search
# arXiv, Semantic Scholar, Citation DB
"""PDF extraction, metadata, and citation tools"""

import os
import re
import json
import hashlib
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from pypdf import PdfReader
import pdfplumber

from config import PDFMetadata


class PDFExtractor:
    """Multi-engine PDF text extraction with metadata and OCR support"""

    # Class-level cache for LLM-extracted metadata
    _metadata_cache = {}

    @staticmethod
    def extract_metadata_with_llm(first_page_text: str, filename: str, llm_client, llm_model: str) -> PDFMetadata:
        """
        Use LLM to extract title, authors, and year from first page text.
        Returns PDFMetadata object.
        """
        # Check cache first
        cache_key = f"{filename}_{hashlib.md5(first_page_text[:500].encode()).hexdigest()}"
        if cache_key in PDFExtractor._metadata_cache:
            cached = PDFExtractor._metadata_cache[cache_key]
            return PDFMetadata(**cached)

        prompt = f"""
Extract the following metadata from this academic paper's first page. Return ONLY valid JSON.

First page text:
{first_page_text[:3000]}

Required JSON format:
{{
    "title": "full paper title (if not found, use null)",
    "authors": ["First Author", "Second Author", ...],
    "year": "YYYY (if not found, use null)"
}}
Do not include any other text or explanation.
"""

        try:
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            content = response.choices[0].message.content.strip()
            # Extract JSON from response (in case LLM adds extra text)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                result = json.loads(content)

            metadata = PDFMetadata(filename=filename)
            metadata.title = result.get("title") or os.path.splitext(filename)[0].replace('_', ' ').title()
            metadata.authors = result.get("authors") or ["Anonymous"]
            metadata.year = result.get("year") or "n.d."

            # Cache the result
            PDFExtractor._metadata_cache[cache_key] = metadata.__dict__
            return metadata

        except Exception as e:
            print(f"LLM metadata extraction failed for {filename}: {e}, falling back to heuristics")
            return PDFExtractor.extract_metadata_from_text(first_page_text, filename)

    @staticmethod
    def extract_metadata_from_text(text: str, filename: str) -> PDFMetadata:
        """
        Heuristic extraction from PDF content when LLM is not available.
        Works for many academic papers but less accurate than LLM.
        """
        metadata = PDFMetadata(filename=filename)
        lines = text.split('\n')

        # ---- TITLE EXTRACTION ----
        title_candidates = []
        for i, line in enumerate(lines[:20]):
            line = line.strip()
            if not line:
                continue

            skip_patterns = [
                r'(?i)^(abstract|keywords|introduction|references|doi|https?://|received|accepted|published|copyright|©|license|cc by)',
                r'^\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
                r'^vol\.?\s*\d+',
                r'^page\s+\d+',
                r'^[A-Z\s]{20,}$',
            ]
            skip = False
            for pat in skip_patterns:
                if re.search(pat, line, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue

            if 20 < len(line) < 200 and re.search(r'[a-zA-Z]', line):
                if line.endswith('.') and len(line) < 50:
                    continue
                title_candidates.append(line)

        if title_candidates:
            metadata.title = title_candidates[0]
        else:
            for line in lines[:10]:
                line = line.strip()
                if 15 < len(line) < 200 and re.search(r'[a-zA-Z]', line):
                    metadata.title = line
                    break

        # ---- AUTHOR EXTRACTION ----
        author_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)*(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)?)',
            r'(\d+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'([A-Z]\.\s*[A-Z][a-z]+(?:\s*,\s*[A-Z]\.\s*[A-Z][a-z]+)*)',
            r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'Authors?:\s*(.+?)(?:\n|$)',
            r'(.+?et\s+al\.?)',
        ]

        author_text = text[:3000]
        for pattern in author_patterns:
            match = re.search(pattern, author_text, re.MULTILINE | re.IGNORECASE)
            if match:
                raw = match.group(1)
                raw = re.sub(r'^\d+\s*', '', raw)
                raw = re.sub(r'\s+', ' ', raw).strip()
                authors = re.split(r',\s*|\s+and\s+', raw)
                authors = [a.strip() for a in authors if a.strip() and len(a) > 2]
                authors = [re.sub(r'\s+et\s+al\.?$', '', a) for a in authors]
                authors = [a for a in authors if a and not a.isdigit() and len(a) > 2]
                if authors:
                    metadata.authors = authors[:5]
                    break

        if not metadata.authors:
            name_matches = re.findall(r'\b([A-Z]\.\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', author_text)
            if name_matches:
                metadata.authors = list(dict.fromkeys(name_matches))[:3]

        # ---- YEAR EXTRACTION ----
        year_patterns = [
            r'(19|20)\d{2}',
            r'received\s+\d+\s+\w+\s+(19|20)\d{2}',
            r'accepted\s+\d+\s+\w+\s+(19|20)\d{2}',
            r'published\s+on\s+\d+\s+\w+\s+(19|20)\d{2}',
            r'copyright\s+[©]?\s*(19|20)\d{2}',
            r'volume\s+\d+,\s+(19|20)\d{2}',
        ]
        for pattern in year_patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                year_match = re.search(r'(19|20)\d{2}', match.group(0))
                if year_match:
                    metadata.year = year_match.group(0)
                    break

        # ---- CLEANUP ----
        if metadata.title:
            metadata.title = re.sub(r'\s+', ' ', metadata.title).strip()
            if metadata.title.endswith('.') and len(metadata.title) < 50:
                metadata.title = metadata.title[:-1]

        if metadata.authors:
            metadata.authors = [re.sub(r'\s+et\s+al\.?$', '', a).strip() for a in metadata.authors]

        return metadata

    @staticmethod
    def extract_metadata_from_pdf(pdf_path: str, first_page_text: str, llm_client=None, llm_model=None) -> PDFMetadata:
        """
        Extract metadata using LLM (preferred) or fallback to heuristics.
        """
        metadata = PDFMetadata(filename=os.path.basename(pdf_path))

        # Try LLM extraction first if client provided
        if llm_client and llm_model and first_page_text:
            try:
                metadata = PDFExtractor.extract_metadata_with_llm(first_page_text, metadata.filename, llm_client, llm_model)
                return metadata
            except Exception as e:
                print(f"LLM extraction failed, falling back to PDF properties: {e}")

        # Fallback to PDF properties
        try:
            doc = fitz.open(pdf_path)
            meta = doc.metadata

            if meta.get("title") and meta["title"].strip():
                metadata.title = meta["title"].strip()
            if meta.get("author") and meta["author"].strip():
                raw_authors = meta["author"]
                authors = re.split(r'[;,]\s*', raw_authors)
                authors = [a.strip() for a in authors if a.strip()]
                if authors:
                    metadata.authors = authors[:5]
            if meta.get("creationDate"):
                year_match = re.search(r'D:(\d{4})', meta["creationDate"])
                if year_match:
                    metadata.year = year_match.group(1)
            doc.close()
        except Exception:
            pass

        # Final fallback to heuristic text extraction
        if not metadata.title or not metadata.authors or metadata.year == "n.d.":
            content_meta = PDFExtractor.extract_metadata_from_text(first_page_text, metadata.filename)
            if not metadata.title and content_meta.title:
                metadata.title = content_meta.title
            if not metadata.authors and content_meta.authors:
                metadata.authors = content_meta.authors
            if metadata.year == "n.d." and content_meta.year != "n.d.":
                metadata.year = content_meta.year

        # Defaults
        if not metadata.title:
            metadata.title = os.path.splitext(metadata.filename)[0].replace('_', ' ').title()
        if not metadata.authors:
            metadata.authors = ["Anonymous"]
        if metadata.year == "n.d.":
            year_match = re.search(r'(19|20)\d{2}', metadata.filename)
            if year_match:
                metadata.year = year_match.group(0)

        return metadata

    @staticmethod
    def extract_with_pymupdf(pdf_path: str) -> Dict[int, str]:
        pages_text = {}
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text and text.strip():
                    pages_text[page_num + 1] = text.strip()
            doc.close()
        except Exception as e:
            print(f"PyMuPDF error: {e}")
        return pages_text

    @staticmethod
    def ocr_scanned_page(image_bytes: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    def extract_full_text(self, pdf_path: str, use_ocr: bool = False, llm_client=None, llm_model=None) -> Tuple[Dict[int, Dict], PDFMetadata]:
        """Extract text and metadata from PDF, optionally using LLM for metadata"""
        pages_text = self.extract_with_pymupdf(pdf_path)
        source = "PyMuPDF"

        if not pages_text:
            pages_text = self.extract_with_pdfplumber(pdf_path)
            source = "pdfplumber"

        if not pages_text:
            pages_text = self.extract_with_pypdf(pdf_path)
            source = "PyPDF"

        first_page_text = pages_text.get(1, "") if pages_text else ""
        metadata = self.extract_metadata_from_pdf(pdf_path, first_page_text, llm_client, llm_model)

        if use_ocr and (not pages_text or all(len(text) < 100 for text in pages_text.values())):
            print(f"Attempting OCR for scanned PDF: {pdf_path}")
            try:
                doc = fitz.open(pdf_path)
                pages_text = {}
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    ocr_text = self.ocr_scanned_page(img_bytes)
                    if ocr_text:
                        pages_text[page_num + 1] = ocr_text
                source = "OCR (Tesseract)"
                first_page_text = pages_text.get(1, "")
                metadata = self.extract_metadata_from_pdf(pdf_path, first_page_text, llm_client, llm_model)
                doc.close()
            except Exception as e:
                print(f"OCR processing failed: {e}")

        result = {}
        for page_num, text in pages_text.items():
            result[page_num] = {"text": text, "source": source}

        return result, metadata

    @staticmethod
    def extract_with_pdfplumber(pdf_path: str) -> Dict[int, str]:
        pages_text = {}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages_text[page_num] = text.strip()
        except Exception as e:
            print(f"pdfplumber error: {e}")
        return pages_text

    @staticmethod
    def extract_with_pypdf(pdf_path: str) -> Dict[int, str]:
        pages_text = {}
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    pages_text[page_num] = text.strip()
        except Exception as e:
            print(f"PyPDF error: {e}")
        return pages_text


class CitationFormatter:
    """Format citations in APA or IEEE style"""

    @staticmethod
    def format_apa(metadata: PDFMetadata, citation_num: int = None) -> str:
        if citation_num is None:
            citation_num = 1
        authors = metadata.get_author_string()
        year = metadata.year
        return f"[{citation_num}] {authors} ({year})"

    @staticmethod
    def format_ieee(metadata: PDFMetadata, citation_num: int) -> str:
        return f"[{citation_num}]"

    @staticmethod
    def create_citation(metadata: PDFMetadata, style: str, citation_num: Optional[int] = None) -> str:
        style = style.upper()
        if style == "APA":
            return CitationFormatter.format_apa(metadata, citation_num)
        if style == "IEEE":
            return CitationFormatter.format_ieee(metadata, citation_num or 1)
        return f"(Source: {metadata.filename})"

    @staticmethod
    def get_reference_entry(metadata: PDFMetadata, style: str, citation_num: int) -> str:
        authors = metadata.get_author_string()
        year = metadata.year
        title = metadata.title
        style = style.upper()
        if style == "APA":
            return f"{authors} ({year}). {title}."
        return f"[{citation_num}] {authors}, \"{title},\" {year}."