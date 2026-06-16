"""Utility functions for text processing and verification"""

import re
from typing import List, Tuple
import os


def split_text_with_overlap(text: str, chunk_size: int, overlap: int):
    """Split text into overlapping chunks safely"""
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
    return chunks


class HallucinationVerifier:
    """Verify that claims are grounded in source chunks"""

    def extract_numbers(self, text: str) -> List[Tuple[str, int]]:
        """
        Extract numeric claims, filtering out trivial numbers like:
        - Section numbers (1., 2.1, 3.2.1)
        - Single digits that are list markers
        - Years (optional)
        - Reference numbers inside brackets [1], [2]
        """
        if not text:
            return []

        # Remove citation markers like [CHUNK 3]
        text = re.sub(r"\[CHUNK\s*\d+\]", " ", text)

        results = []

        # Extract percentages FIRST
        percent_pattern = r"\b\d+(?:\.\d+)?%"
        percent_matches = list(re.finditer(percent_pattern, text))
        for m in percent_matches:
            results.append((m.group(0), m.start()))

        percent_spans = [(m.start(), m.end()) for m in percent_matches]

        def in_percent_span(pos):
            for s, e in percent_spans:
                if s <= pos < e:
                    return True
            return False

        # Extract standalone numbers
        number_pattern = r"\b\d+(?:\.\d+)?\b"
        for m in re.finditer(number_pattern, text):
            if in_percent_span(m.start()):
                continue
            num = m.group(0)
            pos = m.start()

            # Skip numbers that are inside square brackets (reference markers like [1])
            # Look 5 characters before and after for brackets
            context_start = max(0, pos - 5)
            context_end = min(len(text), pos + len(num) + 5)
            context = text[context_start:context_end]
            if re.search(r'\[\d+\]', context):
                continue

            # Filter: skip section numbers like "1", "2.1", "3.2.1" at start of line or after newline
            before = text[pos-2:pos] if pos >= 2 else ""
            after = text[pos+len(num):pos+len(num)+1] if pos+len(num) < len(text) else ""

            # Skip if it looks like a section header (e.g., "1.", "2.1", "3.2" at line start)
            if (before.endswith('\n') or before == '') and after == '.':
                continue
            # Skip single digits that are list markers (e.g., "1 ", "2 " at line start)
            if len(num) == 1 and before.endswith('\n') and after == ' ':
                continue
            # Skip years (optional: you can keep them by removing this condition)
            if re.match(r'19|20\d{2}', num) and len(num) == 4:
                continue

            results.append((num, pos))

        return results

    def verify_claim(self, claim: str, source_text: str, threshold: float = 0.5):
        """
        Check if claim is supported by source text using word overlap.
        Lowered default threshold to 0.5 to reduce false negatives.
        """
        if not claim or not source_text:
            return False, 0.0

        claim_clean = re.sub(r"\([^)]+\)", "", claim)
        words = set(claim_clean.lower().split())

        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "that", "this", "these",
            "those", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "from", "as", "by", "into", "during", "without",
            "it", "its", "they", "them", "their", "we", "our", "you", "your"
        }

        words = words - stopwords

        if not words:
            return True, 1.0

        source_lower = source_text.lower()
        matches = sum(1 for w in words if w in source_lower)
        coverage = matches / len(words)

        return coverage >= threshold, coverage

    def verify_numeric_claim(self, number: str, context_sentence: str, source_text: str) -> Tuple[bool, float]:
        """
        Verify a numeric claim by checking if the number appears with similar context.
        Improved to handle cases where number appears with slight variations.
        """
        if not number or not source_text:
            return False, 0.0

        # Direct match
        if number in source_text:
            # Check surrounding context (50 chars each side)
            idx = source_text.find(number)
            surrounding = source_text[max(0, idx-80):min(len(source_text), idx+80)]
            sentence_words = set(context_sentence.lower().split())
            source_words = set(surrounding.lower().split())
            common = sentence_words.intersection(source_words)
            # Higher confidence if more context words match
            if len(common) >= 3:
                return True, 0.95
            elif len(common) >= 1:
                return True, 0.75
            return True, 0.65

        # Fuzzy match: try stripping commas, spaces, etc.
        cleaned_number = number.replace(',', '').strip()
        if cleaned_number in source_text:
            return True, 0.7

        return False, 0.0


def ensure_folder_exists(folder_path: str):
    """Create folder if it doesn't exist"""
    if not folder_path:
        return
    os.makedirs(folder_path, exist_ok=True)