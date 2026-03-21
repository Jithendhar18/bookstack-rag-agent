"""Semantic-aware text chunking."""

import re
import logging
from typing import List

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SemanticChunker:
    """Split text into semantically meaningful chunks."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        min_chunk_size: int = settings.MIN_CHUNK_SIZE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks respecting semantic boundaries."""
        if not text or len(text.strip()) < self.min_chunk_size:
            return [text.strip()] if text and text.strip() else []

        # Try to split on section headers first
        sections = self._split_on_headers(text)

        chunks = []
        for section in sections:
            if len(section) <= self.chunk_size:
                if len(section.strip()) >= self.min_chunk_size:
                    chunks.append(section.strip())
            else:
                sub_chunks = self._split_with_overlap(section)
                chunks.extend(sub_chunks)

        return chunks

    def _split_on_headers(self, text: str) -> List[str]:
        """Split text on markdown-style headers or double newlines."""
        # Split on headers (# Header, ## Header, etc.)
        parts = re.split(r"\n(?=#{1,6}\s)", text)
        if len(parts) > 1:
            return parts

        # Fall back to paragraph splitting
        parts = re.split(r"\n\n+", text)
        if len(parts) > 1:
            return self._merge_small_sections(parts)

        return [text]

    def _merge_small_sections(self, parts: List[str]) -> List[str]:
        """Merge small consecutive sections to meet minimum chunk size."""
        merged = []
        current = ""
        for part in parts:
            if len(current) + len(part) + 2 <= self.chunk_size:
                current = f"{current}\n\n{part}" if current else part
            else:
                if current and len(current.strip()) >= self.min_chunk_size:
                    merged.append(current.strip())
                current = part
        if current and len(current.strip()) >= self.min_chunk_size:
            merged.append(current.strip())
        return merged if merged else ["\n\n".join(parts)]

    def _split_with_overlap(self, text: str) -> List[str]:
        """Split text with character-level sliding window and overlap."""
        chunks = []
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
            else:
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())

                # Overlap: keep last portion
                overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap else ""
                current_chunk = f"{overlap_text} {sentence}".strip()

        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks
