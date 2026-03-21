"""Content parser: HTML to plain text, normalization."""

import re
import hashlib
from bs4 import BeautifulSoup


class ContentParser:
    """Parse and normalize HTML content from BookStack."""

    @staticmethod
    def html_to_text(html: str) -> str:
        """Convert HTML to clean plain text."""
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        # Normalize whitespace
        lines = (line.strip() for line in text.splitlines())
        text = "\n".join(line for line in lines if line)
        return text

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize whitespace and remove control characters."""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    @staticmethod
    def compute_hash(content: str) -> str:
        """SHA-256 hash for deduplication."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
