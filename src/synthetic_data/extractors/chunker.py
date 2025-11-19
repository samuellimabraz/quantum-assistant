"""Content chunking for processing."""

from dataclasses import dataclass, field
from pathlib import Path

from synthetic_data.parsers.base import Document, ImageReference


@dataclass
class Chunk:
    """A chunk of content for generation."""

    text: str
    source_path: Path
    chunk_id: int

    code_blocks: list[str] = field(default_factory=list)
    images: list[ImageReference] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    previous_chunk_text: str = ""
    next_chunk_text: str = ""
    all_document_code: list[str] = field(default_factory=list)

    @property
    def is_multimodal(self) -> bool:
        """Check if chunk contains images."""
        return len(self.images) > 0

    @property
    def token_estimate(self) -> int:
        """Rough estimate of token count (chars / 4)."""
        return len(self.text) // 4

    @property
    def extended_code_context(self) -> str:
        """Get all available code from document."""
        if self.all_document_code:
            return "\n\n".join(self.all_document_code)
        return "\n\n".join(self.code_blocks) if self.code_blocks else ""


class ContentChunker:
    """Split documents into chunks for generation."""

    def __init__(self, max_length: int = 2048, overlap: int = 200):
        """
        Initialize chunker.

        Args:
            max_length: Maximum chunk length in characters
            overlap: Overlap between chunks in characters
        """
        self.max_length = max_length
        self.overlap = overlap

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split document into chunks with extended context."""
        chunks = []

        # Split content into paragraphs
        paragraphs = self._split_paragraphs(document.content)

        current_text = []
        current_length = 0
        chunk_id = 0

        for para in paragraphs:
            para_length = len(para)

            # If adding this paragraph exceeds max length, create a chunk
            if current_length + para_length > self.max_length and current_text:
                chunk = self._create_chunk(current_text, document, chunk_id)
                chunks.append(chunk)
                chunk_id += 1

                # Keep overlap
                current_text = self._get_overlap_text(current_text)
                current_length = sum(len(t) for t in current_text)

            current_text.append(para)
            current_length += para_length

        # Add remaining content
        if current_text:
            chunk = self._create_chunk(current_text, document, chunk_id)
            chunks.append(chunk)

        # Add neighbor context and full document code to each chunk
        self._add_extended_context(chunks, document)

        return chunks

    def _split_paragraphs(self, content: str) -> list[str]:
        """Split content into paragraphs."""
        # Split on double newlines, but preserve code blocks
        parts = []
        in_code_block = False
        current = []

        for line in content.split("\n"):
            if line.strip().startswith("```"):
                in_code_block = not in_code_block

            current.append(line)

            if not in_code_block and line.strip() == "":
                if current:
                    parts.append("\n".join(current))
                    current = []

        if current:
            parts.append("\n".join(current))

        return [p.strip() for p in parts if p.strip()]

    def _create_chunk(self, text_parts: list[str], document: Document, chunk_id: int) -> Chunk:
        """Create a chunk from text parts."""
        text = "\n\n".join(text_parts)

        # Extract code blocks from this chunk
        code_blocks = [cb for cb in document.code_blocks if any(cb in part for part in text_parts)]

        # Extract images that appear in this chunk
        images = [img for img in document.images if img.path in text or img.alt_text in text]

        return Chunk(
            text=text,
            source_path=document.source_path,
            chunk_id=chunk_id,
            code_blocks=code_blocks,
            images=images,
            metadata=document.metadata.copy(),
        )

    def _get_overlap_text(self, text_parts: list[str]) -> list[str]:
        """Get overlapping text from end of previous chunk."""
        overlap_text = []
        overlap_length = 0

        for part in reversed(text_parts):
            if overlap_length + len(part) > self.overlap:
                break
            overlap_text.insert(0, part)
            overlap_length += len(part)

        return overlap_text

    def _add_extended_context(self, chunks: list[Chunk], document: Document) -> None:
        """Add neighbor context and full document code to chunks."""
        all_code = document.code_blocks.copy()

        for i, chunk in enumerate(chunks):
            # Add full document code
            chunk.all_document_code = all_code

            # Add previous chunk text truncated
            if i > 0:
                prev_text = chunks[i - 1].text
                chunk.previous_chunk_text = prev_text[-500:] if len(prev_text) > 500 else prev_text

            if i < len(chunks) - 1:
                next_text = chunks[i + 1].text
                chunk.next_chunk_text = next_text[:500] if len(next_text) > 500 else next_text
