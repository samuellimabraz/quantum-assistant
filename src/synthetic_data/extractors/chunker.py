"""Content chunking for processing."""

import re
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

    # Internal: track document positions for image association
    _start_pos: int = 0
    _end_pos: int = 0

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

    def __init__(self, max_length: int = 4096, overlap: int = 0):
        """
        Initialize chunker.

        Args:
            max_length: Maximum chunk length in characters
            overlap: Overlap between chunks in characters (default 0 since we use prev/next context)
        """
        self.max_length = max_length
        self.overlap = overlap

    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Split document into chunks with extended context.

        Uses intelligent splitting that:
        - Respects semantic boundaries (paragraphs, headers, code blocks)
        - Excludes data URIs and HTML cruft
        - Adds prev/next chunk context without overlap
        - Includes all document code for reference
        - Associates images based on position in document
        """
        chunks = []

        # Clean content once for consistency
        cleaned_content = self._clean_content(document.content)

        # Split into paragraphs
        paragraphs = self._split_paragraphs(cleaned_content)

        current_text = []
        current_length = 0
        chunk_id = 0
        processed_length = 0  # Track total processed content length

        for para in paragraphs:
            para_length = len(para)

            # If adding this paragraph exceeds max length, create a chunk
            if current_length + para_length > self.max_length and current_text:
                chunk_text = "\n\n".join(current_text)
                chunk = self._create_chunk_with_position(
                    chunk_text,
                    document,
                    chunk_id,
                    processed_length,
                    processed_length + len(chunk_text),
                )
                chunks.append(chunk)
                chunk_id += 1

                # Update processed length
                processed_length += len(chunk_text)

                # Start new chunk (with optional overlap)
                if self.overlap > 0:
                    overlap_text = self._get_overlap_text(current_text)
                    overlap_content = "\n\n".join(overlap_text)
                    current_text = overlap_text
                    current_length = len(overlap_content)
                    processed_length -= len(overlap_content)  # Adjust for overlap
                else:
                    current_text = []
                    current_length = 0

            current_text.append(para)
            current_length = len("\n\n".join(current_text))

        # Add remaining content
        if current_text:
            chunk_text = "\n\n".join(current_text)
            chunk = self._create_chunk_with_position(
                chunk_text, document, chunk_id, processed_length, processed_length + len(chunk_text)
            )
            chunks.append(chunk)

        # Associate images with chunks based on document position
        self._associate_images_by_position(chunks, document, cleaned_content)

        # Add neighbor context and full document code to each chunk
        self._add_extended_context(chunks, document)

        return chunks

    def _create_chunk_with_position(
        self, text: str, document: Document, chunk_id: int, start_pos: int, end_pos: int
    ) -> Chunk:
        """
        Create a chunk with position tracking.

        Args:
            text: Chunk text content
            document: Source document
            chunk_id: Chunk identifier
            start_pos: Start position in cleaned document
            end_pos: End position in cleaned document

        Returns:
            Chunk object (images will be associated later by position)
        """
        # Extract code blocks from this chunk
        code_blocks = [cb for cb in document.code_blocks if cb in text]

        return Chunk(
            text=text,
            source_path=document.source_path,
            chunk_id=chunk_id,
            code_blocks=code_blocks,
            images=[],  # Will be populated by _associate_images_by_position
            metadata=document.metadata.copy(),
            _start_pos=start_pos,
            _end_pos=end_pos,
        )

    def _clean_content(self, content: str) -> str:
        """
        Clean content by removing data URIs and unwanted HTML.

        This ensures chunks don't contain base64 image data.
        """
        # Remove data URI images entirely
        content = re.sub(r'<img[^>]+src="data:image/[^"]+?"[^>]*>', "", content)
        content = re.sub(r"!\[.*?\]\(data:image/[^)]+\)", "", content)

        # Remove standalone HTML tags that create noise
        content = re.sub(r'<p\s+style="[^"]*">\s*$', "", content, flags=re.MULTILINE)
        content = re.sub(r"</p>\s*$", "", content, flags=re.MULTILINE)

        # Collapse multiple blank lines
        content = re.sub(r"\n\n\n+", "\n\n", content)

        return content.strip()

    def _split_paragraphs(self, content: str) -> list[str]:
        """
        Split content into semantic paragraphs.

        Respects:
        - Code blocks (kept together)
        - Headers (paragraph boundaries)
        - Double newlines (paragraph breaks)
        """
        parts = []
        in_code_block = False
        current = []

        for line in content.split("\n"):
            # Track code block state
            if line.strip().startswith("```"):
                in_code_block = not in_code_block

            # Treat headers as paragraph breaks (if not in code block)
            if not in_code_block and line.strip().startswith("#"):
                if current:
                    parts.append("\n".join(current))
                    current = []

            current.append(line)

            # Empty lines create paragraph breaks (if not in code block)
            if not in_code_block and line.strip() == "":
                if current:
                    parts.append("\n".join(current))
                    current = []

        if current:
            parts.append("\n".join(current))

        # Filter out tiny/empty parts
        return [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]

    def _get_overlap_text(self, text_parts: list[str]) -> list[str]:
        """Get overlapping text from end of previous chunk."""
        if self.overlap == 0:
            return []

        overlap_text = []
        overlap_length = 0

        for part in reversed(text_parts):
            if overlap_length + len(part) > self.overlap:
                break
            overlap_text.insert(0, part)
            overlap_length += len(part)

        return overlap_text

    def _associate_images_by_position(
        self, chunks: list[Chunk], document: Document, cleaned_content: str
    ) -> None:
        """
        Associate images with chunks based on their position in the document.

        For each image, find where it appears in the document content and assign it
        to the chunk(s) that contain or are nearest to that position.

        This is more robust than text-matching since most images don't have their
        paths or alt text explicitly mentioned in surrounding content.

        Args:
            chunks: List of chunks to associate images with
            document: Source document containing images
            cleaned_content: Cleaned document content for position matching
        """
        if not document.images or not chunks:
            return

        unassigned_images = []

        for img in document.images:
            # Skip images without transcription
            if not img.transcription:
                continue

            # Try to find image position in document using various markers
            image_pos = self._find_image_position(cleaned_content, img)

            if image_pos is None:
                # Can't find position, add to unassigned list
                unassigned_images.append(img)
                continue

            # Find the chunk that contains or is closest to this position
            best_chunk = None
            min_distance = float("inf")

            for chunk in chunks:
                # Check if image position falls within chunk bounds
                if chunk._start_pos <= image_pos <= chunk._end_pos:
                    best_chunk = chunk
                    break

                # Calculate distance to chunk
                distance = min(abs(image_pos - chunk._start_pos), abs(image_pos - chunk._end_pos))

                if distance < min_distance:
                    min_distance = distance
                    best_chunk = chunk

            if best_chunk:
                best_chunk.images.append(img)

        # Distribute unassigned images evenly across chunks
        # This ensures all transcribed images are available for generation
        if unassigned_images:
            for idx, img in enumerate(unassigned_images):
                chunk_idx = idx % len(chunks)
                chunks[chunk_idx].images.append(img)

    def _find_image_position(self, content: str, img: ImageReference) -> int:
        """
        Find the position of an image in the document content.

        Tries multiple strategies to locate the image.
        """
        # Strategy 1: Look for image path
        if img.path and img.path in content:
            return content.find(img.path)

        # Strategy 2: Look for alt text markers
        if img.alt_text:
            # Try markdown format
            marker = f"![{img.alt_text}]"
            if marker in content:
                return content.find(marker)

            # Try our clean marker format
            marker = f"[Image: {img.alt_text}]"
            if marker in content:
                return content.find(marker)

            # Try just the alt text
            if img.alt_text in content:
                return content.find(img.alt_text)

        # Strategy 3: If image has a resolved path, try to find filename
        if img.resolved_path:
            filename = Path(img.resolved_path).name
            if filename in content:
                return content.find(filename)

        return None

    def _add_extended_context(self, chunks: list[Chunk], document: Document) -> None:
        """
        Add neighbor context and full document code to chunks.

        This provides models with:
        - Previous context (last 500 chars from previous chunk)
        - Next context (first 500 chars from next chunk)
        - All code from document (for reference)
        """
        all_code = document.code_blocks.copy()

        for i, chunk in enumerate(chunks):
            # Add full document code
            chunk.all_document_code = all_code

            # Add previous chunk text (truncated)
            if i > 0:
                prev_text = chunks[i - 1].text
                chunk.previous_chunk_text = prev_text[-500:] if len(prev_text) > 500 else prev_text

            # Add next chunk text (truncated)
            if i < len(chunks) - 1:
                next_text = chunks[i + 1].text
                chunk.next_chunk_text = next_text[:500] if len(next_text) > 500 else next_text
