"""Content chunking with image references and accumulated code context."""

import re
from dataclasses import dataclass, field
from pathlib import Path

from synthetic_data.parsers.base import Document, ImageReference


@dataclass
class Chunk:
    """A chunk of content for generation with image references (not inline transcriptions)."""

    text: str
    source_path: Path
    chunk_id: int

    code_blocks: list[str] = field(default_factory=list)
    images: list[ImageReference] = field(default_factory=list)  # Images in this chunk
    metadata: dict = field(default_factory=dict)

    previous_chunk_text: str = ""
    next_chunk_text: str = ""
    accumulated_code: list[str] = field(default_factory=list)  # Code before this chunk

    @property
    def is_multimodal(self) -> bool:
        """Check if chunk contains transcribed images."""
        return any(img.transcription for img in self.images)

    @property
    def transcribed_images(self) -> list[ImageReference]:
        """Get images with transcriptions."""
        return [img for img in self.images if img.transcription]

    @property
    def token_estimate(self) -> int:
        """Rough estimate of token count (chars / 4)."""
        return len(self.text) // 4

    def get_image_by_id(self, image_id: str) -> ImageReference | None:
        """Get image reference by ID."""
        for img in self.images:
            if img.image_id == image_id:
                return img
        return None

    def build_context_with_transcriptions(
        self,
        target_image_id: str | None = None,
        include_code: bool = True,
    ) -> str:
        """Build context string with image transcriptions inserted at reference points.

        Args:
            target_image_id: If specified, emphasize this image
            include_code: Whether to include accumulated code context

        Returns:
            Context string with transcriptions inserted
        """
        text = self.text

        # Insert transcriptions at their reference points
        for img in self.images:
            if not img.transcription or not img.image_id:
                continue

            # Find reference marker in text
            marker = f"[IMAGE:{img.image_id}]"
            if marker in text:
                # Format transcription
                if target_image_id == img.image_id:
                    # Emphasized version for targeted generation
                    transcription_block = f"\n[TARGET IMAGE: {img.alt_text or 'Image'}]\n{img.transcription}\n[END TARGET IMAGE]\n"
                else:
                    # Normal version
                    transcription_block = (
                        f"\n[IMAGE: {img.alt_text or 'Image'}]\n{img.transcription}\n[END IMAGE]\n"
                    )

                # Replace marker with transcription
                text = text.replace(marker, transcription_block)

        # Build full context
        parts = []

        # Previous context
        if self.previous_chunk_text:
            parts.append(f"[Previous Context]\n{self.previous_chunk_text}")

        # Main content (with transcriptions inserted)
        parts.append(f"[Main Content]\n{text}")

        # Next context
        if self.next_chunk_text:
            parts.append(f"[Next Context]\n{self.next_chunk_text}")

        # Accumulated code (imports and earlier code)
        if include_code and self.accumulated_code:
            code_str = "\n\n".join(self.accumulated_code)
            parts.append(f"[Prior Code Context]\n```python\n{code_str}\n```")

        return "\n\n".join(parts)


class ContentChunker:
    """Split documents into chunks with image references and accumulated code context."""

    def __init__(self, max_length: int = 4096, overlap: int = 0):
        """Initialize chunker.

        Args:
            max_length: Maximum chunk length in characters
            overlap: Overlap between chunks in characters
        """
        self.max_length = max_length
        self.overlap = overlap

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split document into chunks with image references and accumulated code.

        Key changes from previous version:
        - Image transcriptions NOT embedded inline in chunk text
        - Image references marked with [IMAGE:id] placeholders
        - Accumulated code from earlier in document (not all code)
        - Code cells never split across chunks
        """
        # Insert image reference markers in content
        content_with_markers, image_map = self._insert_image_markers(document)

        # Clean content
        cleaned_content = self._clean_content(content_with_markers)

        # Split into semantic units (respecting code blocks)
        units = self._split_into_units(cleaned_content)

        # Group units into chunks
        chunks = self._group_units_into_chunks(units, document, image_map)

        # Add neighbor context
        self._add_extended_context(chunks)

        return chunks

    def _insert_image_markers(self, document: Document) -> tuple[str, dict[str, ImageReference]]:
        """Build image map from document images.

        Images are already replaced with [IMAGE:id] markers during parsing.
        This method builds the image_id -> ImageReference map.

        Returns:
            Tuple of (content, image_id_map)
        """
        image_map = {}

        for img in document.images:
            if img.image_id:
                image_map[img.image_id] = img

        return document.content, image_map

    def _split_into_units(self, content: str) -> list[dict]:
        """Split content into semantic units that respect code block boundaries.

        Returns:
            List of units, where each unit is:
            {
                "text": str,
                "type": "text" | "code",
                "code_block": str | None,  # Original code if type == "code"
            }
        """
        units = []
        lines = content.split("\n")

        in_code_block = False
        current_unit = []
        current_code = []
        current_type = "text"

        for line in lines:
            stripped = line.strip()

            # Detect code block boundaries
            if stripped.startswith("```"):
                if in_code_block:
                    # End of code block
                    current_code.append(line)
                    current_unit.append(line)

                    # Save as code unit
                    code_text = "\n".join(current_unit)
                    # Extract actual code (without ```)
                    code_lines = current_code[1:-1]  # Skip opening and closing ```
                    actual_code = "\n".join(code_lines).strip()

                    units.append(
                        {
                            "text": code_text,
                            "type": "code",
                            "code_block": actual_code if actual_code else None,
                        }
                    )

                    current_unit = []
                    current_code = []
                    in_code_block = False
                else:
                    # Start of code block - save previous text unit if any
                    if current_unit:
                        units.append(
                            {
                                "text": "\n".join(current_unit),
                                "type": "text",
                                "code_block": None,
                            }
                        )
                        current_unit = []

                    in_code_block = True
                    current_code.append(line)
                    current_unit.append(line)
                continue

            if in_code_block:
                current_code.append(line)
                current_unit.append(line)
            else:
                # Text content - break on empty lines or headers
                if stripped == "":
                    if current_unit:
                        units.append(
                            {
                                "text": "\n".join(current_unit),
                                "type": "text",
                                "code_block": None,
                            }
                        )
                        current_unit = []
                elif stripped.startswith("#"):
                    # Header - create new unit
                    if current_unit:
                        units.append(
                            {
                                "text": "\n".join(current_unit),
                                "type": "text",
                                "code_block": None,
                            }
                        )
                        current_unit = []
                    current_unit.append(line)
                else:
                    current_unit.append(line)

        # Save remaining
        if current_unit:
            units.append(
                {
                    "text": "\n".join(current_unit),
                    "type": current_type,
                    "code_block": None,
                }
            )

        return [u for u in units if u["text"].strip()]

    def _group_units_into_chunks(
        self,
        units: list[dict],
        document: Document,
        image_map: dict[str, ImageReference],
    ) -> list[Chunk]:
        """Group semantic units into chunks, never splitting code blocks."""
        chunks = []
        current_units = []
        current_length = 0
        chunk_id = 0
        accumulated_code = []  # Code seen so far in document

        for unit in units:
            unit_length = len(unit["text"])

            # Check if adding this unit would exceed max_length
            if current_length + unit_length > self.max_length and current_units:
                # Create chunk from current units
                chunk_text = "\n\n".join(u["text"] for u in current_units)
                chunk_code = [u["code_block"] for u in current_units if u["code_block"]]

                # Find images in this chunk
                chunk_images = self._find_images_in_text(chunk_text, image_map)

                chunk = Chunk(
                    text=chunk_text,
                    source_path=document.source_path,
                    chunk_id=chunk_id,
                    code_blocks=chunk_code,
                    images=chunk_images,
                    metadata=document.metadata.copy(),
                    accumulated_code=accumulated_code.copy(),  # Code before this chunk
                )
                chunks.append(chunk)
                chunk_id += 1

                # Add current code to accumulated
                accumulated_code.extend(chunk_code)

                # Reset for overlap
                if self.overlap > 0:
                    # Keep last units for overlap
                    overlap_units = self._get_overlap_units(current_units, self.overlap)
                    current_units = overlap_units
                    current_length = sum(len(u["text"]) for u in overlap_units)
                else:
                    current_units = []
                    current_length = 0

            # Add unit
            current_units.append(unit)
            current_length += unit_length

        # Create final chunk
        if current_units:
            chunk_text = "\n\n".join(u["text"] for u in current_units)
            chunk_code = [u["code_block"] for u in current_units if u["code_block"]]
            chunk_images = self._find_images_in_text(chunk_text, image_map)

            chunk = Chunk(
                text=chunk_text,
                source_path=document.source_path,
                chunk_id=chunk_id,
                code_blocks=chunk_code,
                images=chunk_images,
                metadata=document.metadata.copy(),
                accumulated_code=accumulated_code.copy(),
            )
            chunks.append(chunk)

        return chunks

    def _find_images_in_text(
        self,
        text: str,
        image_map: dict[str, ImageReference],
    ) -> list[ImageReference]:
        """Find all images referenced in text by their markers."""
        images = []
        for image_id, img_ref in image_map.items():
            marker = f"[IMAGE:{image_id}]"
            if marker in text:
                images.append(img_ref)
        return images

    def _clean_content(self, content: str) -> str:
        """Clean content by removing any remaining data URIs and unwanted HTML.

        Note: Images should already be replaced with [IMAGE:id] markers during parsing.
        This is a safety net for any remaining data URIs.
        """
        # Remove any remaining HTML img tags with data URIs
        content = re.sub(r"<img[^>]+src=['\"]data:image/[^'\"]+['\"][^>]*>", "", content)
        # Remove any remaining markdown images with data URIs
        content = re.sub(r"!\[[^\]]*\]\(data:image/[^)]+\)", "", content)

        # Remove empty HTML elements
        content = re.sub(r'<p\s+style="[^"]*">\s*$', "", content, flags=re.MULTILINE)
        content = re.sub(r"</p>\s*$", "", content, flags=re.MULTILINE)

        # Normalize whitespace
        content = re.sub(r"\n\n\n+", "\n\n", content)

        return content.strip()

    def _get_overlap_units(self, units: list[dict], overlap_length: int) -> list[dict]:
        """Get units for overlap (from end of current chunk)."""
        if overlap_length == 0:
            return []

        overlap_units = []
        overlap_chars = 0

        for unit in reversed(units):
            if overlap_chars + len(unit["text"]) > overlap_length:
                break
            overlap_units.insert(0, unit)
            overlap_chars += len(unit["text"])

        return overlap_units

    def _add_extended_context(self, chunks: list[Chunk]) -> None:
        """Add neighbor context to chunks."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_text = chunks[i - 1].text
                chunk.previous_chunk_text = prev_text[-600:] if len(prev_text) > 500 else prev_text

            if i < len(chunks) - 1:
                next_text = chunks[i + 1].text
                chunk.next_chunk_text = next_text[:600] if len(next_text) > 500 else next_text
