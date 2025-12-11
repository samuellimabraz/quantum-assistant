"""Content chunking with image references and accumulated code context."""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from synthetic_data.parsers.base import Document, ImageReference


class ChunkQuality(str, Enum):
    """Quality classification for chunks."""

    HIGH = "high"  # Good content with code/images, useful for generation
    MEDIUM = "medium"  # Acceptable content, may need context
    LOW = "low"  # Import-only, TOC, or trivial content


@dataclass
class Chunk:
    """A chunk of content for generation with image references."""

    text: str
    source_path: Path
    chunk_id: int

    code_blocks: list[str] = field(default_factory=list)
    images: list[ImageReference] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    previous_chunk_text: str = ""
    next_chunk_text: str = ""
    accumulated_code: list[str] = field(default_factory=list)

    quality: ChunkQuality = ChunkQuality.MEDIUM

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
        """Build context string with ONLY target image transcription inserted.

        For multimodal samples, only the target image transcription is included.
        Other image markers are removed to avoid confusion.
        Non-multimodal samples (no target_image_id) get clean text without transcriptions.

        Args:
            target_image_id: ID of the target image for multimodal samples
            include_code: Whether to include accumulated code context

        Returns:
            Context string with target image transcription or clean text
        """
        text = self.text
        target_image_code_context = None
        target_image_transcription = None
        target_image_alt = None

        for img in self.images:
            marker = f"[IMAGE:{img.image_id}]"
            if marker not in text:
                continue

            if target_image_id and img.image_id == target_image_id and img.transcription:
                target_image_transcription = img.transcription
                target_image_alt = img.alt_text or "Image"
                if hasattr(img, "code_context") and img.code_context:
                    target_image_code_context = img.code_context
                text = text.replace(marker, "")
            else:
                text = text.replace(marker, "")

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        parts = []

        if target_image_code_context:
            parts.append(
                f"[PRIORITY - Code That Generated Target Image]\n"
                f"```python\n{target_image_code_context}\n```\n"
                f"USE THIS CODE as primary reference for implementation. "
                f"The image shows the OUTPUT of this code."
            )

        if target_image_transcription:
            if target_image_code_context:
                parts.append(
                    f"[Target Image Description - for visual reference only]\n"
                    f"{target_image_alt}: {target_image_transcription[:500]}..."
                    if len(target_image_transcription) > 500
                    else f"[Target Image Description - for visual reference only]\n"
                    f"{target_image_alt}: {target_image_transcription}"
                )
            else:
                parts.append(
                    f"[TARGET IMAGE: {target_image_alt}]\n"
                    f"{target_image_transcription}\n"
                    f"[END TARGET IMAGE]"
                )

        if self.previous_chunk_text:
            prev_text = self._extract_text_only(self.previous_chunk_text)
            if prev_text.strip():
                parts.append(f"[Previous Context]\n{prev_text}")

        parts.append(f"[Main Content]\n{text}")

        if self.next_chunk_text:
            next_text = self._extract_text_only(self.next_chunk_text)
            if next_text.strip():
                parts.append(f"[Next Context]\n{next_text}")

        if include_code and self.accumulated_code:
            code_str = "\n\n".join(self.accumulated_code)
            parts.append(f"[Prior Code Context]\n```python\n{code_str}\n```")

        return "\n\n".join(parts)

    def _extract_text_only(self, content: str) -> str:
        """Extract text content, removing code blocks to avoid duplication.

        Code belongs in accumulated_code, not in text context.

        Args:
            content: Content that may contain code blocks

        Returns:
            Text content without code blocks
        """
        # Remove markdown code blocks
        result = re.sub(r"```[\s\S]*?```", "", content)

        # Clean up excessive whitespace
        result = re.sub(r"\n{3,}", "\n\n", result)
        result = result.strip()

        return result


@dataclass
class _Unit:
    """Internal unit for chunking."""

    text: str
    unit_type: str  # "text", "code", "code_output", "major_header", "minor_header"
    code_block: str | None = None
    has_image: bool = False

    @property
    def length(self) -> int:
        return len(self.text)


class ContentChunker:
    """Split documents into chunks with image references and accumulated code context.

    Strategy:
    - Notebooks (.ipynb): Cell-aware chunking, keep code + output together
    - MDX/Text: Split on major headers (# or ##), accumulate minor headers
    - Quality detection: Mark import-only, TOC, and trivial chunks
    - Respect limits: max code blocks, max images per chunk
    """

    def __init__(
        self,
        max_length: int = 4096,
        min_length: int = 850,
        overlap: int = 0,
        max_code_blocks_per_chunk: int = 4,
        max_images_per_chunk: int = 4,
    ):
        """Initialize chunker.

        Args:
            max_length: Maximum chunk length in characters
            min_length: Minimum chunk length (chunks below this are merged)
            overlap: Overlap between chunks in characters
            max_code_blocks_per_chunk: Maximum code blocks per chunk (notebooks)
            max_images_per_chunk: Maximum images per chunk
        """
        self.max_length = max_length
        self.min_length = min_length
        self.overlap = overlap
        self.max_code_blocks = max_code_blocks_per_chunk
        self.max_images = max_images_per_chunk

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split document into chunks with image references and accumulated code."""
        content, image_map = self._build_image_map(document)

        is_notebook = document.source_path.suffix == ".ipynb"
        is_pdf = document.source_path.suffix == ".pdf"
        is_slides = document.metadata.get("pdf_type") == "slides"

        units = self._parse_into_units(content, is_notebook, is_pdf, is_slides)

        raw_chunks = self._group_units(units, document, image_map, is_notebook, is_slides)

        # Split any oversized chunks (especially from PDFs without headers)
        split_chunks = self._split_oversized_chunks(raw_chunks, document, image_map)

        # Post-process: merge chunks that are too small
        merged_chunks = self._merge_small_chunks(split_chunks, document, image_map)

        # Final pass: split any chunks that still exceed image/code limits
        enforced_chunks = self._enforce_limits(merged_chunks, document, image_map)

        # Second merge pass: clean up tiny chunks created by _enforce_limits
        # This uses force-merge mode to ensure no very small chunks remain
        final_chunks = self._merge_small_chunks(enforced_chunks, document, image_map)

        # Renumber chunk IDs
        for i, chunk in enumerate(final_chunks):
            chunk.chunk_id = i

        # Classify chunk quality
        self._classify_chunk_quality(final_chunks)

        self._add_neighbor_context(final_chunks)

        return final_chunks

    def _build_image_map(self, document: Document) -> tuple[str, dict[str, ImageReference]]:
        """Build image_id -> ImageReference map."""
        image_map = {img.image_id: img for img in document.images if img.image_id}
        return document.content, image_map

    def _parse_into_units(
        self,
        content: str,
        is_notebook: bool,
        is_pdf: bool = False,  # noqa: ARG002
        is_slides: bool = False,
    ) -> list[_Unit]:
        """Parse content into semantic units.

        For notebooks: Parse code blocks and their outputs together.
        For PDFs with slides: Use slide markers (---) as major boundaries.
        For MDX: Distinguish major headers (# ##) from minor headers (### ####).
        """
        units = []
        lines = content.split("\n")

        in_code_block = False
        code_lines = []
        text_buffer = []

        def flush_text():
            if text_buffer:
                text = "\n".join(text_buffer).strip()
                if text:
                    has_img = "[IMAGE:" in text
                    units.append(_Unit(text=text, unit_type="text", has_image=has_img))
                text_buffer.clear()

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Slide separator detection (for PDF slides)
            if is_slides and stripped == "---":
                flush_text()
                # Skip the separator, next line should be slide header
                i += 1
                continue

            # Code block handling
            if stripped.startswith("```"):
                if in_code_block:
                    # End code block
                    code_lines.append(line)
                    code_text = "\n".join(code_lines)
                    inner_lines = code_lines[1:-1]
                    actual_code = "\n".join(inner_lines).strip()

                    flush_text()
                    units.append(
                        _Unit(
                            text=code_text,
                            unit_type="code",
                            code_block=actual_code if actual_code else None,
                        )
                    )

                    # For notebooks: Check if next line is "Output:" and capture it
                    if is_notebook:
                        output_text, output_images, lines_consumed = self._capture_code_output(
                            lines, i + 1
                        )
                        if output_text:
                            has_img = "[IMAGE:" in output_text
                            units.append(
                                _Unit(
                                    text=output_text,
                                    unit_type="code_output",
                                    has_image=has_img,
                                )
                            )
                            i += lines_consumed

                    code_lines = []
                    in_code_block = False
                else:
                    # Start code block
                    flush_text()
                    in_code_block = True
                    code_lines = [line]
                i += 1
                continue

            if in_code_block:
                code_lines.append(line)
                i += 1
                continue

            # Header detection
            if stripped.startswith("#"):
                header_level = sum(1 for c in stripped if c == "#")

                # For slides, ## Slide N is a major boundary
                if is_slides and "Slide" in stripped and header_level == 2:
                    flush_text()
                    units.append(_Unit(text=line, unit_type="slide_header"))
                elif header_level <= 2:
                    flush_text()
                    units.append(_Unit(text=line, unit_type="major_header"))
                else:
                    # Minor headers - keep with text
                    text_buffer.append(line)
            else:
                text_buffer.append(line)

            i += 1

        # Flush remaining
        flush_text()

        return units

    def _capture_code_output(self, lines: list[str], start_idx: int) -> tuple[str, bool, int]:
        """Capture output after a code block (Output: line and image markers).

        Returns (output_text, has_images, lines_consumed).
        """
        if start_idx >= len(lines):
            return "", False, 0

        output_lines = []
        lines_consumed = 0
        has_images = False
        i = start_idx

        # Skip empty lines
        while i < len(lines) and not lines[i].strip():
            i += 1
            lines_consumed += 1

        if i >= len(lines):
            return "", False, lines_consumed

        # Check for "Output:" prefix
        if lines[i].strip().startswith("Output:"):
            output_lines.append(lines[i])
            i += 1
            lines_consumed += 1

            # Capture output content until next code block, header, or significant text
            while i < len(lines):
                line = lines[i].strip()

                # Stop at next code block
                if line.startswith("```"):
                    break

                # Stop at header
                if line.startswith("#"):
                    break

                # Stop at HTML block (new cell marker in notebooks)
                if line.startswith("<div") and "alert" in line.lower():
                    break

                # Capture image markers
                if "[IMAGE:" in lines[i]:
                    output_lines.append(lines[i])
                    has_images = True
                    i += 1
                    lines_consumed += 1
                    continue

                # Capture short output lines (results)
                if line and len(line) < 500:
                    output_lines.append(lines[i])
                    i += 1
                    lines_consumed += 1
                elif not line:
                    # Empty line - include but check next
                    output_lines.append(lines[i])
                    i += 1
                    lines_consumed += 1
                else:
                    break

        # Also capture image markers right after code (without Output: prefix)
        elif "[IMAGE:" in lines[i]:
            while i < len(lines) and "[IMAGE:" in lines[i]:
                output_lines.append(lines[i])
                has_images = True
                i += 1
                lines_consumed += 1

        output_text = "\n".join(output_lines).strip()
        return output_text, has_images, lines_consumed

    def _group_units(
        self,
        units: list[_Unit],
        document: Document,
        image_map: dict[str, ImageReference],
        is_notebook: bool,
        is_slides: bool = False,
    ) -> list[Chunk]:
        """Group units into chunks respecting size and content limits."""
        if not units:
            return []

        chunks = []
        current_units: list[_Unit] = []
        current_length = 0
        current_code_count = 0
        current_image_count = 0
        chunk_id = 0
        accumulated_code: list[str] = []

        for unit in units:
            # Count images in this unit
            unit_images = unit.text.count("[IMAGE:")

            # Decide whether to break before adding this unit
            should_break = False

            # 1. Would exceed max_length
            if current_units and current_length + unit.length > self.max_length:
                should_break = True

            # 2. Would exceed max code blocks (notebooks)
            elif (
                is_notebook
                and unit.unit_type == "code"
                and current_code_count >= self.max_code_blocks
            ):
                should_break = True

            # 3. Would exceed max images
            elif current_image_count + unit_images > self.max_images and current_units:
                should_break = True

            # 4. Major header with substantial content
            elif unit.unit_type == "major_header" and current_length >= self.min_length:
                should_break = True

            # 5. For notebooks: New code cell when we have enough content
            elif (
                is_notebook
                and unit.unit_type == "code"
                and current_code_count >= 2
                and current_length >= self.min_length
            ):
                should_break = True

            # 6. For slides: Each slide is a natural chunk boundary
            elif is_slides and unit.unit_type == "slide_header" and current_units:
                should_break = True

            if should_break and current_units:
                chunk = self._create_chunk_from_units(
                    current_units, document, image_map, chunk_id, accumulated_code
                )
                chunks.append(chunk)
                chunk_id += 1

                # Update accumulated code
                for u in current_units:
                    if u.code_block:
                        accumulated_code.append(u.code_block)

                # Handle overlap
                if self.overlap > 0:
                    current_units, current_length = self._get_overlap(current_units)
                    current_code_count = sum(1 for u in current_units if u.unit_type == "code")
                    current_image_count = sum(u.text.count("[IMAGE:") for u in current_units)
                else:
                    current_units = []
                    current_length = 0
                    current_code_count = 0
                    current_image_count = 0

            current_units.append(unit)
            current_length += unit.length
            if unit.unit_type == "code":
                current_code_count += 1
            current_image_count += unit_images

        # Final chunk
        if current_units:
            chunk = self._create_chunk_from_units(
                current_units, document, image_map, chunk_id, accumulated_code
            )
            chunks.append(chunk)

        return chunks

    def _split_oversized_chunks(
        self,
        chunks: list[Chunk],
        document: Document,
        image_map: dict[str, ImageReference],
    ) -> list[Chunk]:
        """Split chunks that exceed max_length into smaller pieces."""
        result = []

        for chunk in chunks:
            if len(chunk.text) <= self.max_length * 1.5:
                result.append(chunk)
                continue

            split_chunks = self._split_chunk_by_paragraphs(chunk, document, image_map)
            result.extend(split_chunks)

        return result

    def _split_chunk_by_paragraphs(
        self,
        chunk: Chunk,
        document: Document,
        image_map: dict[str, ImageReference],
    ) -> list[Chunk]:
        """Split a single oversized chunk by paragraph boundaries.

        Also respects max_images limit by splitting when images accumulate.
        """
        text = chunk.text
        paragraphs = re.split(r"\n\n+", text)

        result_chunks = []
        current_text = []
        current_length = 0
        current_image_count = 0
        chunk_id = chunk.chunk_id

        for para in paragraphs:
            para_len = len(para)
            para_images = para.count("[IMAGE:")

            # Check if we should split
            should_split = False

            if current_text:
                # Split on length
                if current_length + para_len > self.max_length:
                    should_split = True
                # Split on image count
                elif current_image_count + para_images > self.max_images:
                    should_split = True

            if should_split:
                chunk_text = "\n\n".join(current_text)
                chunk_text = self._normalize_whitespace(chunk_text)
                chunk_text, images = self._extract_images(chunk_text, image_map)

                new_chunk = Chunk(
                    text=chunk_text,
                    source_path=document.source_path,
                    chunk_id=chunk_id,
                    code_blocks=[],
                    images=images,
                    metadata=document.metadata.copy(),
                    accumulated_code=chunk.accumulated_code.copy(),
                )
                result_chunks.append(new_chunk)
                chunk_id += 1

                current_text = [para]
                current_length = para_len
                current_image_count = para_images
            else:
                current_text.append(para)
                current_length += para_len + 2
                current_image_count += para_images

        if current_text:
            chunk_text = "\n\n".join(current_text)
            chunk_text = self._normalize_whitespace(chunk_text)
            chunk_text, images = self._extract_images(chunk_text, image_map)

            new_chunk = Chunk(
                text=chunk_text,
                source_path=document.source_path,
                chunk_id=chunk_id,
                code_blocks=[],
                images=images,
                metadata=document.metadata.copy(),
                accumulated_code=chunk.accumulated_code.copy(),
            )
            result_chunks.append(new_chunk)

        return result_chunks if result_chunks else [chunk]

    def _merge_small_chunks(
        self,
        chunks: list[Chunk],
        document: Document,
        image_map: dict[str, ImageReference],  # noqa: ARG002
    ) -> list[Chunk]:
        """Merge chunks that are below min_length with their neighbors.

        Two-phase approach:
        1. Normal merging respecting limits (code blocks, images)
        2. Force-merge very small chunks even if slightly exceeding limits
        """
        if len(chunks) <= 1:
            return chunks

        # Phase 1: Normal merging respecting limits
        chunks = self._merge_pass(chunks, document, respect_limits=True)

        # Phase 2: Force-merge very small chunks (< min_length/2)
        # These are too small to be useful and must be merged
        very_small_threshold = self.min_length // 2
        has_very_small = any(len(c.text) < very_small_threshold for c in chunks)

        if has_very_small and len(chunks) > 1:
            chunks = self._merge_pass(chunks, document, respect_limits=False)

        return chunks

    def _merge_pass(
        self,
        chunks: list[Chunk],
        document: Document,
        respect_limits: bool,
    ) -> list[Chunk]:
        """Single pass of chunk merging."""
        changed = True
        threshold = self.min_length if respect_limits else self.min_length // 2

        while changed and len(chunks) > 1:
            changed = False
            new_chunks = []
            i = 0

            while i < len(chunks):
                current = chunks[i]

                # Only merge if below threshold
                if len(current.text) < threshold:
                    # Try forward merge first
                    if i < len(chunks) - 1:
                        next_chunk = chunks[i + 1]
                        if self._can_merge(current, next_chunk, respect_limits):
                            merged = self._merge_two_chunks(current, next_chunk, document)
                            new_chunks.append(merged)
                            i += 2
                            changed = True
                            continue

                    # Try backward merge if forward failed
                    if new_chunks:
                        prev_chunk = new_chunks[-1]
                        if self._can_merge(prev_chunk, current, respect_limits):
                            new_chunks.pop()
                            merged = self._merge_two_chunks(prev_chunk, current, document)
                            new_chunks.append(merged)
                            i += 1
                            changed = True
                            continue

                new_chunks.append(current)
                i += 1

            chunks = new_chunks

        return chunks

    def _can_merge(self, first: Chunk, second: Chunk, respect_limits: bool) -> bool:
        """Check if two chunks can be merged."""
        merged_code_count = len(first.code_blocks) + len(second.code_blocks)
        merged_image_count = len(first.images) + len(second.images)
        merged_length = len(first.text) + len(second.text)

        if respect_limits:
            # Normal mode: respect limits with small tolerance
            return (
                merged_code_count <= self.max_code_blocks + 1
                and merged_image_count <= self.max_images + 1
            )
        else:
            # Force mode: only prevent extreme violations
            return (
                merged_code_count <= self.max_code_blocks * 2
                and merged_image_count <= self.max_images * 2
                and merged_length <= self.max_length * 1.5
            )

    def _merge_two_chunks(
        self,
        first: Chunk,
        second: Chunk,
        document: Document,
    ) -> Chunk:
        """Merge two chunks into one."""
        merged_text = first.text + "\n\n" + second.text
        merged_text = self._normalize_whitespace(merged_text)

        merged_code = first.code_blocks + second.code_blocks

        merged_images = list(first.images)
        for img in second.images:
            if img not in merged_images:
                merged_images.append(img)

        return Chunk(
            text=merged_text,
            source_path=document.source_path,
            chunk_id=first.chunk_id,
            code_blocks=merged_code,
            images=merged_images,
            metadata=document.metadata.copy(),
            accumulated_code=first.accumulated_code.copy(),
        )

    def _create_chunk_from_units(
        self,
        units: list[_Unit],
        document: Document,
        image_map: dict[str, ImageReference],
        chunk_id: int,
        accumulated_code: list[str],
    ) -> Chunk:
        """Create a Chunk from a list of units."""
        text_parts = [u.text for u in units]

        chunk_text = "\n\n".join(text_parts)
        chunk_text = self._normalize_whitespace(chunk_text)

        code_blocks = [u.code_block for u in units if u.code_block]

        chunk_text, images = self._extract_images(chunk_text, image_map)

        chunk_text = self._clean_content(chunk_text)

        return Chunk(
            text=chunk_text,
            source_path=document.source_path,
            chunk_id=chunk_id,
            code_blocks=code_blocks,
            images=images,
            metadata=document.metadata.copy(),
            accumulated_code=accumulated_code.copy(),
        )

    def _enforce_limits(
        self,
        chunks: list[Chunk],
        document: Document,
        image_map: dict[str, ImageReference],
    ) -> list[Chunk]:
        """Final pass to split any chunks that exceed code/image limits.

        This handles edge cases where images are densely packed in the same paragraph.
        """
        result = []

        for chunk in chunks:
            # Check if this chunk exceeds limits
            if (
                len(chunk.images) <= self.max_images
                and len(chunk.code_blocks) <= self.max_code_blocks
            ):
                result.append(chunk)
                continue

            # Need to split this chunk by image markers
            if len(chunk.images) > self.max_images:
                split_chunks = self._split_by_image_markers(chunk, document, image_map)
                result.extend(split_chunks)
            else:
                result.append(chunk)

        return result

    def _split_by_image_markers(
        self,
        chunk: Chunk,
        document: Document,
        image_map: dict[str, ImageReference],
    ) -> list[Chunk]:
        """Split a chunk that has too many images by image marker positions."""
        text = chunk.text

        # Find all image marker positions
        marker_pattern = r"\[IMAGE:[^\]]+\]"
        matches = list(re.finditer(marker_pattern, text))

        if len(matches) <= self.max_images:
            return [chunk]

        result_chunks = []
        chunk_id = chunk.chunk_id
        start_pos = 0

        # Group markers into batches of max_images
        for i in range(0, len(matches), self.max_images):
            batch_matches = matches[i : i + self.max_images]

            if i + self.max_images < len(matches):
                # Find a good split point after the last marker in this batch
                last_marker = batch_matches[-1]
                end_pos = last_marker.end()

                # Try to find a paragraph break nearby
                next_para = text.find("\n\n", end_pos)
                if next_para != -1 and next_para < end_pos + 200:
                    end_pos = next_para
            else:
                end_pos = len(text)

            chunk_text = text[start_pos:end_pos].strip()
            if chunk_text:
                chunk_text = self._normalize_whitespace(chunk_text)
                chunk_text, images = self._extract_images(chunk_text, image_map)

                new_chunk = Chunk(
                    text=chunk_text,
                    source_path=document.source_path,
                    chunk_id=chunk_id,
                    code_blocks=chunk.code_blocks if i == 0 else [],  # Keep codes with first chunk
                    images=images,
                    metadata=document.metadata.copy(),
                    accumulated_code=chunk.accumulated_code.copy(),
                )
                result_chunks.append(new_chunk)
                chunk_id += 1

            start_pos = end_pos

        return result_chunks if result_chunks else [chunk]

    def _classify_chunk_quality(self, chunks: list[Chunk]) -> None:
        """Classify chunk quality based on content analysis."""
        for chunk in chunks:
            quality = self._assess_quality(chunk)
            chunk.quality = quality

    def _assess_quality(self, chunk: Chunk) -> ChunkQuality:
        """Assess the quality of a chunk based on its content."""
        text_lower = chunk.text.lower()

        # Check for import-only chunks
        if chunk.code_blocks and not chunk.images:
            code_combined = "\n".join(chunk.code_blocks).lower()
            lines = [l for l in code_combined.split("\n") if l.strip()]
            if lines:
                import_lines = sum(
                    1
                    for line in lines
                    if line.strip().startswith(("import ", "from ", "%pip", "%set_env", "#"))
                )
                if import_lines / len(lines) > 0.7:
                    return ChunkQuality.LOW

        # Check for table of contents
        if "table of contents" in text_lower:
            return ChunkQuality.LOW

        # Check for setup-only content
        setup_keywords = ["setup", "installation", "prerequisites", "requirements"]
        if any(kw in text_lower[:200] for kw in setup_keywords):
            if not chunk.images and len(chunk.code_blocks) <= 1:
                return ChunkQuality.LOW

        # High quality: has code with images, or substantial explanatory content
        if chunk.images and chunk.code_blocks:
            return ChunkQuality.HIGH

        if len(chunk.text) >= 1000 and (chunk.code_blocks or chunk.images):
            return ChunkQuality.HIGH

        return ChunkQuality.MEDIUM

    def _extract_images(
        self,
        text: str,
        image_map: dict[str, ImageReference],
    ) -> tuple[str, list[ImageReference]]:
        """Extract images from text, removing markers for unresolved images."""
        images = []
        cleaned_text = text

        for image_id, img_ref in image_map.items():
            marker = f"[IMAGE:{image_id}]"
            if marker in cleaned_text:
                if img_ref.resolved_path:
                    images.append(img_ref)
                else:
                    cleaned_text = cleaned_text.replace(marker, "")

        cleaned_text = self._normalize_whitespace(cleaned_text)
        return cleaned_text.strip(), images

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace: collapse multiple blank lines to one."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        text = text.lstrip("\n")
        return text

    def _clean_content(self, content: str) -> str:
        """Clean content by removing data URIs and unwanted HTML."""
        content = re.sub(r"<img[^>]+src=['\"]data:image/[^'\"]+['\"][^>]*>", "", content)
        content = re.sub(r"!\[[^\]]*\]\(data:image/[^)]+\)", "", content)
        content = re.sub(r'<p\s+style="[^"]*">\s*$', "", content, flags=re.MULTILINE)
        content = re.sub(r"</p>\s*$", "", content, flags=re.MULTILINE)

        content = self._normalize_whitespace(content)
        return content.strip()

    def _get_overlap(self, units: list[_Unit]) -> tuple[list[_Unit], int]:
        """Get units for overlap from end of current chunk."""
        if self.overlap <= 0:
            return [], 0

        overlap_units = []
        overlap_chars = 0

        for unit in reversed(units):
            if overlap_chars + unit.length > self.overlap:
                break
            overlap_units.insert(0, unit)
            overlap_chars += unit.length

        return overlap_units, overlap_chars

    def _add_neighbor_context(self, chunks: list[Chunk]) -> None:
        """Add neighbor context to chunks."""
        for idx, chunk in enumerate(chunks):
            if idx > 0:
                prev_text = chunks[idx - 1].text
                chunk.previous_chunk_text = prev_text[-800:] if len(prev_text) > 800 else prev_text

            if idx < len(chunks) - 1:
                next_text = chunks[idx + 1].text
                chunk.next_chunk_text = next_text[:800] if len(next_text) > 800 else next_text
