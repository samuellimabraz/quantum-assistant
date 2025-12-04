"""PDF file parser using PyMuPDF with block-level extraction."""

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import fitz  # PyMuPDF

from synthetic_data.parsers.base import Document, DocumentParser, ImageReference


class PDFType(str, Enum):
    """Type of PDF document for specialized handling."""

    SLIDES = "slides"
    PAPER = "paper"
    WIKI = "wiki"
    UNKNOWN = "unknown"


@dataclass
class _TextBlock:
    """A positioned text block from PDF."""

    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    page_num: int
    font_size: float = 12.0
    is_bold: bool = False

    @property
    def y_center(self) -> float:
        """Vertical center position."""
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def height(self) -> float:
        """Block height."""
        return self.bbox[3] - self.bbox[1]


@dataclass
class _ImageBlock:
    """A positioned image from PDF."""

    bbox: tuple[float, float, float, float]
    page_num: int
    image_index: int

    @property
    def y_center(self) -> float:
        """Vertical center position."""
        return (self.bbox[1] + self.bbox[3]) / 2


class PDFParser(DocumentParser):
    """Parser for PDF files with block-level extraction and proper image positioning."""

    def can_parse(self, path: Path) -> bool:
        """Check if file is a PDF."""
        return path.suffix.lower() == ".pdf"

    def parse(self, path: Path) -> Document:
        """Parse PDF with block-level extraction for proper image positioning."""
        doc = fitz.open(str(path))

        try:
            title = self._extract_title(doc)
            pdf_type = self._detect_pdf_type(doc)

            # Extract all blocks from all pages
            all_text_blocks: list[_TextBlock] = []
            all_image_blocks: list[_ImageBlock] = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_blocks = self._extract_text_blocks(page, page_num)
                image_blocks = self._extract_image_blocks(page, page_num)

                all_text_blocks.extend(text_blocks)
                all_image_blocks.extend(image_blocks)

            # Build content with images positioned correctly
            content, images = self._build_content_with_images(
                all_text_blocks, all_image_blocks, path, pdf_type
            )

            metadata = {
                "page_count": len(doc),
                "pdf_type": pdf_type.value,
                "has_images": len(images) > 0,
            }

            return Document(
                source_path=path,
                title=title,
                content=content,
                code_blocks=[],  # PDFs rarely have code blocks
                images=images,
                metadata=metadata,
            )
        finally:
            doc.close()

    def _detect_pdf_type(self, doc: fitz.Document) -> PDFType:
        """Detect PDF type based on content characteristics."""
        if len(doc) == 0:
            return PDFType.UNKNOWN

        # Sample first few pages
        sample_pages = min(5, len(doc))
        total_text_blocks = 0
        total_images = 0
        avg_text_per_page = 0

        for i in range(sample_pages):
            page = doc.load_page(i)
            blocks = page.get_text("dict")["blocks"]
            text_blocks = [b for b in blocks if b.get("type") == 0]
            image_blocks = [b for b in blocks if b.get("type") == 1]

            total_text_blocks += len(text_blocks)
            total_images += len(image_blocks)

            text = page.get_text()
            avg_text_per_page += len(text)

        avg_text_per_page /= sample_pages
        avg_blocks_per_page = total_text_blocks / sample_pages
        avg_images_per_page = total_images / sample_pages

        # Slides: Less text per page, often many images
        if avg_text_per_page < 800 and avg_images_per_page > 0.3:
            return PDFType.SLIDES

        # Papers: Dense text, structured
        if avg_text_per_page > 2000 and avg_blocks_per_page > 10:
            return PDFType.PAPER

        # Wiki: Moderate text with clear structure
        if avg_blocks_per_page > 5:
            return PDFType.WIKI

        return PDFType.UNKNOWN

    def _extract_text_blocks(self, page: fitz.Page, page_num: int) -> list[_TextBlock]:
        """Extract text blocks with position and formatting info."""
        blocks = []
        dict_data = page.get_text("dict")

        for block in dict_data.get("blocks", []):
            if block.get("type") != 0:  # Text blocks only
                continue

            bbox = block.get("bbox", (0, 0, 0, 0))
            lines = block.get("lines", [])

            # Extract text and font info
            text_parts = []
            max_font_size = 0
            has_bold = False

            for line in lines:
                for span in line.get("spans", []):
                    text_parts.append(span.get("text", ""))
                    font_size = span.get("size", 12)
                    max_font_size = max(max_font_size, font_size)

                    font_name = span.get("font", "").lower()
                    if "bold" in font_name or span.get("flags", 0) & 2:
                        has_bold = True

            text = " ".join(text_parts).strip()
            if text:
                blocks.append(
                    _TextBlock(
                        text=text,
                        bbox=bbox,
                        page_num=page_num,
                        font_size=max_font_size,
                        is_bold=has_bold,
                    )
                )

        return blocks

    def _extract_image_blocks(self, page: fitz.Page, page_num: int) -> list[_ImageBlock]:
        """Extract images with their positions."""
        blocks = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            # Get image bounding box
            try:
                xref = img_info[0]
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rect = img_rects[0]
                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                else:
                    # Fallback: use page center
                    page_rect = page.rect
                    bbox = (
                        page_rect.width * 0.1,
                        page_rect.height * 0.3,
                        page_rect.width * 0.9,
                        page_rect.height * 0.7,
                    )
            except Exception:
                # Fallback position
                page_rect = page.rect
                bbox = (0, page_rect.height * 0.5, page_rect.width, page_rect.height * 0.7)

            blocks.append(
                _ImageBlock(
                    bbox=bbox,
                    page_num=page_num,
                    image_index=img_index,
                )
            )

        return blocks

    def _build_content_with_images(
        self,
        text_blocks: list[_TextBlock],
        image_blocks: list[_ImageBlock],
        source_path: Path,
        pdf_type: PDFType,
    ) -> tuple[str, list[ImageReference]]:
        """Build content with images positioned at correct locations."""
        images: list[ImageReference] = []
        content_parts: list[str] = []

        # Group blocks by page
        pages: dict[int, dict] = {}

        for block in text_blocks:
            if block.page_num not in pages:
                pages[block.page_num] = {"text": [], "images": []}
            pages[block.page_num]["text"].append(block)

        for block in image_blocks:
            if block.page_num not in pages:
                pages[block.page_num] = {"text": [], "images": []}
            pages[block.page_num]["images"].append(block)

        # Process each page
        for page_num in sorted(pages.keys()):
            page_data = pages[page_num]
            page_text_blocks = page_data["text"]
            page_image_blocks = page_data["images"]

            # Sort by vertical position
            page_text_blocks.sort(key=lambda b: b.y_center)
            page_image_blocks.sort(key=lambda b: b.y_center)

            # For slides, add page marker
            if pdf_type == PDFType.SLIDES:
                content_parts.append(f"\n---\n## Slide {page_num + 1}\n")

            # Interleave text and images based on position
            page_content = self._interleave_blocks(
                page_text_blocks,
                page_image_blocks,
                source_path,
                page_num,
                images,
                pdf_type,
            )

            if page_content.strip():
                content_parts.append(page_content)

        content = "\n\n".join(content_parts)
        content = self._clean_content(content)

        return content, images

    def _interleave_blocks(
        self,
        text_blocks: list[_TextBlock],
        image_blocks: list[_ImageBlock],
        source_path: Path,
        page_num: int,
        images: list[ImageReference],
        pdf_type: PDFType,
    ) -> str:
        """Interleave text and image blocks based on vertical position."""
        result_parts: list[str] = []

        text_idx = 0
        img_idx = 0

        while text_idx < len(text_blocks) or img_idx < len(image_blocks):
            # Determine which comes next based on y position
            text_y = text_blocks[text_idx].y_center if text_idx < len(text_blocks) else float("inf")
            img_y = image_blocks[img_idx].y_center if img_idx < len(image_blocks) else float("inf")

            if text_y <= img_y and text_idx < len(text_blocks):
                # Add text block
                block = text_blocks[text_idx]
                formatted_text = self._format_text_block(block, pdf_type)
                if formatted_text:
                    result_parts.append(formatted_text)
                text_idx += 1
            elif img_idx < len(image_blocks):
                # Add image marker
                img_block = image_blocks[img_idx]
                img_ref, marker = self._create_image_reference(
                    img_block, source_path, page_num, text_blocks, text_idx
                )
                images.append(img_ref)
                result_parts.append(marker)
                img_idx += 1

        return "\n".join(result_parts)

    def _format_text_block(self, block: _TextBlock, pdf_type: PDFType) -> str:
        """Format a text block, detecting headers."""
        text = block.text.strip()
        if not text:
            return ""

        # Detect headers based on font size and formatting
        is_header = False
        header_level = "##"

        # Large font or bold text might be a header
        if block.font_size > 14 and len(text) < 100:
            is_header = True
            if block.font_size > 18:
                header_level = "#"

        # For slides, short bold text is likely a title
        if pdf_type == PDFType.SLIDES and block.is_bold and len(text) < 80:
            is_header = True

        if is_header:
            return f"\n{header_level} {text}\n"

        return text

    def _create_image_reference(
        self,
        img_block: _ImageBlock,
        source_path: Path,
        page_num: int,
        text_blocks: list[_TextBlock],
        current_text_idx: int,
    ) -> tuple[ImageReference, str]:
        """Create an ImageReference with context from surrounding text."""
        img_ref_path = f"pdf:{source_path.name}:page{page_num}:img{img_block.image_index}"
        img_id = self._generate_image_id(img_ref_path, source_path)

        # Build context from surrounding text blocks
        context_parts = []

        # Get text before the image (up to 2 blocks)
        start_idx = max(0, current_text_idx - 2)
        for i in range(start_idx, current_text_idx):
            if i < len(text_blocks):
                context_parts.append(text_blocks[i].text)

        # Get text after the image (up to 2 blocks)
        for i in range(current_text_idx, min(current_text_idx + 2, len(text_blocks))):
            context_parts.append(text_blocks[i].text)

        context = " ".join(context_parts)[:500]

        img_ref = ImageReference(
            path=img_ref_path,
            alt_text=f"Figure on page {page_num + 1}",
            context=context or f"Image on page {page_num + 1}",
            image_id=img_id,
        )

        marker = f"\n[IMAGE:{img_id}]\n"
        return img_ref, marker

    def _extract_title(self, doc: fitz.Document) -> str:
        """Extract title from PDF metadata or first page."""
        metadata = doc.metadata
        if metadata and metadata.get("title"):
            return metadata["title"]

        if len(doc) > 0:
            page = doc.load_page(0)
            blocks = self._extract_text_blocks(page, 0)

            # Find the largest font text as title
            if blocks:
                blocks.sort(key=lambda b: b.font_size, reverse=True)
                for block in blocks[:3]:
                    if 10 < len(block.text) < 200:
                        return block.text

        return ""

    def _generate_image_id(self, img_ref_path: str, source_path: Path) -> str:
        """Generate unique image ID."""
        unique_str = f"{source_path.name}:{img_ref_path}"
        hash_val = hashlib.md5(unique_str.encode()).hexdigest()[:12]
        return f"img_{hash_val}"

    def _clean_content(self, content: str) -> str:
        """Clean up extracted content."""
        import re

        # Remove excessive whitespace
        content = re.sub(r"\n{4,}", "\n\n\n", content)
        content = re.sub(r" {2,}", " ", content)

        # Clean up slide markers
        content = re.sub(r"\n---\n\n+", "\n---\n", content)

        return content.strip()
