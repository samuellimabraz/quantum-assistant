"""PDF file parser using PyMuPDF."""

import hashlib
from pathlib import Path

import fitz  # PyMuPDF

from synthetic_data.parsers.base import Document, DocumentParser, ImageReference


class PDFParser(DocumentParser):
    """Parser for PDF files."""

    def can_parse(self, path: Path) -> bool:
        """Check if file is a PDF."""
        return path.suffix.lower() == ".pdf"

    def parse(self, path: Path) -> Document:
        """Parse PDF file and extract content.

        Images are replaced with [IMAGE:id] markers in the content.
        """
        doc = fitz.open(str(path))

        try:
            title = self._extract_title(doc)
            content_parts = []
            images = []
            page_count = len(doc)

            for page_num in range(page_count):
                page = doc.load_page(page_num)

                text = page.get_text()
                page_content = [text.strip()] if text.strip() else []

                # Get context for images on this page
                context_lines = text.split("\n")[:5]
                context = " ".join(context_lines).strip()[:200]

                image_list = page.get_images(full=True)
                for img_index, _ in enumerate(image_list):
                    img_ref_path = f"pdf:{path.name}:page{page_num}:img{img_index}"
                    img_id = self._generate_image_id(img_ref_path, path)

                    # Add image reference
                    images.append(
                        ImageReference(
                            path=img_ref_path,
                            alt_text=f"Figure {page_num + 1}.{img_index + 1}",
                            context=context or f"Image on page {page_num + 1}",
                            image_id=img_id,
                        )
                    )

                    # Add marker to content
                    page_content.append(f"\n[IMAGE:{img_id}]\n")

                if page_content:
                    content_parts.append("\n".join(page_content))

            content = "\n\n".join(content_parts)

            metadata = {
                "page_count": page_count,
                "has_images": len(images) > 0,
            }

            return Document(
                source_path=path,
                title=title,
                content=content,
                code_blocks=[],
                images=images,
                metadata=metadata,
            )
        finally:
            doc.close()

    def _extract_title(self, doc: fitz.Document) -> str:
        """Extract title from PDF metadata or first page."""
        # Try metadata first
        metadata = doc.metadata
        if metadata.get("title"):
            return metadata["title"]

        # Try first page
        if len(doc) > 0:
            first_page_text = doc[0].get_text()
            lines = first_page_text.split("\n")
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if len(line) > 10 and len(line) < 200:
                    return line

        return ""

    def _generate_image_id(self, img_ref_path: str, source_path: Path) -> str:
        """Generate unique image ID."""
        unique_str = f"{source_path.name}:{img_ref_path}"
        hash_val = hashlib.md5(unique_str.encode()).hexdigest()[:12]
        return f"img_{hash_val}"
