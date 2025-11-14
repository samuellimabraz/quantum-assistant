"""PDF file parser using PyMuPDF."""

from pathlib import Path

import fitz  # PyMuPDF

from synthetic_data.parsers.base import Document, DocumentParser, ImageReference


class PDFParser(DocumentParser):
    """Parser for PDF files."""

    def can_parse(self, path: Path) -> bool:
        """Check if file is a PDF."""
        return path.suffix.lower() == ".pdf"

    def parse(self, path: Path) -> Document:
        """Parse PDF file and extract content."""
        doc = fitz.open(str(path))

        try:
            title = self._extract_title(doc)
            content_parts = []
            page_count = len(doc)

            for page_num in range(page_count):
                page = doc.load_page(page_num)

                # Extract text
                text = page.get_text()
                if text.strip():
                    content_parts.append(text.strip())

            content = "\n\n".join(content_parts)

            # Create placeholder image references - will be resolved later
            images = self._create_image_references(doc, path)

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

    def _create_image_references(
        self, doc: fitz.Document, source_path: Path
    ) -> list[ImageReference]:
        """Create image references for PDF images."""
        images = []

        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)

                page_text = page.get_text()
                context_lines = page_text.split("\n")[:5]
                context = " ".join(context_lines).strip()[:200]

                for img_index, _ in enumerate(image_list):
                    # Create a reference path that will be used for extraction
                    img_ref_path = f"pdf:{source_path.name}:page{page_num}:img{img_index}"

                    images.append(
                        ImageReference(
                            path=img_ref_path,
                            alt_text=f"Figure {page_num + 1}.{img_index + 1}",
                            context=context or f"Image on page {page_num + 1}",
                        )
                    )
        except Exception as e:
            print(f"Warning: Error creating image references from PDF: {e}")

        return images
