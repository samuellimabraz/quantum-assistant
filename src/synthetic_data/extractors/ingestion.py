"""Document ingestion from sources."""

from pathlib import Path
from typing import TYPE_CHECKING

from synthetic_data.config import SourceConfig
from synthetic_data.parsers import (
    Document,
    DocumentParser,
    JupyterParser,
    MDXParser,
    PDFParser,
    TOCParser,
)
from synthetic_data.utils.image_resolver import ImageResolver

if TYPE_CHECKING:
    from synthetic_data.extractors.transcriber import ImageTranscriber


class DocumentIngestion:
    """Ingest and parse documents from configured sources."""

    def __init__(
        self,
        images_output_dir: Path | None = None,
        image_transcriber: "ImageTranscriber | None" = None,
    ):
        """
        Initialize ingestion with parsers.

        Args:
            images_output_dir: Directory to save extracted/resolved images
            image_transcriber: Optional image transcriber for VLM-based descriptions
        """
        self.image_resolver = ImageResolver(images_output_dir) if images_output_dir else None

        self.parsers: list[DocumentParser] = [
            JupyterParser(image_resolver=self.image_resolver),
            MDXParser(),
            PDFParser(),
        ]
        self.toc_parser = TOCParser()
        self.image_transcriber = image_transcriber

    def ingest_source(self, source: SourceConfig) -> list[Document]:
        """Ingest all documents from a source."""
        source_path = Path(source.path)

        # Handle relative paths
        if not source_path.is_absolute():
            source_path = source_path.resolve()

        if not source_path.exists():
            print(f"Warning: Source path does not exist: {source_path}")
            return []

        if source_path.is_file():
            doc = self._parse_file(source_path)
            return [doc] if doc else []

        documents = []

        # Discover files by pattern
        for pattern in source.include_patterns:
            files = list(source_path.rglob(pattern))
            for file_path in files:
                if self._should_include(file_path, source):
                    doc = self._parse_file(file_path)
                    if doc:
                        documents.append(doc)

        return documents

    def _ingest_from_toc(self, toc_path: Path) -> list[Document]:
        """Ingest documents ordered by TOC."""
        toc = self.toc_parser.parse(toc_path)
        base_dir = toc_path.parent

        documents = []
        ordered_paths = self.toc_parser.get_ordered_paths(toc, base_dir)

        for path in ordered_paths:
            doc = self._parse_file(path)
            if doc:
                # Add TOC information to metadata
                doc.metadata["toc_path"] = str(toc_path.relative_to(path.parent))
                documents.append(doc)

        return documents

    def parse_file(self, path: Path) -> Document | None:
        """Parse a single file (public interface)."""
        for parser in self.parsers:
            if parser.can_parse(path):
                try:
                    doc = parser.parse(path)
                    if doc:
                        if self.image_resolver:
                            self.resolve_document_images(doc)
                        # Note: Transcription is done separately in batch after parsing

                    return doc
                except Exception as e:
                    print(f"Error parsing {path}: {e}")
                    return None
        return None

    def _parse_file(self, path: Path) -> Document | None:
        """Parse a single file (backward compatibility)."""
        return self.parse_file(path)

    def resolve_document_images(self, document: Document):
        """Resolve all image paths in a document (public interface)."""
        for img_ref in document.images:
            resolved = self.image_resolver.resolve_image_path(img_ref.path, document.source_path)
            if resolved:
                img_ref.resolved_path = str(resolved)

    def _resolve_document_images(self, document: Document):
        """Resolve all image paths in a document (backward compatibility)."""
        self.resolve_document_images(document)

    def should_include(self, path: Path, source: SourceConfig) -> bool:
        """Check if file should be included based on patterns (public interface)."""
        path_str = str(path)

        # Check exclude patterns
        for pattern in source.exclude_patterns:
            # Simple pattern matching
            pattern_clean = pattern.replace("**/", "").replace("/**", "")
            if pattern_clean in path_str:
                return False

        return True

    def _should_include(self, path: Path, source: SourceConfig) -> bool:
        """Check if file should be included (backward compatibility)."""
        return self.should_include(path, source)
