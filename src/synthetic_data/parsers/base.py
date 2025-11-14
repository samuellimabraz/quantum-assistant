"""Base classes for document parsing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImageReference:
    """Reference to an image in a document."""

    path: str
    alt_text: str = ""
    caption: str = ""
    context: str = ""
    resolved_path: str | None = None
    transcription: str | None = None  # VLM-generated detailed description


@dataclass
class Document:
    """Parsed document with extracted content."""

    source_path: Path
    title: str = ""
    content: str = ""
    code_blocks: list[str] = field(default_factory=list)
    images: list[ImageReference] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def has_images(self) -> bool:
        """Check if document contains images."""
        return len(self.images) > 0

    @property
    def has_code(self) -> bool:
        """Check if document contains code blocks."""
        return len(self.code_blocks) > 0


class DocumentParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def can_parse(self, path: Path) -> bool:
        """Check if this parser can handle the given file."""
        pass

    @abstractmethod
    def parse(self, path: Path) -> Document:
        """Parse a document and extract content."""
        pass
