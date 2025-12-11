"""Base classes for document parsing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ImageType(str, Enum):
    """Type of image content for targeted multimodal generation."""

    CIRCUIT = "circuit"  # Quantum circuit diagrams
    CHART = "chart"  # Histograms, bar charts, line plots
    BLOCH_SPHERE = "bloch_sphere"  # Bloch sphere visualizations
    FORMULA = "formula"  # Mathematical formulas and equations
    DIAGRAM = "diagram"  # General technical diagrams
    TABLE = "table"  # Tables and structured data
    CODE_OUTPUT = "code_output"  # Code execution outputs
    DECORATIVE = "decorative"  # Decorative/non-essential images
    UNKNOWN = "unknown"  # Unclassified images


@dataclass
class ImageReference:
    """Reference to an image in a document."""

    path: str
    alt_text: str = ""
    caption: str = ""
    context: str = ""  # Surrounding text context
    resolved_path: str | None = None
    transcription: str | None = None  # VLM-generated detailed description
    image_type: ImageType = ImageType.UNKNOWN  # Classified image type
    image_id: str = ""  # Unique identifier for chunk references
    code_context: str = ""  # Code that may have generated this image (for outputs)


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
