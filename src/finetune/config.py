"""Configuration schemas for fine-tuning data preparation."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Question types in the dataset."""

    FUNCTION_COMPLETION = "function_completion"
    CODE_GENERATION = "code_generation"
    QA = "qa"


class ImageConfig(BaseModel):
    """Configuration for image processing."""

    max_size: int = Field(
        default=640, ge=64, le=2048, description="Maximum dimension (width or height)"
    )
    quality: int = Field(default=95, ge=1, le=100, description="JPEG quality for compression")
    format: str = Field(default="JPEG", description="Output image format (JPEG, PNG)")
    preserve_aspect_ratio: bool = Field(default=True, description="Maintain original aspect ratio")


class SwiftFormatConfig(BaseModel):
    """Configuration for ms-swift format output."""

    image_placeholder: str = Field(default="<image>", description="Placeholder for images in text")
    include_system_prompt: bool = Field(
        default=True, description="Include system prompt in messages"
    )
    system_prompt: str = Field(
        default=(
            "You are a quantum computing expert assistant specializing in Qiskit. "
            "Provide accurate, clear, and well-structured responses about quantum computing concepts, "
            "algorithms, and code implementation. Use Qiskit 2.0 best practices."
        ),
        description="System prompt for the assistant",
    )


class FinetuneConfig(BaseModel):
    """Complete configuration for fine-tuning data preparation."""

    # Input paths
    dataset_path: Path = Field(
        default=Path("outputs/final"), description="Path to HuggingFace dataset directory"
    )
    images_source_dir: Path = Field(
        default=Path("outputs/images"), description="Source directory for images"
    )

    # Output paths
    output_dir: Path = Field(
        default=Path("outputs/finetune"), description="Output directory for prepared data"
    )
    images_output_dir: Path | None = Field(
        default=None, description="Output directory for processed images"
    )

    # Processing settings
    image: ImageConfig = Field(default_factory=ImageConfig)
    swift: SwiftFormatConfig = Field(default_factory=SwiftFormatConfig)

    # Split configuration
    splits: list[str] = Field(default_factory=lambda: ["train", "validation", "test"])

    # Optional filtering
    max_samples: int | None = Field(
        default=None, ge=1, description="Maximum samples per split (for testing)"
    )
    question_types: list[QuestionType] | None = Field(
        default=None, description="Filter by question types"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.images_output_dir is None:
            self.images_output_dir = self.output_dir / "images"

    @classmethod
    def from_yaml(cls, path: Path) -> "FinetuneConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="python"), f, default_flow_style=False)
