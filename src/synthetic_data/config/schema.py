"""Configuration schema definitions using Pydantic."""

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()


class SourceType(str, Enum):
    """Type of documentation source."""

    DIRECTORY = "directory"
    GITHUB = "github"


class QuestionType(str, Enum):
    """Type of question/prompt to generate."""

    QA = "qa"
    CODE = "code"
    CAPTION = "caption"
    SUMMARY = "summary"


class DifficultyLevel(str, Enum):
    """Difficulty level for generated samples."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SourceConfig(BaseModel):
    """Configuration for a documentation source."""

    path: str  # Can be local path or GitHub URL
    type: SourceType = SourceType.DIRECTORY
    folders: list[str] = Field(default_factory=list)  # For GitHub: specific folders to process
    include_patterns: list[str] = Field(default_factory=lambda: ["*.ipynb", "*.mdx", "*.pdf"])
    exclude_patterns: list[str] = Field(default_factory=lambda: ["**/node_modules/**"])
    max_files: int | None = Field(default=None, ge=1) 


class CategoryConfig(BaseModel):
    """Configuration for a dataset category."""

    name: str
    description: str = ""
    weight: float = Field(default=1.0, ge=0.0)


class ModelEndpoint(BaseModel):
    """Configuration for a model endpoint."""

    name: str
    base_url: str
    api_key: str = Field(default="")
    model_name: str | None = None
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("base_url", "api_key", "model_name", mode="before")
    @classmethod
    def resolve_env_var(cls, v: str) -> str:
        """Resolve environment variables."""
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1]
            return os.getenv(env_var, "")
        return v


class ModelConfig(BaseModel):
    """Configuration for model endpoints."""

    endpoints: list[ModelEndpoint]

    def get_endpoint(self, name: str) -> ModelEndpoint | None:
        """Get endpoint configuration by name."""
        for endpoint in self.endpoints:
            if endpoint.name == name:
                return endpoint
        return None


class PromptsConfig(BaseModel):
    """Prompts for different generation tasks."""

    question_generation: str
    answer_generation: str
    summary_generation: str
    caption_generation: str
    code_generation: str
    content_quality_check: str
    image_quality_check: str
    image_transcription: str
    category_classification: str


class GenerationConfig(BaseModel):
    """Configuration for synthetic data generation."""

    target_samples: int = Field(default=30000, ge=1)
    question_model: str
    vision_model: str | None = None
    answer_model: str

    # Async batching configuration
    llm_batch_size: int = Field(default=10, ge=1)  # Questions/answers batch size
    llm_concurrency: int = Field(default=20, ge=1)  # LLM concurrent requests
    vlm_batch_size: int = Field(default=16, ge=1)  # VLM batch size
    vlm_concurrency: int = Field(default=16, ge=1)  # VLM concurrent requests

    batch_size: int = Field(default=10, ge=1)  # Legacy, uses llm_batch_size
    multimodal_ratio: float = Field(default=0.5, ge=0.0, le=1.0)

    question_types: list[QuestionType] = Field(default_factory=lambda: list(QuestionType))
    question_type_weights: dict[QuestionType, float] = Field(default_factory=dict)

    difficulty_levels: list[DifficultyLevel] = Field(default_factory=lambda: list(DifficultyLevel))
    difficulty_weights: dict[DifficultyLevel, float] = Field(default_factory=dict)

    max_context_length: int = Field(default=2048, ge=1)
    chunk_overlap: int = Field(default=200, ge=0)

    enable_image_transcription: bool = Field(default=True)
    enable_content_filtering: bool = Field(default=False)
    enable_deduplication: bool = Field(default=True)
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class DatasetConfig(BaseModel):
    """Configuration for dataset output."""

    name: str
    description: str = ""

    # Output directories for each pipeline stage
    parsed_dir: Path = Field(default=Path("outputs/parsed"))
    generated_dir: Path = Field(default=Path("outputs/generated"))
    final_dir: Path = Field(default=Path("outputs/final"))

    train_split: float = Field(default=0.8, ge=0.0, le=1.0)
    val_split: float = Field(default=0.1, ge=0.0, le=1.0)
    test_split: float = Field(default=0.1, ge=0.0, le=1.0)

    images_dir: Path = Field(default=Path("outputs/images")) 

    hub_id: str | None = None
    license: str = "apache-2.0"


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    sources: list[SourceConfig]
    categories: list[CategoryConfig]
    models: ModelConfig
    prompts: PromptsConfig
    generation: GenerationConfig
    dataset: DatasetConfig

    seed: int = Field(default=42)

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="python"), f, default_flow_style=False)
