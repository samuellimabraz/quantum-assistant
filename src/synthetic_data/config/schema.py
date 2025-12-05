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
    """Type of question/prompt to generate.

    - FUNCTION_COMPLETION: Prompt with imports + function signature + docstring (model completes)
    - CODE_GENERATION: Direct task description (model generates full code)
    - QA: Theory/concepts (explanation, summary, analysis - no unit test required)
    """

    FUNCTION_COMPLETION = "function_completion"
    CODE_GENERATION = "code_generation"
    QA = "qa"


class SourceConfig(BaseModel):
    """Configuration for a documentation source."""

    path: str
    type: SourceType = SourceType.DIRECTORY
    folders: list[str] = Field(default_factory=list)
    include_patterns: list[str] = Field(default_factory=lambda: ["*.ipynb", "*.mdx", "*.pdf"])
    exclude_patterns: list[str] = Field(default_factory=lambda: ["**/node_modules/**"])
    max_files: int | None = Field(default=None, ge=1)


class CategoryConfig(BaseModel):
    """Configuration for a dataset category."""

    name: str
    description: str = ""


class ModelEndpoint(BaseModel):
    """Configuration for a model endpoint."""

    name: str
    base_url: str
    api_key: str = Field(default="")
    model_name: str | None = None
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    service_tier: str | None = Field(default=None)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    repetition_penalty: float | None = Field(default=None, ge=0.0)

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
    """Prompts for the generation pipeline.

    Session-Based Generation:
    1. Input Generation Session:
       - input_generation_system: System prompt for question + test generation
       - function_completion_prompt, code_generation_prompt, qa_prompt: Question prompts
       - test_generation_prompt: Test generation

    2. Answer Generation Session:
       - answer_generation_system: System prompt for answer + validation loop
       - function_completion_answer_prompt, code_generation_answer_prompt, qa_answer_prompt
       - answer_correction_prompt: Correction request when validation fails

    3. Candidate Filtering:
       - candidate_filter_system, candidate_filter_prompt: Filter generated inputs

    4. Post-Generation Classification:
       - category_classification, category_classification_system
    """

    # === INPUT GENERATION SESSION ===
    input_generation_system: str = ""
    function_completion_prompt: str = ""
    code_generation_prompt: str = ""
    qa_prompt: str = ""
    test_generation_prompt: str = ""

    # === ANSWER GENERATION SESSION ===
    answer_generation_system: str = ""
    function_completion_answer_prompt: str = ""
    code_generation_answer_prompt: str = ""
    qa_answer_prompt: str = ""
    answer_correction_prompt: str = ""

    # === QUALITY CONTROL ===
    content_filter_system: str = ""
    content_quality_check: str = ""
    image_filter_system: str = ""
    image_quality_check: str = ""
    image_transcription_system: str = ""
    image_transcription: str = ""

    # === CANDIDATE FILTERING ===
    candidate_filter_system: str = ""
    candidate_filter_prompt: str = ""

    # === CLASSIFICATION AND CURATION ===
    category_classification_system: str = ""
    category_classification: str = ""
    sample_curation_system: str = ""
    sample_curation: str = ""


class TypeAllocationConfig(BaseModel):
    """Configuration for a single question type's allocation."""

    ratio: float = Field(default=0.33, ge=0.0, le=1.0)  # Portion of total samples
    multimodal_ratio: float = Field(default=0.5, ge=0.0, le=1.0)  # Portion multimodal


class GenerationConfig(BaseModel):
    """Configuration for synthetic data generation."""

    target_samples: int = Field(default=8000, ge=1)
    question_model: str
    vision_model: str | None = None
    answer_model: str
    curate_model: str
    filter_model: str | None = None

    llm_batch_size: int = Field(default=10, ge=1)
    llm_concurrency: int = Field(default=20, ge=1)
    vlm_batch_size: int = Field(default=16, ge=1)
    vlm_concurrency: int = Field(default=16, ge=1)

    # Per-type allocation configuration
    type_allocations: dict[str, TypeAllocationConfig] = Field(
        default_factory=lambda: {
            "qa": TypeAllocationConfig(ratio=0.30, multimodal_ratio=0.70),
            "code_generation": TypeAllocationConfig(ratio=0.35, multimodal_ratio=0.30),
            "function_completion": TypeAllocationConfig(ratio=0.35, multimodal_ratio=0.30),
        }
    )

    # Legacy fields (kept for backwards compatibility)
    multimodal_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    question_types: list[QuestionType] = Field(default_factory=lambda: list(QuestionType))
    question_type_weights: dict[QuestionType, float] = Field(default_factory=dict)

    max_context_length: int = Field(default=2048, ge=1)
    min_chunk_length: int = Field(default=100, ge=0)
    chunk_overlap: int = Field(default=0, ge=0)
    max_code_blocks_per_chunk: int = Field(default=4, ge=1, le=10)
    max_images_per_chunk: int = Field(default=4, ge=1, le=10)

    # Input planning
    candidates_per_chunk: int = Field(default=2, ge=1, le=10)
    enable_candidate_filtering: bool = Field(default=True)

    enable_image_transcription: bool = Field(default=True)
    enable_content_filtering: bool = Field(default=False)
    enable_curate_filtering: bool = Field(default=True)
    enable_deduplication: bool = Field(default=True)
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    enable_code_verification: bool = Field(default=True)
    code_verification_max_iterations: int = Field(default=3, ge=1, le=10)
    code_verification_timeout: int = Field(default=30, ge=5, le=120)
    code_verification_concurrency: int = Field(default=5, ge=1)

    test_validation_timeout: int = Field(default=60, ge=10, le=300)


class DatasetConfig(BaseModel):
    """Configuration for dataset output."""

    name: str
    description: str = ""

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

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="python"), f, default_flow_style=False)
