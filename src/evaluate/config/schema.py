"""Configuration schema for evaluation using Pydantic."""

import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field, field_validator

# Load .env file - find_dotenv searches up the directory tree
dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    load_dotenv()


class ModelConfig(BaseModel):
    """Configuration for model endpoint."""

    base_url: str
    api_key: str = Field(default="")
    model_name: str = Field(default="qwen")
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout: float = Field(default=300.0, ge=1.0)
    is_vlm: bool = Field(default=False)

    @field_validator("base_url", "api_key", "model_name", mode="before")
    @classmethod
    def resolve_env_var(cls, v: str) -> str:
        """Resolve environment variables."""
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1]
            return os.getenv(env_var, "")
        return v


class DatasetConfig(BaseModel):
    """Configuration for dataset to evaluate."""

    type: str = Field(..., description="Dataset type: 'qiskit_humaneval' or 'synthetic'")
    path: Path = Field(..., description="Path to dataset file or HuggingFace dataset directory")
    images_dir: Path | None = Field(
        default=None, description="Directory containing images (for synthetic multimodal)"
    )
    max_samples: int | None = Field(default=None, ge=1, description="Limit number of samples")
    # Qiskit HumanEval specific
    dataset_variant: str | None = Field(
        default=None,
        description="Dataset variant: 'normal' (completion) or 'hard' (full generation). Auto-detected if None",
    )
    # Synthetic dataset specific
    split: str = Field(
        default="test",
        description="Dataset split to evaluate (train, validation, test). For synthetic datasets.",
    )


class MetricsConfig(BaseModel):
    """Configuration for evaluation metrics."""

    # Code evaluation
    num_samples_per_task: int = Field(
        default=1, ge=1, description="Solutions per task (for pass@k)"
    )
    k_values: list[int] = Field(default=[1], description="K values for Pass@k")
    execution_timeout: int = Field(default=30, ge=5, le=300, description="Code execution timeout")

    # Generation concurrency
    max_concurrent: int = Field(
        default=10, ge=1, le=100, description="Maximum concurrent API requests"
    )

    # Text evaluation (for synthetic QA)
    num_predictions: int = Field(default=1, ge=1, description="Predictions per sample")

    # System prompt configuration
    system_prompt_type: str | None = Field(
        default="qiskit_humaneval",
        description="System prompt type: 'qiskit_humaneval', 'qiskit_humaneval_minimal', 'generic', 'custom', or None",
    )
    custom_system_prompt: str | None = Field(
        default=None, description="Custom system prompt (used when system_prompt_type is 'custom')"
    )

    # Canonical solution verification (for debugging)
    verify_canonical: bool = Field(
        default=False, description="Also verify that canonical solutions pass their tests"
    )


class OutputConfig(BaseModel):
    """Configuration for output."""

    results_file: Path | None = Field(
        default=None,
        description="Explicit path to save results JSON. If None, auto-generated in results_dir",
    )
    results_dir: Path = Field(
        default=Path("outputs/evaluate"),
        description="Base directory for results (used when results_file is None)",
    )
    auto_filename: bool = Field(
        default=True,
        description="Auto-generate descriptive filename based on model, k-values, etc.",
    )


class EvaluationConfig(BaseModel):
    """Complete evaluation configuration."""

    model: ModelConfig
    dataset: DatasetConfig
    metrics: MetricsConfig
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "EvaluationConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        data = self.model_dump(mode="python")

        def convert_paths(obj):
            """Recursively convert Path objects to strings."""
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        data = convert_paths(data)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
