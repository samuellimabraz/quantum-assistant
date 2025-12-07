"""Results management utilities for evaluation outputs."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any


class ResultsManager:
    """
    Manages evaluation result file paths and organization.

    Organizes results into structured directories:
        outputs/evaluate/
        ├── qiskit-humaneval/
        │   └── model_n10_k1-5-10_20241125_143022.json
        ├── qiskit-humaneval-hard/
        │   └── ...
        └── synthetic/
            └── model_n1_k1_20241125_150000.json
    """

    BASE_DIR = Path("outputs/evaluate")

    DATASET_DIRS = {
        "qiskit_humaneval": "qiskit-humaneval",
        "qiskit_humaneval_hard": "qiskit-humaneval-hard",
        "synthetic": "synthetic",
    }

    @classmethod
    def get_dataset_dir(cls, dataset_type: str, dataset_variant: str | None = None) -> Path:
        """
        Get the output directory for a dataset type.

        Args:
            dataset_type: Dataset type (qiskit_humaneval, synthetic)
            dataset_variant: Dataset variant (normal, hard) for qiskit_humaneval

        Returns:
            Path to the output directory
        """
        if dataset_type == "qiskit_humaneval":
            if dataset_variant == "hard":
                dir_name = cls.DATASET_DIRS["qiskit_humaneval_hard"]
            else:
                dir_name = cls.DATASET_DIRS["qiskit_humaneval"]
        else:
            dir_name = cls.DATASET_DIRS.get(dataset_type, dataset_type)

        return cls.BASE_DIR / dir_name

    @classmethod
    def sanitize_model_name(cls, model_name: str) -> str:
        """Sanitize model name for use in filename."""
        sanitized = model_name.replace("/", "-").replace("\\", "-")
        sanitized = re.sub(r"[^\w\-.]", "_", sanitized)
        sanitized = re.sub(r"[-_]+", "-", sanitized)
        sanitized = sanitized.strip("-_")
        return sanitized.lower()

    @classmethod
    def generate_filename(
        cls,
        model_name: str,
        num_samples_per_task: int,
        k_values: list[int],
        timestamp: datetime | None = None,
        suffix: str = "",
    ) -> str:
        """
        Generate a descriptive filename for evaluation results.

        Format: {model}_{params}_{timestamp}.json
        Example: qwen2.5-coder-14b_n10_k1-5-10_20241125_143022.json
        """
        if timestamp is None:
            timestamp = datetime.now()

        model = cls.sanitize_model_name(model_name)
        k_str = "-".join(str(k) for k in sorted(k_values))
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")

        parts = [
            model,
            f"n{num_samples_per_task}",
            f"k{k_str}",
            ts_str,
        ]

        if suffix:
            parts.append(suffix)

        return "_".join(parts) + ".json"

    @classmethod
    def get_result_path(
        cls,
        dataset_type: str,
        model_name: str,
        num_samples_per_task: int,
        k_values: list[int],
        dataset_variant: str | None = None,
        timestamp: datetime | None = None,
        suffix: str = "",
    ) -> Path:
        """
        Get the full path for saving evaluation results.

        Args:
            dataset_type: Type of dataset (qiskit_humaneval, synthetic)
            model_name: Name of the model being evaluated
            num_samples_per_task: Number of solutions per task
            k_values: K values for Pass@k
            dataset_variant: Dataset variant (normal, hard) for qiskit
            timestamp: Optional timestamp (defaults to now)
            suffix: Optional suffix for filename

        Returns:
            Full path for the results file
        """
        output_dir = cls.get_dataset_dir(dataset_type, dataset_variant)
        filename = cls.generate_filename(
            model_name=model_name,
            num_samples_per_task=num_samples_per_task,
            k_values=k_values,
            timestamp=timestamp,
            suffix=suffix,
        )
        return output_dir / filename

    @classmethod
    def ensure_output_dir(cls, path: Path) -> None:
        """Ensure the output directory exists."""
        path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def build_metadata(
        cls,
        dataset_path: str | Path,
        dataset_type: str,
        dataset_variant: str | None,
        model_name: str,
        num_samples: int,
        num_samples_per_task: int,
        k_values: list[int],
        timeout: int,
        system_prompt: str | None,
        verify_canonical: bool,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Build comprehensive metadata for evaluation results.

        Args:
            dataset_path: Path to the dataset file
            dataset_type: Type of dataset
            dataset_variant: Dataset variant (normal/hard)
            model_name: Model name
            num_samples: Number of samples evaluated
            num_samples_per_task: Solutions per task
            k_values: K values for Pass@k
            timeout: Execution timeout
            system_prompt: System prompt used (None if no system prompt)
            verify_canonical: Whether canonical verification was enabled
            timestamp: Run timestamp

        Returns:
            Metadata dictionary
        """
        if timestamp is None:
            timestamp = datetime.now()

        return {
            "run_info": {
                "timestamp": timestamp.isoformat(),
                "timestamp_unix": int(timestamp.timestamp()),
            },
            "model": {
                "name": model_name,
            },
            "dataset": {
                "path": str(dataset_path),
                "type": dataset_type,
                "variant": dataset_variant,
                "num_samples": num_samples,
            },
            "evaluation": {
                "solutions_per_task": num_samples_per_task,
                "k_values": k_values,
                "timeout": timeout,
                "system_prompt": system_prompt or None,
                "verify_canonical": verify_canonical,
            },
        }
