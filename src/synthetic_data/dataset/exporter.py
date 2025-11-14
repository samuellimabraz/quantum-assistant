"""Export dataset to HuggingFace format."""

from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value

from synthetic_data.config import DatasetConfig
from synthetic_data.generators.pipeline import Sample


class HuggingFaceExporter:
    """Export dataset to HuggingFace datasets format."""

    def __init__(self, config: DatasetConfig):
        """
        Initialize exporter.

        Args:
            config: Dataset configuration
        """
        self.config = config

    def export(
        self,
        train_samples: list[Sample],
        val_samples: list[Sample],
        test_samples: list[Sample],
    ) -> DatasetDict:
        """
        Export samples to HuggingFace DatasetDict.

        Args:
            train_samples: Training samples
            val_samples: Validation samples
            test_samples: Test samples

        Returns:
            DatasetDict with train/validation/test splits
        """
        features = self._create_features()

        train_dataset = self._create_dataset(train_samples, features)
        val_dataset = self._create_dataset(val_samples, features)
        test_dataset = self._create_dataset(test_samples, features)

        dataset_dict = DatasetDict(
            {
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset,
            }
        )

        return dataset_dict

    def save_to_disk(self, dataset_dict: DatasetDict) -> Path:
        """
        Save dataset to disk.

        Args:
            dataset_dict: Dataset to save

        Returns:
            Path to saved dataset
        """
        output_path = Path(self.config.final_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dataset_dict.save_to_disk(str(output_path))
        print(f"\nDataset saved to {output_path}")

        self._create_dataset_card(output_path, dataset_dict)

        return output_path

    def push_to_hub(self, dataset_dict: DatasetDict) -> None:
        """
        Push dataset to HuggingFace Hub.

        Args:
            dataset_dict: Dataset to push
        """
        if not self.config.hub_id:
            raise ValueError("hub_id not configured")

        print(f"\nPushing dataset to HuggingFace Hub: {self.config.hub_id}")
        dataset_dict.push_to_hub(
            self.config.hub_id,
            private=False,
        )
        print("Dataset pushed successfully!")

    def _create_features(self) -> Features:
        """Create HuggingFace features schema."""
        return Features(
            {
                "question": Value("string"),
                "answer": Value("string"),
                "category": Value("string"),
                "question_type": Value("string"),
                "image": Image(),
                "has_image": Value("bool"),
                "code_context": Value("string"),
                "source_path": Value("string"),
            }
        )

    def _create_dataset(self, samples: list[Sample], features: Features) -> Dataset:
        """Create HuggingFace Dataset from samples."""
        data = {
            "question": [],
            "answer": [],
            "category": [],
            "question_type": [],
            "image": [],
            "has_image": [],
            "code_context": [],
            "source_path": [],
        }

        for sample in samples:
            data["question"].append(sample.question)
            data["answer"].append(sample.answer)
            data["category"].append(sample.category)
            data["question_type"].append(sample.question_type)
            data["has_image"].append(sample.image_path is not None)
            data["code_context"].append(sample.code_context or "")
            data["source_path"].append(sample.source_path or "")

            if sample.image_path:
                try:
                    image_path = self._resolve_image_path(sample.image_path)
                    if image_path and image_path.exists():
                        data["image"].append(str(image_path))
                    else:
                        data["image"].append(None)
                except Exception:
                    data["image"].append(None)
            else:
                data["image"].append(None)

        return Dataset.from_dict(data, features=features)

    def _resolve_image_path(self, image_path: str) -> Path | None:
        """Resolve relative image path to absolute path."""
        # This is simplified - adjust based on your image path structure
        path = Path(image_path)

        if path.is_absolute() and path.exists():
            return path

        # Try to find in documentation public folder
        # You may need to adjust this based on your setup
        return None

    def _create_dataset_card(self, output_path: Path, dataset_dict: DatasetDict) -> None:
        """Create README.md dataset card."""
        train_count = len(dataset_dict["train"])
        val_count = len(dataset_dict["validation"])
        test_count = len(dataset_dict["test"])
        total_count = train_count + val_count + test_count

        multimodal_count = sum(1 for sample in dataset_dict["train"] if sample["has_image"])

        card_content = f"""---
license: {self.config.license}
task_categories:
- image-text-to-text
language:
- en
tags:
- quantum-computing
- qiskit
- synthetic
- multimodal
size_categories:
- {self._get_size_category(total_count)}
---

# {self.config.name}

{self.config.description}

## Dataset Details

### Dataset Description

This is a synthetic multimodal dataset for quantum computing assistance, with a focus on Qiskit.
The dataset was generated from official Qiskit documentation and learning materials using a synthetic data generation pipeline.

- **Total Samples:** {total_count:,}
- **Train:** {train_count:,}
- **Validation:** {val_count:,}
- **Test:** {test_count:,}
- **Multimodal Samples:** {multimodal_count:,}

### Dataset Structure

```python
{{
    "question": str,           # The question or prompt
    "answer": str,             # The answer or response
    "category": str,           # Knowledge category
    "question_type": str,      # Type of question (qa, code, caption, etc.)
    "image": PIL.Image,        # Image (if multimodal)
    "has_image": bool,         # Whether sample includes image
    "code_context": str,       # Related code examples
    "source_path": str,        # Source document path
}}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{self.config.hub_id or 'path/to/dataset'}")

# Access training data
train_data = dataset["train"]

# Filter by category
quantum_circuits = train_data.filter(lambda x: x["category"] == "quantum_circuits")

# Get multimodal samples only
multimodal = train_data.filter(lambda x: x["has_image"])
```

## Categories

The dataset covers multiple quantum computing topics including:
- Quantum Circuits
- Quantum Algorithms
- Quantum Information Theory
- Quantum Error Correction
- Qiskit Basics
- Quantum Chemistry
- Quantum Machine Learning

## Question Types

- **QA**: Question-answer pairs
- **Code**: Code generation and implementation tasks
- **Caption**: Image description and interpretation
- **Conceptual**: Conceptual understanding questions
- **Problem Solving**: Problem-solving tasks

## License

This dataset is released under the {self.config.license} license.

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{{self.config.name.replace('-', '_')},
  title = {{{self.config.name}}},
  author = {{Your Name}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/datasets/{self.config.hub_id or 'username/dataset'}}}
}}
```
"""

        readme_path = output_path / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card_content)

        print(f"Dataset card saved to {readme_path}")

    def _get_size_category(self, count: int) -> str:
        """Get HuggingFace size category."""
        if count < 1000:
            return "n<1K"
        elif count < 10000:
            return "1K<n<10K"
        elif count < 100000:
            return "10K<n<100K"
        elif count < 1000000:
            return "100K<n<1M"
        else:
            return "n>1M"
