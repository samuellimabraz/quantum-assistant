"""Export dataset to HuggingFace format."""

from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

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

        # Filter out empty splits to avoid HuggingFace save_to_disk bug
        non_empty_dict = DatasetDict(
            {split: dataset for split, dataset in dataset_dict.items() if len(dataset) > 0}
        )

        # Only save if we have at least one non-empty split
        if non_empty_dict:
            non_empty_dict.save_to_disk(str(output_path))
            print(f"\nDataset saved to {output_path}")

            import json

            manifest = {
                "splits": list(dataset_dict.keys()),
                "split_sizes": {split: len(dataset) for split, dataset in dataset_dict.items()},
            }
            with open(output_path / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        else:
            print("\n[Warning] All splits are empty, skipping save")

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
                "type": Value("string"),
                "image": Image(),
                "source": Value("string"),
            }
        )

    def _create_dataset(self, samples: list[Sample], features: Features) -> Dataset:
        """Create HuggingFace Dataset from samples."""
        data = {
            "question": [],
            "answer": [],
            "category": [],
            "type": [],
            "image": [],
            "source": [],
        }

        for sample in samples:
            # Ensure no None values in string fields
            data["question"].append(sample.question or "")
            data["answer"].append(sample.answer or "")
            data["category"].append(sample.category or "uncategorized")
            data["type"].append(sample.question_type or "qa")

            # Convert source path to relative path
            source_relative = self._make_relative_path(sample.source_path)
            data["source"].append(source_relative or "")

            if sample.image_path:
                try:
                    image_path = self._resolve_image_path(sample.image_path)
                    if image_path and image_path.exists():
                        # Load the actual image and convert to PIL Image
                        image_obj = self._load_and_convert_image(image_path)
                        data["image"].append(image_obj)
                    else:
                        # If image doesn't exist, set to None
                        data["image"].append(None)
                except Exception as e:
                    print(f"Warning: Failed to load image {sample.image_path}: {e}")
                    data["image"].append(None)
            else:
                data["image"].append(None)

        return Dataset.from_dict(data, features=features)

    def _make_relative_path(self, source_path: str | None) -> str:
        """Convert absolute source path to relative path."""
        if not source_path:
            return ""

        # Find the data directory marker in the path
        source_str = str(source_path)

        # Look for common data directory patterns
        patterns = [
            "/data/",
            "/quantum-assistant/data/",
            "\\data\\",  # Windows path
            "\\quantum-assistant\\data\\",  # Windows path
        ]

        for pattern in patterns:
            if pattern in source_str:
                # Extract everything after the pattern
                idx = source_str.find(pattern)
                return source_str[idx + len(pattern) :]

        # If no pattern found, try to extract just the filename
        path = Path(source_path)
        if path.exists():
            # Return the last two directories and filename
            parts = path.parts
            if len(parts) >= 2:
                return str(Path(*parts[-2:]))
            else:
                return path.name

        return source_path  # Return as-is if can't convert

    def _resolve_image_path(self, image_path: str) -> Path | None:
        """Resolve image path to absolute path."""
        if not image_path:
            return None

        path = Path(image_path)

        if path.is_absolute() and path.exists():
            return path

        if self.config and self.config.images_dir:
            images_base = Path(self.config.images_dir)
            possible_path = images_base / path
            if possible_path.exists():
                return possible_path

        outputs_images = Path("outputs/images") / path.name
        if outputs_images.exists():
            return outputs_images

        return None

    def _load_and_convert_image(self, image_path: Path):
        """Load an image and convert it to a PIL Image in a standard format."""

        try:
            with PILImage.open(image_path) as img:
                if img.mode not in ("RGB", "L"):
                    if img.mode == "RGBA":
                        rgb_img = PILImage.new("RGB", img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
                        img = rgb_img
                    else:
                        img = img.convert("RGB")

                max_size = 1024
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)

                return img.copy()

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def _create_dataset_card(self, output_path: Path, dataset_dict: DatasetDict) -> None:
        """Create README.md dataset card."""
        train_count = len(dataset_dict["train"])
        val_count = len(dataset_dict["validation"])
        test_count = len(dataset_dict["test"])
        total_count = train_count + val_count + test_count

        multimodal_count = sum(1 for sample in dataset_dict["train"] if sample["image"] is not None)

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
    "category": str,           # Knowledge category (e.g., quantum_ml_optimization)
    "type": str,               # Type of question (qa, code, caption, summary)
    "image": PIL.Image,        # Image (if multimodal, else None)
    "source": str,             # Relative source document path
}}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{self.config.hub_id or 'path/to/dataset'}")

# Access training data
train_data = dataset["train"]

# Filter by category
quantum_ml = train_data.filter(lambda x: x["category"] == "quantum_ml_optimization")

# Get multimodal samples only
multimodal = train_data.filter(lambda x: x["image"] is not None)

# Filter by question type
code_questions = train_data.filter(lambda x: x["type"] == "code")
```

## Categories

The dataset covers 14 quantum computing topics:
- Quantum Fundamentals
- Quantum States & Entanglement
- Quantum Gates & Circuits
- Quantum Algorithms
- Quantum ML & Optimization
- Quantum Simulation & Hamiltonians
- Quantum Error Correction & Noise
- Quantum Information & Communication
- Variational & Hybrid Approaches
- Quantum Thermodynamics
- Advanced Topological Concepts
- Frameworks & Tooling
- Hardware & Backends
- Educational Content

## Question Types

- **qa**: General question-answer pairs
- **code**: Qiskit code implementation and debugging
- **caption**: Image interpretation and circuit description
- **summary**: Concept summarization and explanation

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
