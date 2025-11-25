"""Export dataset to HuggingFace format."""

from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

from synthetic_data.config import DatasetConfig
from synthetic_data.models.types import Sample


class HuggingFaceExporter:
    """Export dataset to HuggingFace datasets format.
    
    Exports samples with the following structure:
    - question: The input prompt/question
    - answer: The reference solution/answer
    - category: One of 14 quantum computing categories
    - type: function_completion, code_generation, or qa
    - test_code: Unit test for code types (null for qa)
    - entry_point: Function name for code types (null for qa)
    - image: Associated image for multimodal samples
    - source: Relative path to source document
    """

    def __init__(self, config: DatasetConfig):
        """Initialize exporter.
        
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
        """Export samples to HuggingFace DatasetDict.
        
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

        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        })

    def save_to_disk(self, dataset_dict: DatasetDict) -> Path:
        """Save dataset to disk.
        
        Args:
            dataset_dict: Dataset to save
            
        Returns:
            Path to saved dataset
        """
        output_path = Path(self.config.final_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Filter out empty splits
        non_empty_dict = DatasetDict({
            split: dataset
            for split, dataset in dataset_dict.items()
            if len(dataset) > 0
        })

        if non_empty_dict:
            non_empty_dict.save_to_disk(str(output_path))
            print(f"\nDataset saved to {output_path}")

            import json
            manifest = {
                "splits": list(dataset_dict.keys()),
                "split_sizes": {
                    split: len(dataset)
                    for split, dataset in dataset_dict.items()
                },
            }
            with open(output_path / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        else:
            print("\n[Warning] All splits are empty, skipping save")

        self._create_dataset_card(output_path, dataset_dict)
        return output_path

    def push_to_hub(self, dataset_dict: DatasetDict) -> None:
        """Push dataset to HuggingFace Hub.
        
        Args:
            dataset_dict: Dataset to push
        """
        if not self.config.hub_id:
            raise ValueError("hub_id not configured")

        print(f"\nPushing dataset to HuggingFace Hub: {self.config.hub_id}")
        dataset_dict.push_to_hub(self.config.hub_id, private=False)
        print("Dataset pushed successfully!")

    def _create_features(self) -> Features:
        """Create HuggingFace features schema."""
        return Features({
            "question": Value("string"),
            "answer": Value("string"),
            "category": Value("string"),
            "type": Value("string"),
            "test_code": Value("string"),
            "entry_point": Value("string"),
            "image": Image(),
            "source": Value("string"),
        })

    def _create_dataset(self, samples: list[Sample], features: Features) -> Dataset:
        """Create HuggingFace Dataset from samples."""
        data = {
            "question": [],
            "answer": [],
            "category": [],
            "type": [],
            "test_code": [],
            "entry_point": [],
            "image": [],
            "source": [],
        }

        for sample in samples:
            data["question"].append(sample.question or "")
            data["answer"].append(sample.answer or "")
            data["category"].append(sample.category or "uncategorized")
            data["type"].append(sample.question_type or "qa")
            data["test_code"].append(sample.test_code or "")
            data["entry_point"].append(sample.entry_point or "")

            source_relative = self._make_relative_path(sample.source_path)
            data["source"].append(source_relative or "")

            if sample.image_path:
                try:
                    image_path = self._resolve_image_path(sample.image_path)
                    if image_path and image_path.exists():
                        image_obj = self._load_and_convert_image(image_path)
                        data["image"].append(image_obj)
                    else:
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

        source_str = str(source_path)
        patterns = ["/data/", "/quantum-assistant/data/", "\\data\\", "\\quantum-assistant\\data\\"]

        for pattern in patterns:
            if pattern in source_str:
                idx = source_str.find(pattern)
                return source_str[idx + len(pattern):]

        path = Path(source_path)
        if path.exists():
            parts = path.parts
            if len(parts) >= 2:
                return str(Path(*parts[-2:]))
            return path.name

        return source_path

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
        """Load an image and convert to PIL Image in standard format."""
        import io

        try:
            if image_path.suffix.lower() == ".svg":
                try:
                    from wand.image import Image as WandImage
                    from wand.color import Color

                    with WandImage(filename=str(image_path), resolution=300) as wand_img:
                        wand_img.background_color = Color("white")
                        wand_img.alpha_channel = "remove"
                        wand_img.format = "png"
                        png_blob = wand_img.make_blob("png")
                        img = PILImage.open(io.BytesIO(png_blob))
                except ImportError:
                    print(f"Warning: Wand not available for SVG {image_path}")
                    return None
            else:
                if str(image_path).lower().endswith(".avif"):
                    try:
                        import pillow_avif  # noqa: F401
                    except ImportError:
                        print(f"Warning: pillow-avif not available for {image_path}")
                        return None
                img = PILImage.open(image_path)

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

        multimodal_count = sum(
            1 for sample in dataset_dict["train"] if sample["image"] is not None
        )
        code_with_tests = sum(
            1 for sample in dataset_dict["train"] if sample["test_code"]
        )

        card_content = f"""---
license: {self.config.license}
task_categories:
- image-text-to-text
- text-generation
language:
- en
tags:
- quantum-computing
- qiskit
- synthetic
- multimodal
- code-generation
size_categories:
- {self._get_size_category(total_count)}
---

# {self.config.name}

{self.config.description}

## Dataset Details

### Dataset Description

High-quality multimodal dataset for quantum computing VLM fine-tuning with Qiskit.
Features three input types with unit tests for code verification.

- **Total Samples:** {total_count:,}
- **Train:** {train_count:,}
- **Validation:** {val_count:,}
- **Test:** {test_count:,}
- **Multimodal Samples:** {multimodal_count:,}
- **Samples with Unit Tests:** {code_with_tests:,}

### Dataset Structure

```python
{{
    "question": str,       # Input prompt/question
    "answer": str,         # Reference solution/answer
    "category": str,       # Knowledge category (14 categories)
    "type": str,           # function_completion, code_generation, or qa
    "test_code": str,      # Unit test for code types (empty for qa)
    "entry_point": str,    # Function name for code types
    "image": PIL.Image,    # Image (if multimodal, else None)
    "source": str,         # Relative source document path
}}
```

## Question Types

### function_completion
Prompt includes imports, function signature, and docstring.
Model completes the function body. Includes unit test for validation.

### code_generation  
Natural language task description (Qiskit HumanEval Hard format).
Model generates complete code with imports. Includes unit test.

### qa
Theory and concepts (explanation, summary, analysis).
No unit test required. Code in answers is syntax/execution verified.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{self.config.hub_id or 'path/to/dataset'}")

# Get samples with unit tests
with_tests = dataset["train"].filter(lambda x: x["test_code"])

# Get multimodal samples
multimodal = dataset["train"].filter(lambda x: x["image"] is not None)

# Filter by type
func_completion = dataset["train"].filter(lambda x: x["type"] == "function_completion")
```

## Categories

14 quantum computing categories:
- quantum_fundamentals
- quantum_states_entanglement
- quantum_gates_circuits
- quantum_algorithms
- quantum_ml_optimization
- quantum_simulation_hamiltonians
- quantum_error_correction_noise
- quantum_information_communication
- variational_hybrid_approaches
- quantum_thermodynamics
- advanced_topological
- frameworks_tooling
- hardware_backends
- education_meta

## License

Released under {self.config.license} license.

## Citation

```bibtex
@misc{{{self.config.name.replace('-', '_')},
  title = {{{self.config.name}}},
  author = {{Samuel Lima Braz}},
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
        return "n>1M"
