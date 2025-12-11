"""Dataset preparation orchestrator for ms-swift fine-tuning."""

import json
from dataclasses import dataclass
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm

from .config import FinetuneConfig
from .formatter import SwiftFormatter, SwiftSample
from .image_processor import ImageProcessor


@dataclass
class PrepareResult:
    """Result of dataset preparation.

    Attributes:
        output_dir: Directory containing prepared files
        splits: Dictionary mapping split names to output paths
        statistics: Preparation statistics
    """

    output_dir: Path
    splits: dict[str, Path]
    statistics: dict[str, dict[str, int]]


class DatasetPreparer:
    """Prepare datasets for ms-swift fine-tuning.

    Supports loading from:
    - HuggingFace Hub (e.g., "samuellimabraz/quantum-test")
    - Local parquet files (downloaded from Hub)
    - Local Arrow format (from save_to_disk)

    Orchestrates the complete preparation pipeline:
    1. Load HuggingFace dataset
    2. Process and resize images
    3. Format samples for ms-swift
    4. Write JSONL files for each split
    """

    def __init__(self, config: FinetuneConfig):
        """Initialize preparer.

        Args:
            config: Finetune configuration
        """
        self.config = config
        self.image_processor = ImageProcessor(config.image, config.images_output_dir)
        self.formatter = SwiftFormatter(config.swift)

    def prepare(self) -> PrepareResult:
        """Run the complete preparation pipeline.

        Returns:
            PrepareResult with output paths and statistics
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        dataset_dict = self._load_dataset()
        results: dict[str, Path] = {}
        statistics: dict[str, dict[str, int]] = {}

        for split_name in self.config.splits:
            if split_name not in dataset_dict:
                print(f"  Skipping split '{split_name}' (not found in dataset)")
                continue

            split_data = dataset_dict[split_name]
            output_path = self.config.output_dir / f"{split_name}.jsonl"

            samples, stats = self._process_split(split_data, split_name)
            self.formatter.write_jsonl(samples, output_path)

            results[split_name] = output_path
            statistics[split_name] = stats

        self._write_summary(statistics)

        return PrepareResult(
            output_dir=self.config.output_dir,
            splits=results,
            statistics=statistics,
        )

    def _load_dataset(self) -> DatasetDict:
        """Load HuggingFace dataset from Hub, parquet, or local Arrow format.

        Supports three loading methods:
        1. HuggingFace Hub: if hub_id is provided
        2. Parquet files: if dataset_path contains .parquet files or data/ subfolder
        3. Arrow format: if dataset_path is from save_to_disk (has dataset_dict.json)

        Returns:
            DatasetDict with train/validation/test splits
        """
        # Priority 1: HuggingFace Hub
        if self.config.hub_id:
            print(f"Loading dataset from HuggingFace Hub: {self.config.hub_id}")
            return load_dataset(self.config.hub_id)

        # Priority 2: Local path
        if not self.config.dataset_path:
            raise ValueError("Either hub_id or dataset_path must be provided")

        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        # Check if it's Arrow format (from save_to_disk)
        if (dataset_path / "dataset_dict.json").exists():
            print(f"Loading Arrow dataset from: {dataset_path}")
            return load_from_disk(str(dataset_path))

        # Check for parquet files in data/ subfolder (HuggingFace Hub download structure)
        data_dir = dataset_path / "data"
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
            if parquet_files:
                print(f"Loading parquet dataset from: {data_dir}")
                return self._load_parquet_splits(data_dir)

        # Check for parquet files directly in the path
        parquet_files = list(dataset_path.glob("*.parquet"))
        if parquet_files:
            print(f"Loading parquet dataset from: {dataset_path}")
            return self._load_parquet_splits(dataset_path)

        raise ValueError(
            f"Could not determine dataset format at {dataset_path}. "
            "Expected Arrow format (dataset_dict.json) or parquet files."
        )

    def _load_parquet_splits(self, parquet_dir: Path) -> DatasetDict:
        """Load dataset splits from parquet files.

        Parquet files are expected to be named like:
        - train-00000-of-00001.parquet
        - validation-00000-of-00001.parquet
        - test-00000-of-00001.parquet

        Args:
            parquet_dir: Directory containing parquet files

        Returns:
            DatasetDict with available splits
        """
        splits = {}
        split_names = ["train", "validation", "test"]

        for split_name in split_names:
            # Find parquet files for this split
            split_files = list(parquet_dir.glob(f"{split_name}-*.parquet"))
            if split_files:
                # Load all parquet files for this split
                file_paths = [str(f) for f in sorted(split_files)]
                splits[split_name] = load_dataset("parquet", data_files=file_paths, split="train")

        if not splits:
            raise ValueError(f"No parquet files found in {parquet_dir}")

        return DatasetDict(splits)

    def _process_split(
        self,
        split_data,
        split_name: str,
    ) -> tuple[list[SwiftSample], dict[str, int]]:
        """Process a single dataset split.

        Args:
            split_data: HuggingFace dataset split
            split_name: Name of the split

        Returns:
            Tuple of (processed samples, statistics dict)
        """
        samples = []
        stats = {
            "total": 0,
            "multimodal": 0,
            "text_only": 0,
            "function_completion": 0,
            "code_generation": 0,
            "qa": 0,
            "skipped": 0,
        }

        max_samples = self.config.max_samples
        items = list(split_data)
        if max_samples:
            items = items[:max_samples]

        print(f"\nProcessing {split_name} split ({len(items)} samples)")

        for item in tqdm(items, desc=f"  {split_name}"):
            sample = self._process_sample(item, stats)
            if sample:
                samples.append(sample)

        return samples, stats

    def _process_sample(self, item: dict, stats: dict[str, int]) -> SwiftSample | None:
        """Process a single sample.

        Args:
            item: Raw sample from dataset
            stats: Statistics dictionary to update

        Returns:
            SwiftSample or None if skipped
        """
        stats["total"] += 1

        question_type = item.get("type", "qa")
        if self.config.question_types:
            allowed = [qt.value for qt in self.config.question_types]
            if question_type not in allowed:
                stats["skipped"] += 1
                return None

        if question_type in stats:
            stats[question_type] += 1

        question = item.get("question", "")
        answer = item.get("answer", "")

        if not question or not answer:
            stats["skipped"] += 1
            return None

        image_path = None
        raw_image = item.get("image")
        if raw_image is not None:
            pil_image = self._extract_pil_image(raw_image)
            if pil_image:
                image_path = self.image_processor.process(pil_image)
                stats["multimodal"] += 1
            else:
                stats["text_only"] += 1
        else:
            stats["text_only"] += 1

        return self.formatter.format_sample(
            question=question,
            answer=answer,
            question_type=question_type,
            image_path=image_path,
        )

    def _extract_pil_image(self, raw_image) -> Image.Image | None:
        """Extract PIL Image from various formats.

        Args:
            raw_image: Image data in various formats

        Returns:
            PIL Image or None
        """
        if raw_image is None:
            return None

        if isinstance(raw_image, Image.Image):
            return raw_image

        if isinstance(raw_image, dict):
            if "bytes" in raw_image:
                import io

                return Image.open(io.BytesIO(raw_image["bytes"]))
            if "path" in raw_image and raw_image["path"]:
                path = Path(raw_image["path"])
                if path.exists():
                    return Image.open(path)

        return None

    def _write_summary(self, statistics: dict[str, dict[str, int]]) -> None:
        """Write preparation summary to JSON file."""
        summary = {
            "config": {
                "image_max_size": self.config.image.max_size,
                "image_format": self.config.image.format,
                "include_system_prompt": self.config.swift.include_system_prompt,
            },
            "splits": statistics,
            "images_processed": self.image_processor.processed_count,
        }

        summary_path = self.config.output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to {summary_path}")
        print(f"Total images processed: {self.image_processor.processed_count}")
