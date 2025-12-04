"""Dataset analysis and statistics computation."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from synthetic_data.models import Sample


@dataclass
class SplitStatistics:
    """Statistics for a single dataset split."""

    name: str
    total: int
    by_type: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    multimodal: int = 0
    text_only: int = 0
    with_tests: int = 0

    # Detailed multimodal breakdown
    multimodal_by_type: dict[str, int] = field(default_factory=dict)
    multimodal_by_category: dict[str, int] = field(default_factory=dict)

    @property
    def multimodal_ratio(self) -> float:
        """Calculate multimodal ratio."""
        return self.multimodal / self.total if self.total > 0 else 0.0

    @property
    def test_coverage(self) -> float:
        """Calculate test coverage ratio."""
        return self.with_tests / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total": self.total,
            "by_type": self.by_type,
            "by_category": self.by_category,
            "multimodal": self.multimodal,
            "text_only": self.text_only,
            "with_tests": self.with_tests,
            "multimodal_ratio": round(self.multimodal_ratio, 4),
            "test_coverage": round(self.test_coverage, 4),
            "multimodal_by_type": self.multimodal_by_type,
            "multimodal_by_category": self.multimodal_by_category,
        }


@dataclass
class DatasetStatistics:
    """Complete dataset statistics across all splits."""

    splits: dict[str, SplitStatistics] = field(default_factory=dict)

    @property
    def total_samples(self) -> int:
        """Total samples across all splits."""
        return sum(s.total for s in self.splits.values())

    @property
    def all_types(self) -> list[str]:
        """Get all unique question types."""
        types = set()
        for split in self.splits.values():
            types.update(split.by_type.keys())
        return sorted(types)

    @property
    def all_categories(self) -> list[str]:
        """Get all unique categories sorted by total count."""
        category_counts = defaultdict(int)
        for split in self.splits.values():
            for cat, count in split.by_category.items():
                category_counts[cat] += count
        return [cat for cat, _ in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)]

    def get_type_distribution(self) -> dict[str, dict[str, int]]:
        """Get type distribution per split."""
        return {name: split.by_type for name, split in self.splits.items()}

    def get_category_distribution(self) -> dict[str, dict[str, int]]:
        """Get category distribution per split."""
        return {name: split.by_category for name, split in self.splits.items()}

    def get_modality_distribution(self) -> dict[str, dict[str, int]]:
        """Get modality distribution per split."""
        return {
            name: {"multimodal": split.multimodal, "text_only": split.text_only}
            for name, split in self.splits.items()
        }

    def get_multimodal_by_type(self) -> dict[str, dict[str, int]]:
        """Get multimodal counts by type per split."""
        return {name: split.multimodal_by_type for name, split in self.splits.items()}

    def get_multimodal_by_category(self) -> dict[str, dict[str, int]]:
        """Get multimodal counts by category per split."""
        return {name: split.multimodal_by_category for name, split in self.splits.items()}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_samples": self.total_samples,
            "splits": {name: split.to_dict() for name, split in self.splits.items()},
            "aggregated": {
                "by_type": self._aggregate_dict("by_type"),
                "by_category": self._aggregate_dict("by_category"),
                "multimodal": sum(s.multimodal for s in self.splits.values()),
                "text_only": sum(s.text_only for s in self.splits.values()),
                "with_tests": sum(s.with_tests for s in self.splits.values()),
                "multimodal_by_type": self._aggregate_dict("multimodal_by_type"),
                "multimodal_by_category": self._aggregate_dict("multimodal_by_category"),
            },
        }

    def _aggregate_dict(self, attr: str) -> dict[str, int]:
        """Aggregate a dict attribute across all splits."""
        aggregated = defaultdict(int)
        for split in self.splits.values():
            for key, value in getattr(split, attr).items():
                aggregated[key] += value
        return dict(aggregated)


class DatasetAnalyzer:
    """Analyze dataset distributions and compute statistics."""

    def __init__(self, dataset_path: Optional[Path] = None):
        """Initialize analyzer.

        Args:
            dataset_path: Path to dataset directory (optional, can load later)
        """
        self.dataset_path = dataset_path
        self._samples: dict[str, list[Sample]] = {}
        self._statistics: Optional[DatasetStatistics] = None

    def load_from_splits(
        self,
        train_samples: list[Sample],
        val_samples: list[Sample],
        test_samples: list[Sample],
    ) -> None:
        """Load samples directly from split lists.

        Args:
            train_samples: Training samples
            val_samples: Validation samples
            test_samples: Test samples
        """
        self._samples = {
            "train": train_samples,
            "validation": val_samples,
            "test": test_samples,
        }
        self._statistics = None

    def load_from_pickle(self, splits_dir: Path) -> None:
        """Load samples from pickle files.

        Args:
            splits_dir: Directory containing train.pkl, val.pkl, test.pkl
        """
        import pickle

        splits_dir = Path(splits_dir)
        self._samples = {}

        split_files = {
            "train": "train.pkl",
            "validation": "val.pkl",
            "test": "test.pkl",
        }

        for split_name, filename in split_files.items():
            pkl_file = splits_dir / filename
            if pkl_file.exists():
                with open(pkl_file, "rb") as f:
                    self._samples[split_name] = pickle.load(f)
            else:
                self._samples[split_name] = []

        self._statistics = None

    def load_from_huggingface(self, dataset_path: Path) -> None:
        """Load samples from HuggingFace dataset format.

        Args:
            dataset_path: Path to HuggingFace dataset directory
        """
        from datasets import load_from_disk

        dataset_path = Path(dataset_path)
        dataset_dict = load_from_disk(str(dataset_path))

        self._samples = {}
        for split_name, dataset in dataset_dict.items():
            samples = []
            for row in dataset:
                sample = Sample(
                    question=row.get("question", ""),
                    answer=row.get("answer", ""),
                    category=row.get("category", ""),
                    question_type=row.get("type", ""),
                    test_code=row.get("test_code"),
                    entry_point=row.get("entry_point"),
                    image_path=row.get("source") if row.get("image") else None,
                    source_path=row.get("source"),
                )
                samples.append(sample)
            self._samples[split_name] = samples

        self._statistics = None

    def analyze(self) -> DatasetStatistics:
        """Compute statistics for all loaded samples.

        Returns:
            DatasetStatistics with distributions across all splits
        """
        if not self._samples:
            raise ValueError("No samples loaded. Call load_from_* first.")

        stats = DatasetStatistics()

        for split_name, samples in self._samples.items():
            split_stats = self._compute_split_statistics(split_name, samples)
            stats.splits[split_name] = split_stats

        self._statistics = stats
        return stats

    def _compute_split_statistics(self, split_name: str, samples: list[Sample]) -> SplitStatistics:
        """Compute statistics for a single split.

        Args:
            split_name: Name of the split
            samples: List of samples in the split

        Returns:
            SplitStatistics for this split
        """
        by_type = defaultdict(int)
        by_category = defaultdict(int)
        multimodal_by_type = defaultdict(int)
        multimodal_by_category = defaultdict(int)
        multimodal = 0
        text_only = 0
        with_tests = 0

        for sample in samples:
            qtype = sample.question_type or "unknown"
            category = sample.category or "uncategorized"

            by_type[qtype] += 1
            by_category[category] += 1

            if sample.image_path:
                multimodal += 1
                multimodal_by_type[qtype] += 1
                multimodal_by_category[category] += 1
            else:
                text_only += 1

            if sample.test_code:
                with_tests += 1

        return SplitStatistics(
            name=split_name,
            total=len(samples),
            by_type=dict(by_type),
            by_category=dict(by_category),
            multimodal=multimodal,
            text_only=text_only,
            with_tests=with_tests,
            multimodal_by_type=dict(multimodal_by_type),
            multimodal_by_category=dict(multimodal_by_category),
        )

    def get_statistics(self) -> DatasetStatistics:
        """Get computed statistics, analyzing if needed.

        Returns:
            DatasetStatistics
        """
        if self._statistics is None:
            return self.analyze()
        return self._statistics

    def get_multimodal_samples_by_type(self) -> dict[str, Sample]:
        """Get one multimodal sample for each question type.

        Returns:
            Dictionary mapping question type to a multimodal sample
        """
        samples_by_type: dict[str, Sample] = {}

        for split_samples in self._samples.values():
            for sample in split_samples:
                if sample.image_path and sample.question_type not in samples_by_type:
                    samples_by_type[sample.question_type] = sample

        return samples_by_type

    def save_statistics(self, output_path: Path) -> Path:
        """Save statistics to JSON file.

        Args:
            output_path: Path to save statistics JSON

        Returns:
            Path to saved file
        """
        stats = self.get_statistics()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, indent=2, ensure_ascii=False)

        return output_path

    @property
    def samples(self) -> dict[str, list[Sample]]:
        """Get loaded samples by split."""
        return self._samples
