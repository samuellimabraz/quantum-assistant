"""Dataset building with balanced stratified splitting."""

import random
from collections import defaultdict
from pathlib import Path

from synthetic_data.config import DatasetConfig
from synthetic_data.models import Sample


class DatasetBuilder:
    """Build and split dataset with balanced stratification."""

    def __init__(self, config: DatasetConfig, seed: int = 42):
        """
        Initialize dataset builder.

        Args:
            config: Dataset configuration
            seed: Random seed for splitting
        """
        self.config = config
        self.seed = seed
        random.seed(seed)

    def stratified_build(
        self, samples: list[Sample]
    ) -> tuple[list[Sample], list[Sample], list[Sample]]:
        """
        Split samples with stratification by category, question type, and multimodal status.

        Ensures balanced distribution across all three dimensions in train/val/test splits.

        Args:
            samples: List of generated samples

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        if len(samples) < 10:
            return self._simple_split(samples)

        # Group by (category, question_type, is_multimodal)
        strata = defaultdict(list)
        for sample in samples:
            is_multimodal = sample.image_path is not None
            key = (sample.category, sample.question_type, is_multimodal)
            strata[key].append(sample)

        train_samples = []
        val_samples = []
        test_samples = []

        # Split each stratum proportionally
        for stratum_samples in strata.values():
            random.shuffle(stratum_samples)
            n = len(stratum_samples)

            if n == 1:
                # Single sample goes to train
                train_samples.append(stratum_samples[0])
            elif n == 2:
                # Two samples: 1 train, 1 val
                train_samples.append(stratum_samples[0])
                val_samples.append(stratum_samples[1])
            else:
                # Multiple samples: split proportionally with minimum 1 per split
                train_size = max(1, int(n * self.config.train_split))
                val_size = max(1, int(n * self.config.val_split))

                # Adjust if we've allocated too many
                if train_size + val_size > n - 1:
                    # Ensure at least 1 for test if n >= 3
                    test_size = 1
                    remaining = n - test_size
                    train_size = max(
                        1,
                        int(
                            remaining
                            * self.config.train_split
                            / (self.config.train_split + self.config.val_split)
                        ),
                    )
                    val_size = remaining - train_size

                train_samples.extend(stratum_samples[:train_size])
                val_samples.extend(stratum_samples[train_size : train_size + val_size])
                test_samples.extend(stratum_samples[train_size + val_size :])

        # Final shuffle to mix strata
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)

        return train_samples, val_samples, test_samples

    def _simple_split(
        self, samples: list[Sample]
    ) -> tuple[list[Sample], list[Sample], list[Sample]]:
        """Simple split for small datasets."""
        shuffled = samples.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        if n < 3:
            return shuffled, [], []

        train_size = max(1, int(n * self.config.train_split))
        val_size = max(1, int(n * self.config.val_split))

        train_samples = shuffled[:train_size]
        val_samples = shuffled[train_size : train_size + val_size]
        test_samples = shuffled[train_size + val_size :]

        return train_samples, val_samples, test_samples

    def get_distribution_stats(self, samples: list[Sample]) -> dict:
        """
        Get distribution statistics for a sample set.

        Args:
            samples: List of samples

        Returns:
            Dictionary with distribution statistics
        """
        if not samples:
            return {
                "total": 0,
                "by_category": {},
                "by_type": {},
                "multimodal": 0,
                "text_only": 0,
                "multimodal_ratio": 0.0,
            }

        by_category = defaultdict(int)
        by_type = defaultdict(int)
        multimodal_count = 0

        for sample in samples:
            by_category[sample.category] += 1
            by_type[sample.question_type] += 1
            if sample.image_path:
                multimodal_count += 1

        return {
            "total": len(samples),
            "by_category": dict(by_category),
            "by_type": dict(by_type),
            "multimodal": multimodal_count,
            "text_only": len(samples) - multimodal_count,
            "multimodal_ratio": multimodal_count / len(samples) if samples else 0.0,
        }

    def print_split_comparison(
        self,
        train_samples: list[Sample],
        val_samples: list[Sample],
        test_samples: list[Sample],
    ):
        """Print comparison of distributions across splits."""
        train_stats = self.get_distribution_stats(train_samples)
        val_stats = self.get_distribution_stats(val_samples)
        test_stats = self.get_distribution_stats(test_samples)

        print("\n[Split Distribution Analysis]")
        print(
            f"Train: {train_stats['total']} samples "
            f"({train_stats['multimodal_ratio']:.1%} multimodal)"
        )
        print(
            f"Val:   {val_stats['total']} samples "
            f"({val_stats['multimodal_ratio']:.1%} multimodal)"
        )
        print(
            f"Test:  {test_stats['total']} samples "
            f"({test_stats['multimodal_ratio']:.1%} multimodal)"
        )

    def save_samples(self, samples: list[Sample], split: str) -> Path:
        """
        Save samples to disk.

        Args:
            samples: List of samples
            split: Split name (train/val/test)

        Returns:
            Path to saved file
        """
        import json

        output_dir = Path(self.config.final_dir) / split
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "data.jsonl"

        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                data = {
                    "question": sample.question,
                    "answer": sample.answer,
                    "category": sample.category,
                    "question_type": sample.question_type,
                    "image_path": sample.image_path,
                    "source_path": sample.source_path,
                    "metadata": sample.metadata,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"Saved {len(samples)} samples to {output_file}")
        return output_file
