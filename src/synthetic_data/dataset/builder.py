"""Dataset building and splitting."""

import random
from pathlib import Path

from synthetic_data.config import DatasetConfig
from synthetic_data.models import Sample


class DatasetBuilder:
    """Build and split dataset into train/val/test."""

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

    def build(self, samples: list[Sample]) -> tuple[list[Sample], list[Sample], list[Sample]]:
        """
        Split samples into train/val/test sets.

        Args:
            samples: List of generated samples

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        # Shuffle samples
        shuffled = samples.copy()
        random.shuffle(shuffled)

        # Calculate split sizes
        total = len(shuffled)
        train_size = int(total * self.config.train_split)
        val_size = int(total * self.config.val_split)

        # Split
        train_samples = shuffled[:train_size]
        val_samples = shuffled[train_size : train_size + val_size]
        test_samples = shuffled[train_size + val_size :]

        return train_samples, val_samples, test_samples

    def stratified_build(
        self, samples: list[Sample]
    ) -> tuple[list[Sample], list[Sample], list[Sample]]:
        """
        Split samples with stratification by category and question type.

        Args:
            samples: List of generated samples

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        if len(samples) < 10:
            return self.build(samples)

        # Group by category and question type
        groups = {}
        for sample in samples:
            key = (sample.category, sample.question_type)
            if key not in groups:
                groups[key] = []
            groups[key].append(sample)

        train_samples = []
        val_samples = []
        test_samples = []

        # Split each group
        for group_samples in groups.values():
            random.shuffle(group_samples)

            total = len(group_samples)

            # Ensure at least 1 sample per split if group is large enough
            if total >= 3:
                train_size = max(1, int(total * self.config.train_split))
                val_size = max(1, int(total * self.config.val_split))
            else:
                # For tiny groups, just add to train
                train_samples.extend(group_samples)
                continue

            train_samples.extend(group_samples[:train_size])
            val_samples.extend(group_samples[train_size : train_size + val_size])
            test_samples.extend(group_samples[train_size + val_size :])

        # Final shuffle
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)

        return train_samples, val_samples, test_samples

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
