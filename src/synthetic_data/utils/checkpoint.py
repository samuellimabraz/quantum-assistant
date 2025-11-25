"""Checkpoint manager for incremental pipeline progress tracking."""

import json
import pickle
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class CheckpointManager(Generic[T]):
    """Manages incremental checkpoints for pipeline stages with batch processing."""

    def __init__(self, checkpoint_dir: Path, stage_name: str):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
            stage_name: Name of the pipeline stage (parse, transcribe, etc.)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.stage_name = stage_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.checkpoint_dir / f"{stage_name}_checkpoint.pkl"
        self.metadata_file = self.checkpoint_dir / f"{stage_name}_metadata.json"

    def save_checkpoint(self, data: list[T], metadata: dict[str, Any]) -> None:
        """
        Save checkpoint with current progress.

        Args:
            data: List of processed items
            metadata: Metadata about the checkpoint (processed_count, total, etc.)
        """
        # Save data
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(data, f)

        # Save metadata
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_checkpoint(self) -> tuple[list[T], dict[str, Any]] | None:
        """
        Load checkpoint if it exists.

        Returns:
            Tuple of (data, metadata) or None if no checkpoint exists
        """
        if not self.checkpoint_file.exists() or not self.metadata_file.exists():
            return None

        try:
            # Load data
            with open(self.checkpoint_file, "rb") as f:
                data = pickle.load(f)

            # Load metadata
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)

            return data, metadata
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return None

    def clear_checkpoint(self) -> None:
        """Remove checkpoint files."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()

    def exists(self) -> bool:
        """Check if checkpoint exists."""
        return self.checkpoint_file.exists() and self.metadata_file.exists()


class BatchCheckpointProcessor:
    """Processes items in batches with automatic checkpoint saving."""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        batch_size: int = 100,
        auto_save_interval: int = 1,
    ):
        """
        Initialize batch checkpoint processor.

        Args:
            checkpoint_manager: CheckpointManager instance
            batch_size: Size of batches for processing
            auto_save_interval: Save checkpoint every N batches
        """
        self.checkpoint_manager = checkpoint_manager
        self.batch_size = batch_size
        self.auto_save_interval = auto_save_interval

    def process_with_checkpoints(
        self,
        items: list[Any],
        process_fn: Callable[[list[Any]], list[Any]],
        resume: bool = True,
    ) -> list[Any]:
        """
        Process items in batches with checkpoint saving.

        Args:
            items: Items to process
            process_fn: Function that processes a batch of items
            resume: Whether to resume from checkpoint if available

        Returns:
            List of processed results
        """
        results = []
        processed_indices = set()

        # Try to load checkpoint
        if resume and self.checkpoint_manager.exists():
            checkpoint_data = self.checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                results, metadata = checkpoint_data
                processed_indices = set(metadata.get("processed_indices", []))
                print(
                    f"Resuming from checkpoint: {len(results)} items already processed"
                )

        # Process remaining items
        remaining_items = [
            (i, item) for i, item in enumerate(items) if i not in processed_indices
        ]

        if not remaining_items:
            print("All items already processed!")
            return results

        # Process in batches
        for batch_idx in range(0, len(remaining_items), self.batch_size):
            batch = remaining_items[batch_idx : batch_idx + self.batch_size]
            batch_items = [item for _, item in batch]
            batch_indices = [idx for idx, _ in batch]

            # Process batch
            batch_results = process_fn(batch_items)

            # Update results
            results.extend(batch_results)
            processed_indices.update(batch_indices)

            # Save checkpoint at intervals
            if (batch_idx // self.batch_size + 1) % self.auto_save_interval == 0:
                metadata = {
                    "processed_indices": list(processed_indices),
                    "total_items": len(items),
                    "completed_items": len(processed_indices),
                }
                self.checkpoint_manager.save_checkpoint(results, metadata)
                print(
                    f"  Checkpoint saved: {len(processed_indices)}/{len(items)} items"
                )

        # Final save
        metadata = {
            "processed_indices": list(processed_indices),
            "total_items": len(items),
            "completed_items": len(processed_indices),
        }
        self.checkpoint_manager.save_checkpoint(results, metadata)

        return results


