"""Checkpoint manager for incremental pipeline progress tracking."""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class GenerationStageState:
    """State for a generation pipeline stage.

    This captures the progress within a specific stage, including:
    - Which items have been processed (by index)
    - Results from processing
    - Additional stage-specific data
    """

    stage_name: str
    processed_indices: list[int] = field(default_factory=list)
    results: list[Any] = field(default_factory=list)
    total_count: int = 0
    extra_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization (excludes results - stored separately)."""
        return {
            "stage_name": self.stage_name,
            "processed_indices": self.processed_indices,
            "total_count": self.total_count,
            "extra_data": self.extra_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], results: list[Any]) -> "GenerationStageState":
        """Create from dictionary with results."""
        return cls(
            stage_name=data.get("stage_name", ""),
            processed_indices=data.get("processed_indices", []),
            results=results,
            total_count=data.get("total_count", 0),
            extra_data=data.get("extra_data", {}),
        )


@dataclass
class GenerationCheckpoint:
    """Complete checkpoint state for the generation pipeline.

    This is the canonical checkpoint format. All checkpoint saves and loads
    should use this structure.
    """

    # Global pipeline state
    samples: list[Any] = field(default_factory=list)
    rejected_samples: list[dict] = field(default_factory=list)
    code_failures: list[dict] = field(default_factory=list)

    # Items waiting to be processed by current/next stage
    pending_candidates: list[Any] = field(default_factory=list)
    pending_samples: list[Any] = field(default_factory=list)

    # Current stage progress
    stage_state: GenerationStageState | None = None

    # Metadata
    current_stage: str = "start"
    attempt: int = 0

    def is_complete(self) -> bool:
        """Check if checkpoint represents a completed attempt."""
        return self.current_stage == "complete"


class CheckpointManager(Generic[T]):
    """Manages incremental checkpoints for pipeline stages."""

    def __init__(self, checkpoint_dir: Path, stage_name: str):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
            stage_name: Name of the pipeline stage
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.stage_name = stage_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.checkpoint_dir / f"{stage_name}_checkpoint.pkl"
        self.metadata_file = self.checkpoint_dir / f"{stage_name}_metadata.json"

    def save_checkpoint(self, data: Any, metadata: dict[str, Any]) -> None:
        """Save checkpoint with current progress.

        Args:
            data: Checkpoint data (pickled)
            metadata: Metadata about the checkpoint (JSON)
        """
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(data, f)

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def load_checkpoint(self) -> tuple[Any, dict[str, Any]] | None:
        """Load checkpoint if it exists.

        Returns:
            Tuple of (data, metadata) or None if no checkpoint exists
        """
        if not self.checkpoint_file.exists() or not self.metadata_file.exists():
            return None

        try:
            with open(self.checkpoint_file, "rb") as f:
                data = pickle.load(f)

            with open(self.metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            return data, metadata
        except (OSError, json.JSONDecodeError, pickle.UnpicklingError) as e:
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

    def save_generation_checkpoint(
        self,
        checkpoint: GenerationCheckpoint,
    ) -> None:
        """Save a generation checkpoint.

        Args:
            checkpoint: Complete checkpoint state
        """
        # Serialize stage state separately (results stored in pickle, metadata in JSON)
        stage_dict = None
        stage_results = []
        if checkpoint.stage_state:
            stage_dict = checkpoint.stage_state.to_dict()
            stage_results = checkpoint.stage_state.results

        data = {
            "samples": checkpoint.samples,
            "rejected_samples": checkpoint.rejected_samples,
            "code_failures": checkpoint.code_failures,
            "pending_candidates": checkpoint.pending_candidates,
            "pending_samples": checkpoint.pending_samples,
            "stage_state": stage_dict,
            "stage_results": stage_results,
        }

        metadata = {
            "current_stage": checkpoint.current_stage,
            "attempt": checkpoint.attempt,
            "samples_count": len(checkpoint.samples),
            "pending_candidates_count": len(checkpoint.pending_candidates),
            "pending_samples_count": len(checkpoint.pending_samples),
        }

        if checkpoint.stage_state:
            metadata["stage_name"] = checkpoint.stage_state.stage_name
            metadata["stage_processed"] = len(checkpoint.stage_state.processed_indices)
            metadata["stage_total"] = checkpoint.stage_state.total_count

        self.save_checkpoint(data, metadata)

    def load_generation_checkpoint(self) -> GenerationCheckpoint | None:
        """Load a generation checkpoint.

        Returns:
            GenerationCheckpoint or None if no checkpoint exists
        """
        result = self.load_checkpoint()
        if not result:
            return None

        data, metadata = result

        # Handle dict format (current format)
        if isinstance(data, dict):
            # Reconstruct stage state if present
            stage_state = None
            stage_dict = data.get("stage_state")
            stage_results = data.get("stage_results", [])
            if stage_dict:
                stage_state = GenerationStageState.from_dict(stage_dict, stage_results)

            return GenerationCheckpoint(
                samples=data.get("samples", []),
                rejected_samples=data.get("rejected_samples", []),
                code_failures=data.get("code_failures", []),
                pending_candidates=data.get("pending_candidates", []),
                pending_samples=data.get("pending_samples", []),
                stage_state=stage_state,
                current_stage=metadata.get("current_stage", "start"),
                attempt=metadata.get("attempt", 0),
            )

        # Handle list format (very old legacy - samples only)
        if isinstance(data, list):
            return GenerationCheckpoint(
                samples=data,
                current_stage=metadata.get("current_stage", "complete"),
            )

        return None

    # Keep old methods for backward compatibility with other stages (transcribe, filter, etc.)
    def save_stage_checkpoint(
        self,
        stage_state: GenerationStageState,
        global_state: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Save checkpoint with stage-level granularity.

        Args:
            stage_state: Current stage state with results
            global_state: Global pipeline state
            metadata: Additional metadata
        """
        checkpoint = GenerationCheckpoint(
            samples=global_state.get("samples", []),
            rejected_samples=global_state.get("rejected_samples", []),
            code_failures=global_state.get("code_failures", []),
            pending_candidates=global_state.get("pending_candidates", []),
            pending_samples=global_state.get("pending_samples", []),
            stage_state=stage_state,
            current_stage=metadata.get("current_stage", "start"),
            attempt=metadata.get("attempt", 0),
        )
        self.save_generation_checkpoint(checkpoint)

    def load_stage_checkpoint(
        self,
    ) -> tuple[dict[str, Any], GenerationStageState | None, dict[str, Any]] | None:
        """Load checkpoint with stage-level data.

        Returns:
            Tuple of (global_state, stage_state, metadata) or None
        """
        checkpoint = self.load_generation_checkpoint()
        if not checkpoint:
            return None

        global_state = {
            "samples": checkpoint.samples,
            "rejected_samples": checkpoint.rejected_samples,
            "code_failures": checkpoint.code_failures,
            "pending_candidates": checkpoint.pending_candidates,
            "pending_samples": checkpoint.pending_samples,
        }

        metadata = {
            "current_stage": checkpoint.current_stage,
            "attempt": checkpoint.attempt,
        }

        return global_state, checkpoint.stage_state, metadata


class BatchCheckpointProcessor:
    """Processes items in batches with automatic checkpoint saving."""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        batch_size: int = 100,
        auto_save_interval: int = 1,
    ):
        """Initialize batch checkpoint processor.

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
        """Process items in batches with checkpoint saving.

        Args:
            items: Items to process
            process_fn: Function that processes a batch of items
            resume: Whether to resume from checkpoint if available

        Returns:
            List of processed results
        """
        results = []
        processed_indices = set()

        if resume and self.checkpoint_manager.exists():
            checkpoint_data = self.checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                results, metadata = checkpoint_data
                processed_indices = set(metadata.get("processed_indices", []))
                print(f"Resuming from checkpoint: {len(results)} items already processed")

        remaining_items = [(i, item) for i, item in enumerate(items) if i not in processed_indices]

        if not remaining_items:
            print("All items already processed!")
            return results

        for batch_idx in range(0, len(remaining_items), self.batch_size):
            batch = remaining_items[batch_idx : batch_idx + self.batch_size]
            batch_items = [item for _, item in batch]
            batch_indices = [idx for idx, _ in batch]

            batch_results = process_fn(batch_items)

            results.extend(batch_results)
            processed_indices.update(batch_indices)

            if (batch_idx // self.batch_size + 1) % self.auto_save_interval == 0:
                metadata = {
                    "processed_indices": list(processed_indices),
                    "total_items": len(items),
                    "completed_items": len(processed_indices),
                }
                self.checkpoint_manager.save_checkpoint(results, metadata)
                print(f"  Checkpoint saved: {len(processed_indices)}/{len(items)} items")

        metadata = {
            "processed_indices": list(processed_indices),
            "total_items": len(items),
            "completed_items": len(processed_indices),
        }
        self.checkpoint_manager.save_checkpoint(results, metadata)

        return results
