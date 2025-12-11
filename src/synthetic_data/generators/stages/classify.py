"""Classify stage - Classify samples into categories."""

import asyncio
import json
import pickle
from pathlib import Path

from rich.console import Console
from rich.table import Table

from synthetic_data.config import PipelineConfig
from synthetic_data.generators.category import CategoryClassifier
from synthetic_data.models import ModelRegistry, Sample
from synthetic_data.utils import CheckpointManager

console = Console()


class ClassifyStage:
    """Classify stage - classifies samples into categories.

    Input: curated/samples.pkl
    Output: generated/samples.pkl
    """

    stage_name = "generated"

    def __init__(
        self,
        config: PipelineConfig,
        model_registry: ModelRegistry,
        base_dir: Path,
        no_cache: bool = False,
    ):
        self.config = config
        self.gen_config = config.generation
        self.model_registry = model_registry
        self.base_dir = Path(base_dir)
        self.no_cache = no_cache

        self.output_dir = self.base_dir / self.stage_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.base_dir / ".checkpoints"
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir, "classify")

        self.classification_prompt = config.prompts.category_classification
        self.system_prompt = config.prompts.category_classification_system

    @property
    def output_file(self) -> Path:
        return self.output_dir / "samples.pkl"

    def get_input_path(self) -> Path:
        """Get input path - curated samples."""
        curated_dir = self.base_dir / "curated"
        if (curated_dir / "samples.pkl").exists():
            return curated_dir / "samples.pkl"

        # Fallback to answered if curation was skipped
        answered_dir = self.base_dir / "answered"
        if (answered_dir / "samples.pkl").exists():
            return answered_dir / "samples.pkl"

        raise FileNotFoundError("No samples found. Run 'answer' or 'curate' first.")

    def run(self, progress_callback=None) -> list[Sample]:
        """Run the classify stage."""
        return asyncio.run(self.run_async(progress_callback))

    async def run_async(self, progress_callback=None) -> list[Sample]:
        """Run classify stage asynchronously."""
        # Check if output exists
        if self.output_file.exists() and not self.no_cache:
            console.print(f"[cyan]Loading existing classified samples: {self.output_file}[/cyan]")
            with open(self.output_file, "rb") as f:
                return pickle.load(f)

        # Load input
        console.print(f"[cyan]Loading samples from: {self.get_input_path()}[/cyan]")
        with open(self.get_input_path(), "rb") as f:
            samples = pickle.load(f)

        console.print(f"[cyan]Loaded {len(samples)} samples[/cyan]")

        # Load checkpoint
        skip_indices, cached_categories = self._load_checkpoint()

        # Get LLM client
        llm_client = self.model_registry.get_llm_client(self.gen_config.curate_model)
        classifier = CategoryClassifier(self.config.categories, llm_client)

        try:
            classified_samples = await self._classify_samples_async(
                samples, classifier, skip_indices, cached_categories, progress_callback
            )
        finally:
            await llm_client.aclose()

        # Save output
        with open(self.output_file, "wb") as f:
            pickle.dump(classified_samples, f)

        # Clear checkpoint
        self.checkpoint_manager.clear_checkpoint()

        # Save JSONL
        self._save_jsonl(classified_samples)

        # Save summary
        self._save_summary(classified_samples)

        # Print summary
        self._print_summary(classified_samples)

        return classified_samples

    async def _classify_samples_async(
        self,
        samples: list[Sample],
        classifier: CategoryClassifier,
        skip_indices: set[int],
        cached_categories: dict[int, str],
        progress_callback=None,
    ) -> list[Sample]:
        """Classify samples into categories."""
        # Build list of samples to classify
        remaining_samples = []
        remaining_indices = []

        for idx, sample in enumerate(samples):
            if idx in skip_indices:
                # Use cached category
                if idx in cached_categories:
                    sample.category = cached_categories[idx]
            else:
                remaining_samples.append(sample)
                remaining_indices.append(idx)

        if not remaining_samples:
            return samples

        # Set progress total
        if progress_callback and hasattr(progress_callback, "set_total"):
            progress_callback.set_total(len(samples))

        # Track for checkpoint saving
        checkpoint_interval = 20
        last_checkpoint = [0]

        # Create callback adapter with checkpointing
        def update_callback(completed: int):
            actual_completed = len(skip_indices) + completed
            if progress_callback:
                if hasattr(progress_callback, "update"):
                    progress_callback.update(actual_completed)
                else:
                    progress_callback(actual_completed)

            # Save checkpoint periodically
            if completed - last_checkpoint[0] >= checkpoint_interval or completed == len(
                remaining_samples
            ):
                current_categories = dict(cached_categories)
                for i in range(completed):
                    idx = remaining_indices[i]
                    current_categories[idx] = ""  # Will be filled after batch completes

                data = {"categories": current_categories}
                metadata = {
                    "processed_indices": list(skip_indices) + remaining_indices[:completed],
                    "total_processed": len(skip_indices) + completed,
                }
                self.checkpoint_manager.save_checkpoint(data, metadata)
                last_checkpoint[0] = completed

        # Classify remaining samples
        categories = await classifier.classify_samples_async(
            remaining_samples,
            self.classification_prompt,
            self.system_prompt,
            max_concurrent=self.gen_config.llm_concurrency,
            progress_callback=update_callback,
        )

        # Update samples with categories
        for i, category in enumerate(categories):
            idx = remaining_indices[i]
            samples[idx].category = category

        return samples

    def _save_jsonl(self, samples: list[Sample]) -> None:
        """Save samples as JSONL."""
        jsonl_file = self.output_dir / "samples.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for sample in samples:
                json.dump(
                    {
                        "question": sample.question,
                        "answer": sample.answer,
                        "category": sample.category,
                        "question_type": sample.question_type,
                        "test_code": sample.test_code,
                        "entry_point": sample.entry_point,
                        "image_path": sample.image_path,
                        "source_path": sample.source_path,
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

    def _save_summary(self, samples: list[Sample]) -> None:
        """Save summary JSON."""
        summary = {
            "total_samples": len(samples),
            "multimodal_samples": sum(1 for s in samples if s.image_path),
            "text_only_samples": sum(1 for s in samples if not s.image_path),
            "samples_with_tests": sum(1 for s in samples if s.test_code),
            "by_type": {},
            "by_category": {},
        }

        for sample in samples:
            summary["by_type"][sample.question_type] = (
                summary["by_type"].get(sample.question_type, 0) + 1
            )
            summary["by_category"][sample.category] = (
                summary["by_category"].get(sample.category, 0) + 1
            )

        summary_file = self.output_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _load_checkpoint(self) -> tuple[set[int], dict[int, str]]:
        """Load checkpoint if exists."""
        if self.no_cache:
            return set(), {}

        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        if not checkpoint_data:
            return set(), {}

        data, metadata = checkpoint_data
        processed_indices = set(metadata.get("processed_indices", []))
        cached_categories = data.get("categories", {})

        if processed_indices:
            console.print(f"[cyan]Resuming: {len(processed_indices)} already classified[/cyan]")

        return processed_indices, cached_categories

    def _print_summary(self, samples: list[Sample]) -> None:
        """Print summary table."""
        table = Table(title="Classification Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Samples", str(len(samples)))
        table.add_row("With Tests", str(sum(1 for s in samples if s.test_code)))
        table.add_row("Multimodal", str(sum(1 for s in samples if s.image_path)))
        table.add_row("Text-only", str(sum(1 for s in samples if not s.image_path)))

        console.print("\n")
        console.print(table)

        # Type distribution
        type_counts = {}
        for s in samples:
            type_counts[s.question_type] = type_counts.get(s.question_type, 0) + 1

        type_table = Table(title="By Question Type")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", style="green", justify="right")

        for qtype, count in sorted(type_counts.items()):
            type_table.add_row(qtype, str(count))

        console.print("\n")
        console.print(type_table)

        # Category distribution
        cat_counts = {}
        for s in samples:
            cat_counts[s.category] = cat_counts.get(s.category, 0) + 1

        if cat_counts:
            cat_table = Table(title="By Category")
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Count", style="green", justify="right")

            for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
                cat_table.add_row(cat, str(count))

            console.print("\n")
            console.print(cat_table)

        console.print(f"\n[green]✓ Saved to: {self.output_file}[/green]")
