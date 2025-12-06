"""Curate stage - Quality curation of generated samples."""

import asyncio
import json
import pickle
from pathlib import Path

from rich.console import Console
from rich.table import Table

from synthetic_data.config import PipelineConfig
from synthetic_data.models import Message, ModelRegistry, Sample
from synthetic_data.utils import CheckpointManager

console = Console()


class CurateStage:
    """Curate stage - quality curation of samples.

    Input: answered/samples.pkl
    Output: curated/samples.pkl
    """

    stage_name = "curated"

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
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir, "curate")

        self.curation_prompt = config.prompts.sample_curation
        self.system_prompt = config.prompts.sample_curation_system

    @property
    def output_file(self) -> Path:
        return self.output_dir / "samples.pkl"

    def get_input_path(self) -> Path:
        """Get input path - answered samples."""
        answered_dir = self.base_dir / "answered"
        if (answered_dir / "samples.pkl").exists():
            return answered_dir / "samples.pkl"
        raise FileNotFoundError("No answered samples found. Run 'answer' first.")

    def run(self, progress_callback=None) -> tuple[list[Sample], list[dict]]:
        """Run the curate stage."""
        return asyncio.run(self.run_async(progress_callback))

    async def run_async(self, progress_callback=None) -> tuple[list[Sample], list[dict]]:
        """Run curate stage asynchronously."""
        # Check if curation is enabled
        if not self.gen_config.enable_curate_filtering:
            console.print("[yellow]Quality curation disabled in config[/yellow]")
            return self._copy_input_to_output(), []

        # Check if output exists
        if self.output_file.exists() and not self.no_cache:
            console.print(f"[cyan]Loading existing curated samples: {self.output_file}[/cyan]")
            with open(self.output_file, "rb") as f:
                samples = pickle.load(f)
            return samples, []

        # Load input
        console.print(f"[cyan]Loading samples from: {self.get_input_path()}[/cyan]")
        with open(self.get_input_path(), "rb") as f:
            samples = pickle.load(f)

        console.print(f"[cyan]Loaded {len(samples)} samples[/cyan]")

        # Load checkpoint
        skip_indices, cached_decisions = self._load_checkpoint()

        # Get LLM client
        llm_client = self.model_registry.get_llm_client(self.gen_config.curate_model)

        try:
            curated, rejected = await self._curate_samples_async(
                samples, llm_client, skip_indices, cached_decisions, progress_callback
            )
        finally:
            await llm_client.aclose()

        # Save output
        with open(self.output_file, "wb") as f:
            pickle.dump(curated, f)

        # Clear checkpoint
        self.checkpoint_manager.clear_checkpoint()

        # Save rejected
        if rejected:
            rejected_file = self.output_dir / "rejected.jsonl"
            with open(rejected_file, "w", encoding="utf-8") as f:
                for entry in rejected:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write("\n")

        # Print summary
        self._print_summary(len(samples), len(curated), len(rejected))

        return curated, rejected

    def _copy_input_to_output(self) -> list[Sample]:
        """Copy input directly to output when curation is disabled."""
        import shutil

        input_path = self.get_input_path()
        shutil.copy2(input_path, self.output_file)

        with open(self.output_file, "rb") as f:
            samples = pickle.load(f)

        console.print(f"[green]✓ Copied {len(samples)} samples (curation disabled)[/green]")
        return samples

    async def _curate_samples_async(
        self,
        samples: list[Sample],
        llm_client,
        skip_indices: set[int],
        cached_decisions: dict[int, tuple[str, str]],
        progress_callback=None,
    ) -> tuple[list[Sample], list[dict]]:
        """Curate samples for quality."""
        # Build curation requests
        curation_batches = []
        batch_indices = []

        for idx, sample in enumerate(samples):
            if idx in skip_indices:
                continue

            context_preview = ""
            if sample.metadata:
                context_preview = sample.metadata.get("context_preview", "")
            if not context_preview:
                context_preview = "N/A"

            image_description = "N/A"
            if sample.image_path and sample.metadata:
                image_description = sample.metadata.get("image_transcription", "N/A")

            user_prompt = self.curation_prompt.format(
                question=sample.question,
                answer=sample.answer,
                question_type=sample.question_type,
                has_image="yes" if sample.image_path else "no",
                has_test="yes" if sample.test_code else "no",
                context_preview=context_preview[:800] if context_preview else "N/A",
                image_description=image_description[:500] if image_description else "N/A",
            )

            curation_batches.append(
                [
                    Message(role="system", content=self.system_prompt),
                    Message(role="user", content=user_prompt),
                ]
            )
            batch_indices.append(idx)

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
                curation_batches
            ):
                current_decisions = dict(decisions)
                for i in range(completed):
                    batch_idx = batch_indices[i]
                    if batch_idx not in current_decisions:
                        current_decisions[batch_idx] = ("PENDING", "")

                data = {"decisions": current_decisions}
                metadata = {
                    "processed_indices": list(skip_indices) + batch_indices[:completed],
                    "total_processed": len(skip_indices) + completed,
                }
                self.checkpoint_manager.save_checkpoint(data, metadata)
                last_checkpoint[0] = completed

        # Process cached decisions
        decisions = {idx: cached_decisions[idx] for idx in skip_indices if idx in cached_decisions}

        # Generate curation decisions
        if curation_batches:
            responses = await llm_client.generate_batch_async(
                curation_batches,
                max_concurrent=self.gen_config.llm_concurrency,
                temperature=0.1,
                progress_callback=update_callback,
            )

            # Parse responses
            for resp_idx, response in enumerate(responses):
                sample_idx = batch_indices[resp_idx]
                decision, reason = self._parse_curation_response(response)
                decisions[sample_idx] = (decision, reason)

        # Build curated and rejected lists
        curated = []
        rejected = []

        for idx, sample in enumerate(samples):
            if idx in decisions:
                decision, reason = decisions[idx]
                if decision == "PASS":
                    curated.append(sample)
                else:
                    rejected.append(
                        {
                            "question": sample.question,
                            "answer": sample.answer,
                            "question_type": sample.question_type,
                            "image_path": sample.image_path,
                            "rejection_reason": reason,
                        }
                    )

        return curated, rejected

    def _parse_curation_response(self, response: str) -> tuple[str, str]:
        """Parse curation response."""
        response = response.strip()
        if not response:
            return "REJECT", "Empty response from model"

        lines = [line.strip() for line in response.split("\n") if line.strip()]

        decision = None
        reason = ""

        for line in lines:
            line_upper = line.upper()

            if line_upper.startswith("DECISION:") or line_upper.startswith("DEC:"):
                decision_value = line.split(":", 1)[1].strip().upper()
                if "PASS" in decision_value:
                    decision = "PASS"
                elif "REJECT" in decision_value:
                    decision = "REJECT"

            elif line_upper.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        if decision is None:
            first_word = lines[0].upper().split()[0] if lines else ""
            if "PASS" in first_word:
                decision = "PASS"
            elif "REJECT" in first_word:
                decision = "REJECT"
            else:
                decision = "REJECT"
                reason = f"Unclear response format: {response[:100]}"

        if decision == "REJECT" and not reason:
            reason = "Quality check failed (no reason provided)"

        return decision, reason

    def _load_checkpoint(self) -> tuple[set[int], dict[int, tuple[str, str]]]:
        """Load checkpoint if exists."""
        if self.no_cache:
            return set(), {}

        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        if not checkpoint_data:
            return set(), {}

        data, metadata = checkpoint_data
        processed_indices = set(metadata.get("processed_indices", []))
        cached_decisions = data.get("decisions", {})

        if processed_indices:
            console.print(f"[cyan]Resuming: {len(processed_indices)} already curated[/cyan]")

        return processed_indices, cached_decisions

    def _print_summary(self, total: int, passed: int, rejected: int) -> None:
        """Print summary table."""
        table = Table(title="Quality Curation Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Input Samples", str(total))
        table.add_row("Passed", str(passed))
        table.add_row("Rejected", str(rejected))
        table.add_row("Pass Rate", f"{passed / total * 100:.1f}%" if total > 0 else "N/A")

        console.print("\n")
        console.print(table)
        console.print(f"\n[green]✓ Saved to: {self.output_file}[/green]")
