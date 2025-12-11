"""Filter candidates stage - Filter candidates for quality."""

import asyncio
import json
import pickle
from pathlib import Path

from rich.console import Console
from rich.table import Table

from synthetic_data.config import PipelineConfig
from synthetic_data.models import LLMClient, Message, ModelRegistry
from synthetic_data.utils import CheckpointManager

from synthetic_data.generators.types import InputCandidate

console = Console()


class FilterCandidatesStage:
    """Filter candidates stage - filters candidates for quality.

    Input: planned/candidates.pkl
    Output: filtered_candidates/candidates.pkl
    """

    stage_name = "filtered_candidates"

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
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir, "filter_candidates")

        self.filter_prompt = config.prompts.candidate_filter_prompt
        self.system_prompt = config.prompts.candidate_filter_system

    @property
    def output_file(self) -> Path:
        return self.output_dir / "candidates.pkl"

    def get_input_path(self) -> Path:
        """Get input path - planned candidates."""
        planned_dir = self.base_dir / "planned"
        if (planned_dir / "candidates.pkl").exists():
            return planned_dir / "candidates.pkl"
        raise FileNotFoundError("No planned candidates found. Run 'plan' first.")

    def run(self, progress_callback=None) -> list[InputCandidate]:
        """Run the filter candidates stage."""
        return asyncio.run(self.run_async(progress_callback))

    async def run_async(self, progress_callback=None) -> list[InputCandidate]:
        """Run filter candidates stage asynchronously."""
        # Check if filtering is enabled
        if not self.gen_config.enable_candidate_filtering:
            console.print("[yellow]Candidate filtering disabled in config[/yellow]")
            return self._copy_input_to_output()

        # Check if output exists
        if self.output_file.exists() and not self.no_cache:
            console.print(f"[cyan]Loading existing filtered candidates: {self.output_file}[/cyan]")
            with open(self.output_file, "rb") as f:
                return pickle.load(f)

        # Load input
        console.print(f"[cyan]Loading candidates from: {self.get_input_path()}[/cyan]")
        with open(self.get_input_path(), "rb") as f:
            candidates = pickle.load(f)

        console.print(f"[cyan]Loaded {len(candidates)} candidates[/cyan]")

        # Load checkpoint
        skip_indices, cached_decisions = self._load_checkpoint()

        # Get LLM client
        llm_client = self.model_registry.get_llm_client(self.gen_config.question_model)

        try:
            filtered, debug_info = await self._filter_candidates_async(
                candidates, llm_client, skip_indices, cached_decisions, progress_callback
            )
        finally:
            await llm_client.aclose()

        # Save output
        with open(self.output_file, "wb") as f:
            pickle.dump(filtered, f)

        # Clear checkpoint
        self.checkpoint_manager.clear_checkpoint()

        # Save debug info
        debug_file = self.output_dir / "filter_decisions.jsonl"
        with open(debug_file, "w", encoding="utf-8") as f:
            for entry in debug_info:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")

        # Print summary
        self._print_summary(len(candidates), len(filtered))

        return filtered

    def _copy_input_to_output(self) -> list[InputCandidate]:
        """Copy input directly to output when filtering is disabled."""
        import shutil

        input_path = self.get_input_path()
        shutil.copy2(input_path, self.output_file)

        with open(self.output_file, "rb") as f:
            candidates = pickle.load(f)

        console.print(f"[green]✓ Copied {len(candidates)} candidates (filtering disabled)[/green]")
        return candidates

    async def _filter_candidates_async(
        self,
        candidates: list[InputCandidate],
        llm_client: LLMClient,
        skip_indices: set[int],
        cached_decisions: dict[int, tuple[str, str]],
        progress_callback=None,
    ) -> tuple[list[InputCandidate], list[dict]]:
        """Filter candidates for quality."""
        # Build filter requests
        filter_batches = []
        batch_indices = []
        debug_info = []

        for idx, candidate in enumerate(candidates):
            if idx in skip_indices:
                continue

            prompt = self.filter_prompt.format(
                question=candidate.question,
                question_type=candidate.question_type.value,
                has_image="yes" if candidate.is_multimodal else "no",
                has_test="yes" if candidate.test_code else "no",
                context_preview=(
                    candidate.context[:2200] + "..."
                    if len(candidate.context) > 2200
                    else candidate.context
                ),
            )

            filter_batches.append(
                [
                    Message(role="system", content=self.system_prompt),
                    Message(role="user", content=prompt),
                ]
            )
            batch_indices.append(idx)

        # Process cached decisions first
        decisions = {idx: cached_decisions[idx] for idx in skip_indices if idx in cached_decisions}

        # Set progress total
        if progress_callback and hasattr(progress_callback, "set_total"):
            progress_callback.set_total(len(candidates))

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
                filter_batches
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

        # Generate filter decisions
        if filter_batches:
            responses = await llm_client.generate_batch_async(
                filter_batches,
                max_concurrent=self.gen_config.llm_concurrency,
                temperature=0.2,
                progress_callback=update_callback,
            )

            # Parse responses
            for resp_idx, response in enumerate(responses):
                candidate_idx = batch_indices[resp_idx]
                decision, reason = self._parse_filter_response(response)
                decisions[candidate_idx] = (decision, reason)

                debug_info.append(
                    {
                        "index": candidate_idx,
                        "question_preview": candidates[candidate_idx].question[:200],
                        "decision": decision,
                        "reason": reason,
                    }
                )

        # Build filtered list
        filtered = []
        for idx, candidate in enumerate(candidates):
            if idx in decisions:
                decision, reason = decisions[idx]
                if decision == "PASS":
                    filtered.append(candidate)

        return filtered, debug_info

    def _parse_filter_response(self, response: str) -> tuple[str, str]:
        """Parse filter response to get decision and reason."""
        response = response.strip().upper()

        if "PASS" in response or "YES" in response:
            return "PASS", ""

        reason = "Quality check failed"
        lines = response.split("\n")
        for line in lines:
            if "REASON" in line.upper() and ":" in line:
                reason = line.split(":", 1)[1].strip()
                break

        return "REJECT", reason

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
            console.print(f"[cyan]Resuming: {len(processed_indices)} already filtered[/cyan]")

        return processed_indices, cached_decisions

    def _print_summary(self, total: int, passed: int) -> None:
        """Print summary table."""
        rejected = total - passed

        table = Table(title="Candidate Filtering Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Input Candidates", str(total))
        table.add_row("Passed", str(passed))
        table.add_row("Rejected", str(rejected))
        table.add_row("Pass Rate", f"{passed / total * 100:.1f}%" if total > 0 else "N/A")

        console.print("\n")
        console.print(table)
        console.print(f"\n[green]✓ Saved to: {self.output_file}[/green]")
