"""Answer stage - Generate and validate answers for candidates."""

import asyncio
import json
import pickle
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from synthetic_data.config import PipelineConfig, QuestionType
from synthetic_data.generators.prompts import PromptSet
from synthetic_data.generators.sessions import AnswerBatchProcessor, AnswerResult, AnswerSession
from synthetic_data.models import ModelRegistry, Sample
from synthetic_data.utils import CheckpointManager

from synthetic_data.generators.types import InputCandidate

console = Console()


class AnswerStage:
    """Answer stage - generates and validates answers.

    Input: filtered_candidates/candidates.pkl
    Output: answered/samples.pkl
    """

    stage_name = "answered"

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
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir, "answer")

        self.prompts = self._build_prompts()

    def _build_prompts(self) -> PromptSet:
        """Build prompt set from config."""
        return PromptSet(
            input_generation_system=self.config.prompts.input_generation_system,
            function_completion_prompt=self.config.prompts.function_completion_prompt,
            code_generation_prompt=self.config.prompts.code_generation_prompt,
            qa_prompt=self.config.prompts.qa_prompt,
            test_generation_prompt=self.config.prompts.test_generation_prompt,
            answer_generation_system=self.config.prompts.answer_generation_system,
            function_completion_answer_prompt=self.config.prompts.function_completion_answer_prompt,
            code_generation_answer_prompt=self.config.prompts.code_generation_answer_prompt,
            qa_answer_prompt=self.config.prompts.qa_answer_prompt,
            answer_correction_prompt=self.config.prompts.answer_correction_prompt,
        )

    @property
    def output_file(self) -> Path:
        return self.output_dir / "samples.pkl"

    def get_input_path(self) -> Path:
        """Get input path - filtered candidates."""
        filtered_dir = self.base_dir / "filtered_candidates"
        if (filtered_dir / "candidates.pkl").exists():
            return filtered_dir / "candidates.pkl"

        # Fallback to planned if filtering was skipped
        planned_dir = self.base_dir / "planned"
        if (planned_dir / "candidates.pkl").exists():
            return planned_dir / "candidates.pkl"

        raise FileNotFoundError("No candidates found. Run 'plan' or 'filter-candidates' first.")

    def run(self, progress_callback=None) -> tuple[list[Sample], list[dict]]:
        """Run the answer stage."""
        return asyncio.run(self.run_async(progress_callback))

    async def run_async(self, progress_callback=None) -> tuple[list[Sample], list[dict]]:
        """Run answer stage asynchronously."""
        # Check if output exists
        if self.output_file.exists() and not self.no_cache:
            console.print(f"[cyan]Loading existing samples: {self.output_file}[/cyan]")
            with open(self.output_file, "rb") as f:
                samples = pickle.load(f)
            return samples, []

        # Load input
        console.print(f"[cyan]Loading candidates from: {self.get_input_path()}[/cyan]")
        with open(self.get_input_path(), "rb") as f:
            candidates = pickle.load(f)

        console.print(f"[cyan]Loaded {len(candidates)} candidates[/cyan]")

        # Load checkpoint
        skip_indices, cached_results = self._load_checkpoint()

        # Get LLM client
        llm_client = self.model_registry.get_llm_client(self.gen_config.answer_model)

        try:
            samples, failures = await self._generate_answers_async(
                candidates, llm_client, skip_indices, cached_results, progress_callback
            )
        finally:
            await llm_client.aclose()

        # Save output
        with open(self.output_file, "wb") as f:
            pickle.dump(samples, f)

        # Clear checkpoint
        self.checkpoint_manager.clear_checkpoint()

        # Save failures
        if failures:
            failures_file = self.output_dir / "failures.jsonl"
            with open(failures_file, "w", encoding="utf-8") as f:
                for entry in failures:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write("\n")

        # Print summary
        self._print_summary(len(candidates), len(samples), len(failures))

        return samples, failures

    async def _generate_answers_async(
        self,
        candidates: list[InputCandidate],
        llm_client,
        skip_indices: set[int],
        cached_results: dict[int, AnswerResult],
        progress_callback=None,
    ) -> tuple[list[Sample], list[dict]]:
        """Generate answers for candidates."""
        # Build sessions
        sessions = []
        session_indices = []

        for idx, candidate in enumerate(candidates):
            if idx in skip_indices:
                continue

            use_image = candidate.is_multimodal
            system_prompt = self.prompts.get_answer_system_prompt(use_image=use_image)

            answer_prompt = self.prompts.get_answer_prompt(
                question_type=candidate.question_type,
                question=candidate.question,
                context=candidate.context,
                test_code=candidate.test_code,
            )

            session = AnswerSession(
                llm_client=llm_client,
                system_prompt=system_prompt,
                answer_prompt=answer_prompt,
                correction_prompt=self.prompts.answer_correction_prompt,
                candidate=candidate,
                max_iterations=self.gen_config.code_verification_max_iterations,
                timeout_seconds=self.gen_config.test_validation_timeout,
                skip_ibm_service_validation=self.gen_config.skip_ibm_service_validation,
            )
            sessions.append(session)
            session_indices.append(idx)

        # Process cached results
        all_results: list[Optional[AnswerResult]] = [None] * len(candidates)
        for idx, result in cached_results.items():
            all_results[idx] = result

        # Set progress total
        if progress_callback and hasattr(progress_callback, "set_total"):
            progress_callback.set_total(len(candidates))

        # Create callback adapter for cumulative progress
        def update_callback(completed: int):
            actual_completed = len(skip_indices) + completed
            if progress_callback:
                if hasattr(progress_callback, "update"):
                    progress_callback.update(actual_completed)
                else:
                    progress_callback(actual_completed)

        # Generate remaining
        if sessions:
            processor = AnswerBatchProcessor(
                llm_client=llm_client,
                max_concurrent=self.gen_config.llm_concurrency,
            )

            def checkpoint_callback(results: list, completed_indices: set[int]) -> None:
                # Update all_results
                for i, session_idx in enumerate(session_indices):
                    if i < len(results) and results[i] is not None:
                        all_results[session_idx] = results[i]
                self._save_checkpoint(all_results, skip_indices | completed_indices)

            results = await processor.process_sessions_async(
                sessions,
                progress_callback=update_callback,
                checkpoint_callback=checkpoint_callback,
                checkpoint_interval=self.gen_config.answer_checkpoint_interval,
            )

            # Merge results
            for i, result in enumerate(results):
                if result is not None:
                    all_results[session_indices[i]] = result

        # Convert to samples
        samples = []
        failures = []

        for idx, candidate in enumerate(candidates):
            result = all_results[idx]
            if result is None:
                failures.append(self._create_failure_entry(candidate, "No result"))
                continue

            if result.passed and result.answer:
                sample = self._create_sample(candidate, result.answer)
                samples.append(sample)
            elif result.answer:
                # QA can pass without test validation
                if candidate.question_type == QuestionType.QA:
                    sample = self._create_sample(candidate, result.answer)
                    samples.append(sample)
                else:
                    failures.append(
                        self._create_failure_entry(
                            candidate, "Failed validation", result.answer, result.error_history
                        )
                    )
            else:
                failures.append(self._create_failure_entry(candidate, "Empty answer"))

        return samples, failures

    def _create_sample(self, candidate: InputCandidate, answer: str) -> Sample:
        """Create a Sample from candidate and answer."""
        metadata = dict(candidate.chunk.metadata) if candidate.chunk.metadata else {}

        context_preview = candidate.chunk.text[:800] if candidate.chunk.text else ""
        metadata["context_preview"] = context_preview

        if candidate.target_image and candidate.target_image.transcription:
            metadata["image_transcription"] = candidate.target_image.transcription

        return Sample(
            question=candidate.question,
            answer=answer,
            category="",  # Will be set in classify stage
            question_type=candidate.question_type.value,
            test_code=candidate.test_code,
            entry_point=candidate.entry_point,
            image_path=candidate.target_image.resolved_path if candidate.target_image else None,
            source_path=str(candidate.chunk.source_path),
            metadata=metadata,
        )

    def _create_failure_entry(
        self,
        candidate: InputCandidate,
        error: str,
        answer: str = "",
        error_history: list = None,
    ) -> dict:
        """Create a failure entry."""
        return {
            "question": candidate.question,
            "answer": answer,
            "test_code": candidate.test_code,
            "entry_point": candidate.entry_point,
            "question_type": candidate.question_type.value,
            "error": error,
            "error_history": error_history or [],
        }

    def _load_checkpoint(self) -> tuple[set[int], dict[int, AnswerResult]]:
        """Load checkpoint if exists."""
        if self.no_cache:
            return set(), {}

        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        if not checkpoint_data:
            return set(), {}

        data, metadata = checkpoint_data
        processed_indices = set(metadata.get("processed_indices", []))
        cached_results = data.get("results", {})

        if processed_indices:
            console.print(f"[cyan]Resuming: {len(processed_indices)} already processed[/cyan]")

        return processed_indices, cached_results

    def _save_checkpoint(self, results: list[Optional[AnswerResult]], processed: set[int]) -> None:
        """Save checkpoint."""
        # Convert results to serializable format
        results_dict = {}
        for idx, result in enumerate(results):
            if result is not None:
                results_dict[idx] = result

        data = {"results": results_dict}
        metadata = {
            "processed_indices": list(processed),
            "total_processed": len(processed),
        }
        self.checkpoint_manager.save_checkpoint(data, metadata)

    def _print_summary(self, total: int, success: int, failed: int) -> None:
        """Print summary table."""
        table = Table(title="Answer Generation Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Input Candidates", str(total))
        table.add_row("Successful Samples", str(success))
        table.add_row("Failed", str(failed))
        table.add_row("Success Rate", f"{success / total * 100:.1f}%" if total > 0 else "N/A")

        console.print("\n")
        console.print(table)
        console.print(f"\n[green]✓ Saved to: {self.output_file}[/green]")
