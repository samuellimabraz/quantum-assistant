"""Generation pipeline using input planning and post-generation classification."""

import asyncio
import random
from pathlib import Path
from typing import Optional

from synthetic_data.config import PipelineConfig, QuestionType
from synthetic_data.extractors.chunker import Chunk
from synthetic_data.generators.allocation import (
    AllocationConfig,
    AllocationMetrics,
    Allocator,
    TypeAllocationConfig as AllocTypeConfig,
)
from synthetic_data.generators.category import CategoryClassifier
from synthetic_data.generators.planner import InputCandidate, InputPlanner
from synthetic_data.generators.prompts import PromptSet
from synthetic_data.generators.sessions import AnswerBatchProcessor, AnswerSession
from synthetic_data.models import ModelRegistry, Sample
from synthetic_data.utils import CheckpointManager, Deduplicator, GenerationTracer


class GenerationPipeline:
    """Pipeline for generating synthetic dataset samples.

    The pipeline uses:
    1. Input Planning: Generate candidates with over-allocation
    2. Candidate Filtering: Filter candidates for quality
    3. Answer Generation: Generate and validate answers
    4. Quality Curation: Final quality check
    5. Post-Generation Classification: Classify samples by content
    6. Trimming: Trim excess samples to match target distribution
    """

    def __init__(
        self,
        config: PipelineConfig,
        model_registry: ModelRegistry,
        checkpoint_manager: Optional[CheckpointManager] = None,
        output_dir: Optional[Path] = None,
        enable_tracing: bool = True,
    ):
        """Initialize generation pipeline.

        Args:
            config: Pipeline configuration
            model_registry: Model registry for LLM/VLM access
            checkpoint_manager: Optional checkpoint manager for resuming
            output_dir: Optional output directory for trace files
            enable_tracing: Whether to enable detailed prompt/response tracing
        """
        self.config = config
        self.gen_config = config.generation
        self.model_registry = model_registry
        self.checkpoint_manager = checkpoint_manager

        # Initialize tracer
        self.tracer = None
        if enable_tracing and output_dir:
            trace_dir = output_dir / "traces"
            self.tracer = GenerationTracer(trace_dir, enabled=True)

        # Create prompt set
        self.prompts = PromptSet(
            input_generation_system=config.prompts.input_generation_system,
            function_completion_prompt=config.prompts.function_completion_prompt,
            code_generation_prompt=config.prompts.code_generation_prompt,
            qa_prompt=config.prompts.qa_prompt,
            test_generation_prompt=config.prompts.test_generation_prompt,
            answer_generation_system=config.prompts.answer_generation_system,
            function_completion_answer_prompt=config.prompts.function_completion_answer_prompt,
            code_generation_answer_prompt=config.prompts.code_generation_answer_prompt,
            qa_answer_prompt=config.prompts.qa_answer_prompt,
            answer_correction_prompt=config.prompts.answer_correction_prompt,
            content_quality_check=config.prompts.content_quality_check,
            content_filter_system=config.prompts.content_filter_system,
            image_quality_check=config.prompts.image_quality_check,
            image_filter_system=config.prompts.image_filter_system,
            image_transcription=config.prompts.image_transcription,
            image_transcription_system=config.prompts.image_transcription_system,
            candidate_filter_system=config.prompts.candidate_filter_system,
            candidate_filter_prompt=config.prompts.candidate_filter_prompt,
            category_classification=config.prompts.category_classification,
            category_classification_system=config.prompts.category_classification_system,
            sample_curation=config.prompts.sample_curation,
            sample_curation_system=config.prompts.sample_curation_system,
        )

        random.seed(config.seed)

        # Initialize deduplicator
        self.deduplicator = None
        if self.gen_config.enable_deduplication:
            self.deduplicator = Deduplicator(self.gen_config.similarity_threshold)

    def generate_samples(
        self,
        chunks: list[Chunk],
        progress_callbacks: dict = None,
    ) -> list[Sample]:
        """Generate synthetic samples from content chunks.

        Args:
            chunks: Content chunks to process
            progress_callbacks: Dict with callbacks for progress updates

        Returns:
            List of generated samples
        """
        if progress_callbacks is None:
            progress_callbacks = {}

        # Load checkpoint if available
        if self.checkpoint_manager and not progress_callbacks.get("no_cache"):
            checkpoint_data = self.checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                checkpoint_state, _ = checkpoint_data
                existing_samples = checkpoint_state.get("samples", [])
                if len(existing_samples) >= self.gen_config.target_samples:
                    print(f"Checkpoint complete with {len(existing_samples)} samples")
                    return existing_samples
                elif existing_samples:
                    print(f"Resuming from checkpoint with {len(existing_samples)} samples")

        # Run generation
        all_samples = asyncio.run(self._generate_all_samples_async(chunks, progress_callbacks))

        # Deduplicate
        if self.deduplicator and all_samples:
            print("\nDeduplicating samples...")
            original_count = len(all_samples)
            all_samples = self.deduplicator.deduplicate(all_samples)
            removed = original_count - len(all_samples)
            if removed > 0:
                print(f"Removed {removed} duplicate/similar samples")

        return all_samples

    async def _generate_all_samples_async(
        self,
        chunks: list[Chunk],
        progress_callbacks: dict,
    ) -> list[Sample]:
        """Generate all samples using the planning approach with over-allocation."""
        # Load checkpoint state
        final_samples = []
        rejected_list = []
        failures_list = []
        pending_candidates = []
        current_stage = "start"

        if self.checkpoint_manager and not progress_callbacks.get("no_cache"):
            checkpoint_data = self.checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                checkpoint_state, metadata = checkpoint_data
                final_samples = checkpoint_state.get("samples", [])
                rejected_list = checkpoint_state.get("rejected_samples", [])
                failures_list = checkpoint_state.get("code_failures", [])
                pending_candidates = checkpoint_state.get("pending_candidates", [])
                current_stage = metadata.get("current_stage", "start")

                if pending_candidates:
                    print(
                        f"  Resuming from stage: {current_stage} "
                        f"with {len(pending_candidates)} pending candidates"
                    )

        target = self.gen_config.target_samples
        max_attempts = getattr(self.gen_config, "max_generation_attempts", 3)
        attempt = 0

        while len(final_samples) < target and attempt < max_attempts:
            # Calculate per-type needs based on what we already have
            type_counts, mm_counts = self._count_samples_by_type(final_samples)
            allocation_config = self._build_adjusted_allocation_config(
                type_counts, mm_counts, is_first_attempt=(attempt == 0)
            )

            # Check if we actually need more samples
            total_needed = sum(allocation_config.get_type_target(qt) for qt in QuestionType)
            if total_needed == 0:
                break

            over_allocated = sum(
                allocation_config.get_over_allocated_target(qt) for qt in QuestionType
            )

            print(f"\n[Attempt {attempt+1}/{max_attempts}] Need {total_needed} more samples")
            print(
                f"  Over-allocating to {over_allocated} candidates (factor: {allocation_config.over_allocation_factor}x)"
            )

            # Select chunks for this batch (shuffle for diversity)
            shuffled_chunks = chunks.copy()
            random.shuffle(shuffled_chunks)

            # Stage 1: Generate input candidates
            if current_stage in ("start", "input_generation"):
                print("  Stage 1: Generating input candidates...")

                if progress_callbacks.get("stage_change"):
                    progress_callbacks["stage_change"]("input_generation")

                candidates, metrics = await self._generate_candidates_async(
                    shuffled_chunks, progress_callbacks, allocation_config
                )
                print(f"    Generated {len(candidates)} candidates")
                self._log_allocation_metrics(metrics)

                if self.checkpoint_manager:
                    self._save_substage_checkpoint(
                        final_samples,
                        rejected_list,
                        failures_list,
                        candidates,
                        "candidate_filtering",
                        attempt,
                    )
                current_stage = "candidate_filtering"
            else:
                candidates = pending_candidates

            # Stage 2: Filter candidates
            if current_stage == "candidate_filtering":
                if (
                    self.gen_config.enable_candidate_filtering
                    and self.prompts.candidate_filter_prompt
                ):
                    print("  Stage 2: Filtering candidates...")

                    if progress_callbacks.get("stage_change"):
                        progress_callbacks["stage_change"]("candidate_filtering")

                    candidates = await self._filter_candidates_async(candidates, progress_callbacks)
                    print(f"    {len(candidates)} candidates passed filtering")

                if self.checkpoint_manager:
                    self._save_substage_checkpoint(
                        final_samples,
                        rejected_list,
                        failures_list,
                        candidates,
                        "answer_generation",
                        attempt,
                    )
                current_stage = "answer_generation"

            # Stage 3: Generate answers
            if current_stage == "answer_generation":
                print("  Stage 3: Generating answers...")

                if progress_callbacks.get("stage_change"):
                    progress_callbacks["stage_change"]("answer_generation")

                samples, failures = await self._generate_answers_async(
                    candidates, progress_callbacks
                )
                failures_list.extend(failures)

                if failures and progress_callbacks.get("save_code_failures"):
                    progress_callbacks["save_code_failures"](failures)
                print(f"    Generated {len(samples)} answers, {len(failures)} failed")

                if self.checkpoint_manager:
                    self._save_substage_checkpoint(
                        final_samples + samples,
                        rejected_list,
                        failures_list,
                        [],
                        "quality_curation",
                        attempt,
                    )
                current_stage = "quality_curation"

            # Stage 4: Quality curation
            if current_stage == "quality_curation":
                if self.gen_config.enable_curate_filtering:
                    print("  Stage 4: Quality curation...")

                    if progress_callbacks.get("stage_change"):
                        progress_callbacks["stage_change"]("quality_curation")

                    samples, rejected = await self._curate_samples_async(
                        samples, progress_callbacks
                    )
                    rejected_list.extend(rejected)
                    if progress_callbacks.get("save_rejected"):
                        progress_callbacks["save_rejected"](rejected)
                    print(f"    {len(samples)} passed, {len(rejected)} rejected")

                current_stage = "classification"

            # Stage 5: Post-generation classification
            if current_stage == "classification":
                print("  Stage 5: Classifying samples...")

                if progress_callbacks.get("stage_change"):
                    progress_callbacks["stage_change"]("classification")

                samples = await self._classify_samples_async(samples, progress_callbacks)

            final_samples.extend(samples)
            pending_candidates = []
            current_stage = "start"

            # Save checkpoint
            if self.checkpoint_manager:
                self._save_checkpoint(final_samples, rejected_list, failures_list, attempt)

            print(f"  Progress: {len(final_samples)}/{target} samples")
            attempt += 1

        # Save tracer summary
        if self.tracer:
            summary = self.tracer.get_summary()
            print(
                f"\n  Tracing summary: {summary.get('total_conversations', 0)} conversations, "
                f"{summary.get('success_rate', 0)*100:.1f}% success rate"
            )

        # Trim excess samples to match target distribution
        final_samples = self._trim_to_targets(final_samples)

        await self._cleanup_async()
        return final_samples

    def _log_allocation_metrics(self, metrics: AllocationMetrics) -> None:
        """Log allocation metrics for visibility."""
        print(
            f"    Chunk coverage: {metrics.chunks_used}/{metrics.total_chunks} "
            f"({metrics.chunk_coverage:.1%})"
        )
        print(
            f"    Image coverage: {metrics.images_used}/{metrics.total_images} "
            f"({metrics.image_coverage:.1%})"
        )
        if metrics.avg_chunk_usage > 1.0:
            print(f"    Avg chunk reuse: {metrics.avg_chunk_usage:.2f}x")

    async def _generate_candidates_async(
        self,
        chunks: list[Chunk],
        progress_callbacks: dict,
        allocation_config: AllocationConfig | None = None,
    ) -> tuple[list[InputCandidate], AllocationMetrics]:
        """Generate input candidates from chunks."""
        question_client = self.model_registry.get_llm_client(self.gen_config.question_model)

        planner = InputPlanner(
            llm_client=question_client,
            prompts=self.prompts,
            max_concurrent=self.gen_config.llm_concurrency,
            tracer=self.tracer,
        )

        if allocation_config is None:
            allocation_config = self._build_allocation_config()

        if progress_callbacks.get("set_questions_total"):
            # Use over-allocated total for progress
            over_allocated = sum(
                allocation_config.get_over_allocated_target(qt) for qt in QuestionType
            )
            progress_callbacks["set_questions_total"](over_allocated)

        diversity_weight = getattr(self.gen_config, "diversity_weight", 0.4)
        candidates, allocation_result = await planner.generate_candidates_async(
            chunks,
            allocation_config,
            progress_callback=progress_callbacks.get("questions"),
            diversity_weight=diversity_weight,
        )

        # Filter out invalid candidates
        valid = [c for c in candidates if c.is_valid]
        return valid, allocation_result.metrics

    def _count_samples_by_type(
        self, samples: list[Sample]
    ) -> tuple[dict[QuestionType, int], dict[QuestionType, int]]:
        """Count samples by question type and multimodal status."""
        total_counts = {qt: 0 for qt in QuestionType}
        mm_counts = {qt: 0 for qt in QuestionType}
        for sample in samples:
            try:
                qt = QuestionType(sample.question_type)
                total_counts[qt] += 1
                if sample.image_path:
                    mm_counts[qt] += 1
            except ValueError:
                pass
        return total_counts, mm_counts

    def _build_allocation_config(self) -> AllocationConfig:
        """Build allocation config from generation config."""
        type_configs = {}

        if self.gen_config.type_allocations:
            for type_name, type_cfg in self.gen_config.type_allocations.items():
                try:
                    qt = QuestionType(type_name)
                    type_configs[qt] = AllocTypeConfig(
                        ratio=type_cfg.ratio,
                        multimodal_ratio=type_cfg.multimodal_ratio,
                    )
                except ValueError:
                    pass

        if not type_configs:
            type_configs = {
                QuestionType.QA: AllocTypeConfig(ratio=0.30, multimodal_ratio=0.70),
                QuestionType.CODE_GENERATION: AllocTypeConfig(ratio=0.35, multimodal_ratio=0.30),
                QuestionType.FUNCTION_COMPLETION: AllocTypeConfig(
                    ratio=0.35, multimodal_ratio=0.30
                ),
            }

        over_allocation = getattr(self.gen_config, "over_allocation_factor", 1.8)

        return AllocationConfig(
            target_samples=self.gen_config.target_samples,
            type_configs=type_configs,
            over_allocation_factor=over_allocation,
        )

    def _trim_to_targets(self, samples: list[Sample]) -> list[Sample]:
        """Trim or scale samples to match target distribution.

        Behavior depends on `keep_extra_samples` config:
        - False: Trim excess samples to match exact targets (original behavior)
        - True: Keep all samples, scaling up proportionally to maintain ratios
        """
        base_config = self._build_allocation_config()
        keep_extra = getattr(self.gen_config, "keep_extra_samples", True)

        # Group samples by type and modality
        by_type_mm: dict[QuestionType, list[Sample]] = {qt: [] for qt in QuestionType}
        by_type_text: dict[QuestionType, list[Sample]] = {qt: [] for qt in QuestionType}

        for sample in samples:
            try:
                qt = QuestionType(sample.question_type)
                if sample.image_path:
                    by_type_mm[qt].append(sample)
                else:
                    by_type_text[qt].append(sample)
            except ValueError:
                pass

        # Calculate totals for scaling
        total_available = len(samples)
        target_total = self.gen_config.target_samples

        if keep_extra and total_available > target_total:
            # Scale mode: keep all samples, maintain proportions
            return self._scale_to_proportions(by_type_mm, by_type_text, base_config)
        else:
            # Trim mode: original behavior
            return self._trim_to_exact_targets(by_type_mm, by_type_text, base_config)

    def _scale_to_proportions(
        self,
        by_type_mm: dict[QuestionType, list[Sample]],
        by_type_text: dict[QuestionType, list[Sample]],
        base_config: AllocationConfig,
    ) -> list[Sample]:
        """Keep all samples while maintaining configured ratios as closely as possible.

        Strategy: Scale up targets proportionally based on available samples,
        then trim any type/modality that has excess beyond its scaled target.
        """
        # Calculate actual totals
        total_mm = sum(len(samples) for samples in by_type_mm.values())
        total_text = sum(len(samples) for samples in by_type_text.values())
        total_available = total_mm + total_text

        if total_available == 0:
            return []

        # Calculate scaling factor (how much larger the final dataset will be)
        scale_factor = total_available / self.gen_config.target_samples

        result = []
        kept_stats = {qt: {"mm": 0, "text": 0} for qt in QuestionType}

        for qt in QuestionType:
            # Calculate scaled targets for this type
            base_type_target = base_config.get_type_target(qt)
            base_mm_target = base_config.get_multimodal_target(qt)
            base_text_target = base_type_target - base_mm_target

            # Scale up targets proportionally
            scaled_mm_target = int(base_mm_target * scale_factor)
            scaled_text_target = int(base_text_target * scale_factor)

            mm_samples = by_type_mm[qt]
            text_samples = by_type_text[qt]

            # Shuffle for diversity
            random.shuffle(mm_samples)
            random.shuffle(text_samples)

            # Keep up to scaled target (this allows keeping extras while respecting proportions)
            kept_mm = mm_samples[:scaled_mm_target] if scaled_mm_target > 0 else []
            kept_text = text_samples[:scaled_text_target] if scaled_text_target > 0 else []

            # If one modality has excess capacity and the other has shortfall, redistribute
            mm_shortfall = scaled_mm_target - len(mm_samples)
            text_shortfall = scaled_text_target - len(text_samples)

            if mm_shortfall > 0 and len(text_samples) > scaled_text_target:
                # Fill multimodal shortfall with extra text samples
                extra_text = min(mm_shortfall, len(text_samples) - scaled_text_target)
                kept_text = text_samples[: scaled_text_target + extra_text]
            elif text_shortfall > 0 and len(mm_samples) > scaled_mm_target:
                # Fill text shortfall with extra multimodal samples
                extra_mm = min(text_shortfall, len(mm_samples) - scaled_mm_target)
                kept_mm = mm_samples[: scaled_mm_target + extra_mm]

            result.extend(kept_mm)
            result.extend(kept_text)
            kept_stats[qt]["mm"] = len(kept_mm)
            kept_stats[qt]["text"] = len(kept_text)

        # Log the scaling result
        original_target = self.gen_config.target_samples
        final_count = len(result)
        extra_kept = final_count - original_target

        if extra_kept > 0:
            print(
                f"\n  Kept {extra_kept} extra samples (total: {final_count}, target was: {original_target})"
            )
            print(f"  Scale factor: {scale_factor:.2f}x - Proportions maintained")
        elif extra_kept < 0:
            print(f"\n  Final count: {final_count} (below target {original_target})")

        return result

    def _trim_to_exact_targets(
        self,
        by_type_mm: dict[QuestionType, list[Sample]],
        by_type_text: dict[QuestionType, list[Sample]],
        base_config: AllocationConfig,
    ) -> list[Sample]:
        """Trim samples to match exact target counts (original behavior)."""
        trimmed = []
        total_trimmed = 0

        for qt in QuestionType:
            target_total = base_config.get_type_target(qt)
            target_mm = base_config.get_multimodal_target(qt)
            target_text = target_total - target_mm

            mm_samples = by_type_mm[qt]
            text_samples = by_type_text[qt]

            # Shuffle for diversity in selection
            random.shuffle(mm_samples)
            random.shuffle(text_samples)

            # Select up to target for each modality
            kept_mm = mm_samples[:target_mm]
            kept_text = text_samples[:target_text]

            # If we don't have enough of one type, fill from the other
            if len(kept_mm) < target_mm and len(text_samples) > target_text:
                shortfall = target_mm - len(kept_mm)
                kept_text = text_samples[: target_text + shortfall]
            elif len(kept_text) < target_text and len(mm_samples) > target_mm:
                shortfall = target_text - len(kept_text)
                kept_mm = mm_samples[: target_mm + shortfall]

            # Ensure we don't exceed total target
            total_kept = len(kept_mm) + len(kept_text)
            if total_kept > target_total:
                if len(kept_text) > target_text:
                    kept_text = kept_text[:target_text]

            trimmed.extend(kept_mm)
            trimmed.extend(kept_text)

            original_count = len(mm_samples) + len(text_samples)
            total_kept = len(kept_mm) + len(kept_text)
            total_trimmed += max(0, original_count - total_kept)

        if total_trimmed > 0:
            print(f"\n  Trimmed {total_trimmed} excess samples to match target distribution")

        return trimmed

    def _build_adjusted_allocation_config(
        self,
        current_counts: dict[QuestionType, int],
        current_mm_counts: dict[QuestionType, int],
        is_first_attempt: bool = False,
    ) -> AllocationConfig:
        """Build allocation config adjusted for samples already generated."""
        base_config = self._build_allocation_config()

        # Calculate remaining needs per type
        remaining = {}
        remaining_mm = {}
        total_remaining = 0

        for qt in QuestionType:
            target_total = base_config.get_type_target(qt)
            target_mm = base_config.get_multimodal_target(qt)

            current = current_counts.get(qt, 0)
            current_mm = current_mm_counts.get(qt, 0)

            needed = max(0, target_total - current)
            needed_mm = max(0, target_mm - current_mm)

            remaining[qt] = needed
            remaining_mm[qt] = needed_mm
            total_remaining += needed

        if total_remaining == 0:
            return AllocationConfig(
                target_samples=0,
                type_configs={
                    qt: AllocTypeConfig(ratio=0.0, multimodal_ratio=0.0) for qt in QuestionType
                },
                over_allocation_factor=1.0,
            )

        # Build new config with adjusted ratios
        new_type_configs = {}
        for qt in QuestionType:
            needed = remaining[qt]
            if needed > 0:
                ratio = needed / total_remaining
                needed_mm = remaining_mm[qt]
                mm_ratio = min(1.0, needed_mm / needed) if needed_mm > 0 else 0.0

                new_type_configs[qt] = AllocTypeConfig(
                    ratio=ratio,
                    multimodal_ratio=mm_ratio,
                )
            else:
                new_type_configs[qt] = AllocTypeConfig(ratio=0.0, multimodal_ratio=0.0)

        # Log adjusted targets
        print("    Adjusted targets (remaining needs):")
        for qt in QuestionType:
            needed = remaining[qt]
            if needed > 0:
                needed_mm = remaining_mm[qt]
                mm_ratio = new_type_configs[qt].multimodal_ratio
                print(
                    f"      {qt.value}: {needed} needed ({needed_mm} mm, mm_ratio={mm_ratio:.0%})"
                )

        # Use full over-allocation on first attempt, reduced on retries
        over_factor = base_config.over_allocation_factor if is_first_attempt else 1.5

        return AllocationConfig(
            target_samples=total_remaining,
            type_configs=new_type_configs,
            over_allocation_factor=over_factor,
        )

    async def _filter_candidates_async(
        self,
        candidates: list[InputCandidate],
        progress_callbacks: dict,
    ) -> list[InputCandidate]:
        """Filter candidates for quality."""
        if not candidates:
            return []

        question_client = self.model_registry.get_llm_client(self.gen_config.question_model)

        planner = InputPlanner(
            llm_client=question_client,
            prompts=self.prompts,
            max_concurrent=self.gen_config.llm_concurrency,
            tracer=self.tracer,
        )

        if progress_callbacks.get("set_filter_total"):
            progress_callbacks["set_filter_total"](len(candidates))

        filtered = await planner.filter_candidates_async(
            candidates,
            self.prompts.candidate_filter_prompt,
            self.prompts.candidate_filter_system,
            progress_callback=progress_callbacks.get("filter"),
        )

        return filtered

    async def _generate_answers_async(
        self,
        candidates: list[InputCandidate],
        progress_callbacks: dict,
    ) -> tuple[list[Sample], list[dict]]:
        """Generate and validate answers for candidates."""
        if not candidates:
            return [], []

        answer_client = self.model_registry.get_llm_client(self.gen_config.answer_model)

        sessions = []
        for candidate in candidates:
            use_image = candidate.is_multimodal
            system_prompt = self.prompts.get_answer_system_prompt(use_image=use_image)

            answer_prompt = self.prompts.get_answer_prompt(
                question_type=candidate.question_type,
                question=candidate.question,
                context=candidate.context,
                test_code=candidate.test_code,
            )

            session = AnswerSession(
                llm_client=answer_client,
                system_prompt=system_prompt,
                answer_prompt=answer_prompt,
                correction_prompt=self.prompts.answer_correction_prompt,
                candidate=candidate,
                max_iterations=self.gen_config.code_verification_max_iterations,
                timeout_seconds=self.gen_config.test_validation_timeout,
                tracer=self.tracer,
            )
            sessions.append(session)

        if progress_callbacks.get("set_answers_total"):
            progress_callbacks["set_answers_total"](len(sessions))

        processor = AnswerBatchProcessor(
            llm_client=answer_client,
            max_concurrent=self.gen_config.llm_concurrency,
            tracer=self.tracer,
        )

        results = await processor.process_sessions_async(
            sessions,
            progress_callback=progress_callbacks.get("answers"),
        )

        samples = []
        failures = []

        for candidate, result in zip(candidates, results):
            if result.passed and result.answer:
                sample = self._create_sample(candidate, result.answer)
                samples.append(sample)
            elif result.answer:
                if candidate.question_type == QuestionType.QA:
                    sample = self._create_sample(candidate, result.answer)
                    samples.append(sample)
                else:
                    failures.append(
                        {
                            "question": candidate.question,
                            "answer": result.answer,
                            "test_code": candidate.test_code,
                            "entry_point": candidate.entry_point,
                            "question_type": candidate.question_type.value,
                            "error": "Failed validation",
                            "error_history": result.error_history,
                            "iterations": result.iterations_used,
                        }
                    )
            else:
                failures.append(
                    {
                        "question": candidate.question,
                        "answer": "",
                        "question_type": candidate.question_type.value,
                        "error": "Empty answer",
                    }
                )

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
            category="",
            question_type=candidate.question_type.value,
            test_code=candidate.test_code,
            entry_point=candidate.entry_point,
            image_path=candidate.target_image.resolved_path if candidate.target_image else None,
            source_path=str(candidate.chunk.source_path),
            metadata=metadata,
        )

    async def _curate_samples_async(
        self,
        samples: list[Sample],
        progress_callbacks: dict,
    ) -> tuple[list[Sample], list[dict]]:
        """Quality curation for samples."""
        if not samples:
            return [], []

        from synthetic_data.models import Message

        curation_inputs = []
        for sample in samples:
            context_preview = ""
            if sample.metadata:
                context_preview = sample.metadata.get("context_preview", "")
            if not context_preview and hasattr(sample, "context"):
                context_preview = sample.context[:800] if sample.context else ""
            if not context_preview:
                context_preview = "N/A"

            image_description = "N/A"
            if sample.image_path and sample.metadata:
                image_description = sample.metadata.get("image_transcription", "N/A")

            user_prompt = self.prompts.sample_curation.format(
                question=sample.question,
                answer=sample.answer,
                question_type=sample.question_type,
                has_image="yes" if sample.image_path else "no",
                has_test="yes" if sample.test_code else "no",
                context_preview=context_preview[:800] if context_preview else "N/A",
                image_description=image_description[:500] if image_description else "N/A",
            )

            curation_inputs.append(
                [
                    Message(role="system", content=self.prompts.sample_curation_system),
                    Message(role="user", content=user_prompt),
                ]
            )

        if progress_callbacks.get("set_curation_total"):
            progress_callbacks["set_curation_total"](len(curation_inputs))

        curate_client = self.model_registry.get_llm_client(self.gen_config.curate_model)
        responses = await curate_client.generate_batch_async(
            curation_inputs,
            max_concurrent=self.gen_config.llm_concurrency,
            temperature=0.1,
            progress_callback=progress_callbacks.get("curation"),
        )

        validated = []
        rejected = []

        for sample, response in zip(samples, responses):
            decision, reason = self._parse_curation_response(response)

            if decision == "PASS":
                validated.append(sample)
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

        return validated, rejected

    async def _classify_samples_async(
        self,
        samples: list[Sample],
        progress_callbacks: dict,
    ) -> list[Sample]:
        """Classify samples into categories."""
        if not samples:
            return []

        curate_client = self.model_registry.get_llm_client(self.gen_config.curate_model)
        classifier = CategoryClassifier(self.config.categories, curate_client)

        if progress_callbacks.get("set_classification_total"):
            progress_callbacks["set_classification_total"](len(samples))

        categories = await classifier.classify_samples_async(
            samples,
            self.prompts.category_classification,
            self.prompts.category_classification_system,
            max_concurrent=self.gen_config.llm_concurrency,
            progress_callback=progress_callbacks.get("classification"),
        )

        for sample, category in zip(samples, categories):
            sample.category = category

        return samples

    def _parse_curation_response(self, response: str) -> tuple[str, str]:
        """Parse curation response."""
        response_upper = response.strip().upper()

        if "PASS" in response_upper or response_upper.startswith("YES"):
            return "PASS", ""

        reason = "Quality check failed"
        lines = response.split("\n")
        for line in lines:
            if "REASON" in line.upper() and ":" in line:
                reason = line.split(":", 1)[1].strip()
                break

        return "REJECT", reason

    def _save_checkpoint(
        self,
        samples: list[Sample],
        rejected: list[dict],
        failures: list[dict],
        attempt: int,
    ):
        """Save checkpoint state."""
        checkpoint_state = {
            "samples": samples,
            "rejected_samples": rejected,
            "code_failures": failures,
            "pending_candidates": [],
        }
        metadata = {
            "generated_count": len(samples),
            "target_samples": self.gen_config.target_samples,
            "attempt": attempt + 1,
            "current_stage": "complete",
        }
        self.checkpoint_manager.save_checkpoint(checkpoint_state, metadata)

    def _save_substage_checkpoint(
        self,
        samples: list[Sample],
        rejected: list[dict],
        failures: list[dict],
        pending_candidates: list[InputCandidate],
        next_stage: str,
        attempt: int,
    ):
        """Save checkpoint at substage level for resumability."""
        serialized_candidates = []
        for c in pending_candidates:
            serialized_candidates.append(
                {
                    "chunk_id": c.chunk.chunk_id,
                    "question": c.question,
                    "question_type": c.question_type.value,
                    "test_code": c.test_code,
                    "entry_point": c.entry_point,
                    "context": c.context,
                    "is_multimodal": c.is_multimodal,
                    "target_image_id": c.target_image.image_id if c.target_image else None,
                }
            )

        checkpoint_state = {
            "samples": samples,
            "rejected_samples": rejected,
            "code_failures": failures,
            "pending_candidates": serialized_candidates,
        }
        metadata = {
            "generated_count": len(samples),
            "target_samples": self.gen_config.target_samples,
            "attempt": attempt + 1,
            "current_stage": next_stage,
            "pending_count": len(pending_candidates),
        }
        self.checkpoint_manager.save_checkpoint(checkpoint_state, metadata)

    async def _cleanup_async(self):
        """Close all async HTTP clients."""
        clients_to_close = []

        try:
            clients_to_close.append(
                self.model_registry.get_llm_client(self.gen_config.question_model)
            )
        except (AttributeError, KeyError, ValueError):
            pass

        try:
            clients_to_close.append(
                self.model_registry.get_llm_client(self.gen_config.answer_model)
            )
        except (AttributeError, KeyError, ValueError):
            pass

        try:
            clients_to_close.append(
                self.model_registry.get_llm_client(self.gen_config.curate_model)
            )
        except (AttributeError, KeyError, ValueError):
            pass

        for client in clients_to_close:
            try:
                await client.aclose()
            except Exception:
                pass
