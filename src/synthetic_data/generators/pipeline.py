"""Generation pipeline using input planning and post-generation classification."""

import asyncio
import random
from pathlib import Path
from typing import Optional

from synthetic_data.config import PipelineConfig, QuestionType
from synthetic_data.extractors.chunker import Chunk
from synthetic_data.models import ModelRegistry, Sample
from synthetic_data.utils import Deduplicator, CheckpointManager, GenerationTracer
from synthetic_data.generators.category import CategoryClassifier
from synthetic_data.generators.planner import InputPlanner, InputCandidate
from synthetic_data.generators.prompts import PromptSet
from synthetic_data.generators.sessions import AnswerSession, AnswerBatchProcessor


class GenerationPipeline:
    """Pipeline for generating synthetic dataset samples.

    The pipeline uses:
    1. Input Planning: Generate k candidate inputs per chunk
    2. Candidate Filtering: Filter candidates for quality
    3. Answer Generation: Generate and validate answers
    4. Quality Curation: Final quality check
    5. Post-Generation Classification: Classify samples by content
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

        # Initialize tracer for prompt/response logging
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
            all_samples = self.deduplicator.deduplicate(all_samples)

        return all_samples

    async def _generate_all_samples_async(
        self,
        chunks: list[Chunk],
        progress_callbacks: dict,
    ) -> list[Sample]:
        """Generate all samples using the planning approach."""
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
                        f"  Resuming from stage: {current_stage} with {len(pending_candidates)} pending candidates"
                    )

        target = self.gen_config.target_samples
        max_attempts = 5
        attempt = 0

        while len(final_samples) < target and attempt < max_attempts:
            needed = target - len(final_samples)
            print(f"\n[Attempt {attempt+1}/{max_attempts}] Need {needed} more samples")

            # Select chunks for this batch
            batch_size = min(needed * 2, len(chunks))
            batch_chunks = random.sample(chunks, min(batch_size, len(chunks)))

            # Stage 1: Generate input candidates
            if current_stage in ("start", "input_generation"):
                print("  Stage 1: Generating input candidates...")

                # Notify stage change
                if progress_callbacks.get("stage_change"):
                    progress_callbacks["stage_change"]("input_generation")

                candidates = await self._generate_candidates_async(batch_chunks, progress_callbacks)
                print(f"    Generated {len(candidates)} candidates")

                # Checkpoint after candidate generation
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

                # Checkpoint after filtering
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
                print(f"    Generated {len(samples)} answers, {len(failures)} failed")

                # Checkpoint after answer generation
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
            pending_candidates = []  # Clear pending for next iteration
            current_stage = "start"  # Reset for next attempt

            # Save failures callback
            if failures_list and progress_callbacks.get("save_code_failures"):
                progress_callbacks["save_code_failures"](failures_list)

            # Save checkpoint
            if self.checkpoint_manager:
                self._save_checkpoint(final_samples, rejected_list, failures_list, attempt)

            print(f"  Progress: {len(final_samples)}/{target} samples")
            attempt += 1

        # Save tracer summary if available
        if self.tracer:
            summary = self.tracer.get_summary()
            print(
                f"\n  Tracing summary: {summary.get('total_conversations', 0)} conversations, "
                f"{summary.get('success_rate', 0)*100:.1f}% success rate"
            )

        await self._cleanup_async()
        return final_samples

    async def _generate_candidates_async(
        self,
        chunks: list[Chunk],
        progress_callbacks: dict,
    ) -> list[InputCandidate]:
        """Generate input candidates from chunks."""
        question_client = self.model_registry.get_llm_client(self.gen_config.question_model)

        planner = InputPlanner(
            llm_client=question_client,
            prompts=self.prompts,
            candidates_per_chunk=self.gen_config.candidates_per_chunk,
            max_concurrent=self.gen_config.llm_concurrency,
            tracer=self.tracer,
        )

        if progress_callbacks.get("set_questions_total"):
            estimated = len(chunks) * self.gen_config.candidates_per_chunk
            progress_callbacks["set_questions_total"](estimated)

        candidates = await planner.generate_candidates_async(
            chunks,
            self.gen_config.question_type_weights,
            multimodal_ratio=self.gen_config.multimodal_ratio,
            progress_callback=progress_callbacks.get("questions"),
        )

        # Filter out invalid candidates
        valid = [c for c in candidates if c.is_valid]
        return valid

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

        # Create answer sessions
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

        # Process sessions
        processor = AnswerBatchProcessor(
            llm_client=answer_client,
            max_concurrent=self.gen_config.llm_concurrency,
            tracer=self.tracer,
        )

        results = await processor.process_sessions_async(
            sessions,
            progress_callback=progress_callbacks.get("answers"),
        )

        # Build samples
        samples = []
        failures = []

        for candidate, result in zip(candidates, results):
            if result.passed and result.answer:
                sample = self._create_sample(candidate, result.answer)
                samples.append(sample)
            elif result.answer:
                # QA without code can still be valid
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
        return Sample(
            question=candidate.question,
            answer=answer,
            category="",  # Will be set by classification
            question_type=candidate.question_type.value,
            test_code=candidate.test_code,
            entry_point=candidate.entry_point,
            image_path=candidate.target_image.resolved_path if candidate.target_image else None,
            source_path=str(candidate.chunk.source_path),
            metadata=candidate.chunk.metadata,
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
            user_prompt = self.prompts.sample_curation.format(
                question=sample.question,
                answer=sample.answer,
                question_type=sample.question_type,
                has_image="yes" if sample.image_path else "no",
                has_test="yes" if sample.test_code else "no",
                test_code=sample.test_code or "N/A",
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
        """Save checkpoint at substage level for resumability.

        Args:
            samples: Samples generated so far
            rejected: Rejected samples so far
            failures: Code failures so far
            pending_candidates: Candidates waiting for next stage
            next_stage: The stage to resume at
            attempt: Current attempt number
        """
        # Serialize candidates for checkpoint (they may contain non-serializable objects)
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
