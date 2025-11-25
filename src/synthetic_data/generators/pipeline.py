"""Main synthetic data generation pipeline with async batch processing.

Generates high-quality multimodal samples for quantum computing VLM fine-tuning.

Three input types:
- function_completion: Prompt with imports + function signature + docstring
- code_generation: Natural language task (Qiskit HumanEval Hard format)
- qa: Theory/concepts (no unit test required)

Code types include unit tests for validation. Answers must pass their tests.
"""

import asyncio
import random
import re
from typing import Optional

from synthetic_data.config import PipelineConfig, QuestionType
from synthetic_data.extractors.chunker import Chunk
from synthetic_data.models import Message, ModelRegistry, Sample
from synthetic_data.utils import Deduplicator, CheckpointManager
from synthetic_data.utils.code_verifier import CodeVerifier
from synthetic_data.utils.test_generator import TestGenerator, CodeWithTestValidator
from synthetic_data.generators.category import CategoryManager
from synthetic_data.generators.prompts import (
    PromptSet,
    build_context,
    extract_entry_point_from_prompt,
)


class GenerationPipeline:
    """Pipeline for generating synthetic dataset samples with unit tests."""

    def __init__(
        self,
        config: PipelineConfig,
        model_registry: ModelRegistry,
        category_manager: CategoryManager,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        """Initialize generation pipeline.

        Args:
            config: Pipeline configuration
            model_registry: Model registry for LLM/VLM access
            category_manager: Category manager
            checkpoint_manager: Optional checkpoint manager for resuming
        """
        self.config = config
        self.gen_config = config.generation
        self.model_registry = model_registry
        self.category_manager = category_manager
        self.checkpoint_manager = checkpoint_manager

        # Create prompt set from config
        self.prompts = PromptSet(
            function_completion_prompt=config.prompts.function_completion_prompt,
            code_generation_prompt=config.prompts.code_generation_prompt,
            qa_prompt=config.prompts.qa_prompt,
            answer_with_test_prompt=config.prompts.answer_with_test_prompt,
            answer_without_test_prompt=config.prompts.answer_without_test_prompt,
            question_generation_system=config.prompts.question_generation_system,
            answer_generation_system=config.prompts.answer_generation_system,
            test_generation_system=config.prompts.test_generation_system,
            content_quality_check=config.prompts.content_quality_check,
            content_filter_system=config.prompts.content_filter_system,
            image_quality_check=config.prompts.image_quality_check,
            image_filter_system=config.prompts.image_filter_system,
            image_transcription=config.prompts.image_transcription,
            image_transcription_system=config.prompts.image_transcription_system,
            category_classification=config.prompts.category_classification,
            category_classification_system=config.prompts.category_classification_system,
            sample_curation=config.prompts.sample_curation,
            sample_curation_system=config.prompts.sample_curation_system,
        )

        random.seed(config.seed)

        # Initialize deduplicator if enabled
        self.deduplicator = None
        if self.gen_config.enable_deduplication:
            self.deduplicator = Deduplicator(self.gen_config.similarity_threshold)

        # Initialize code verifier (for QA samples with code)
        self.code_verifier = None
        if self.gen_config.enable_code_verification:
            correction_client = model_registry.get_llm_client(self.gen_config.answer_model)
            self.code_verifier = CodeVerifier(
                llm_client=correction_client,
                max_iterations=self.gen_config.code_verification_max_iterations,
                timeout_seconds=self.gen_config.code_verification_timeout,
            )

        # Initialize test generator and validator (for code types)
        answer_client = model_registry.get_llm_client(self.gen_config.answer_model)
        self.test_generator = TestGenerator(
            llm_client=answer_client,
            timeout_seconds=self.gen_config.test_validation_timeout,
        )
        self.test_validator = CodeWithTestValidator(
            llm_client=answer_client,
            max_correction_attempts=self.gen_config.code_verification_max_iterations,
            timeout_seconds=self.gen_config.test_validation_timeout,
        )

    def generate_samples(
        self,
        chunks_by_category: dict[str, list[Chunk]],
        progress_callbacks: dict = None,
    ) -> list[Sample]:
        """Generate synthetic samples from content chunks.

        Args:
            chunks_by_category: Chunks organized by category
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
                    return existing_samples[: self.gen_config.target_samples]
                elif existing_samples:
                    print(f"Resuming from checkpoint with {len(existing_samples)} samples")

        # Run generation
        all_samples = asyncio.run(
            self._generate_all_samples_async(chunks_by_category, progress_callbacks)
        )

        # Deduplicate if enabled
        if self.deduplicator:
            print("\nDeduplicating samples...")
            all_samples = self.deduplicator.deduplicate(all_samples)

        return all_samples

    async def _generate_all_samples_async(
        self,
        chunks_by_category: dict[str, list[Chunk]],
        progress_callbacks: dict,
    ) -> list[Sample]:
        """Generate all samples with batching and retry loop."""
        # Load checkpoint state
        final_samples = []
        rejected_samples_list = []
        code_failures_list = []

        if self.checkpoint_manager and not progress_callbacks.get("no_cache"):
            checkpoint_data = self.checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                checkpoint_state, _ = checkpoint_data
                final_samples = checkpoint_state.get("samples", [])
                rejected_samples_list = checkpoint_state.get("rejected_samples", [])
                code_failures_list = checkpoint_state.get("code_failures", [])

        target_samples = self.gen_config.target_samples
        max_attempts = 5
        attempt = 0

        # Build chunk pools per category
        chunk_pools = self._build_chunk_pools(chunks_by_category)

        while len(final_samples) < target_samples and attempt < max_attempts:
            needed = target_samples - len(final_samples)
            batch_target = int(needed * 1.3) + 10  # Buffer for rejections

            print(
                f"\n[Attempt {attempt+1}/{max_attempts}] Generating ~{batch_target} samples (Need: {needed})"
            )

            # Prepare sample specifications
            sample_specs = self._prepare_sample_specs(batch_target, chunks_by_category, chunk_pools)

            if not sample_specs:
                print("No samples to generate.")
                break

            print(f"Generating {len(sample_specs)} samples...")

            # Step 1: Generate questions
            print("  Step 1: Generating questions...")
            questions_data = await self._batch_generate_questions_async(
                sample_specs, progress_callbacks
            )

            # Step 2: Generate tests for code types
            print("  Step 2: Generating tests for code types...")
            tests_data = await self._batch_generate_tests_async(questions_data, progress_callbacks)

            # Step 3: Generate answers
            print("  Step 3: Generating answers...")
            answers_data = await self._batch_generate_answers_async(tests_data, progress_callbacks)

            # Step 4: Validate code types against tests
            print("  Step 4: Validating code samples...")
            validated_samples, validation_failures = await self._validate_code_samples_async(
                answers_data, progress_callbacks
            )
            code_failures_list.extend(validation_failures)

            # Step 5: Verify QA samples with code (syntax/execution only)
            if self.code_verifier:
                print("  Step 5: Verifying QA samples...")
                verified_samples, verify_failures = await self._verify_qa_samples_async(
                    validated_samples, progress_callbacks
                )
                code_failures_list.extend(verify_failures)
            else:
                verified_samples = validated_samples

            # Step 6: Quality curation
            if self.gen_config.enable_curate_filtering:
                print("  Step 6: Quality curation...")
                curated_samples, rejected = await self._curate_samples_async(
                    verified_samples, progress_callbacks
                )
                rejected_samples_list.extend(rejected)
                if progress_callbacks.get("save_rejected"):
                    progress_callbacks["save_rejected"](rejected)
                final_samples.extend(curated_samples)
            else:
                final_samples.extend(verified_samples)

            # Save code failures
            if code_failures_list and progress_callbacks.get("save_code_failures"):
                progress_callbacks["save_code_failures"](code_failures_list)

            # Save checkpoint
            if self.checkpoint_manager:
                self._save_checkpoint(
                    final_samples, rejected_samples_list, code_failures_list, attempt
                )

            print(f"Progress: {len(final_samples)}/{target_samples} samples")
            attempt += 1

            if len(final_samples) >= target_samples:
                final_samples = final_samples[:target_samples]
                break

        await self._cleanup_async_clients()
        return final_samples

    def _build_chunk_pools(self, chunks_by_category: dict[str, list[Chunk]]) -> dict[str, dict]:
        """Build multimodal and text pools per category."""
        pools = {}
        for cat, chunks in chunks_by_category.items():
            mm_pool = []
            for c in chunks:
                if c.images:
                    for i, img in enumerate(c.images):
                        if img.transcription:
                            mm_pool.append((c, i))
            pools[cat] = {"mm": mm_pool, "txt": chunks}
        return pools

    def _prepare_sample_specs(
        self,
        batch_target: int,
        chunks_by_category: dict[str, list[Chunk]],
        chunk_pools: dict,
    ) -> list[dict]:
        """Prepare sample specifications for generation."""
        target_distribution = self.category_manager.get_target_distribution(
            batch_target, chunks_by_category
        )

        all_specs = []

        for category, cat_target in target_distribution.items():
            if cat_target == 0:
                continue

            pools = chunk_pools.get(category)
            if not pools:
                continue

            # Determine multimodal vs text split
            n_mm = int(cat_target * self.gen_config.multimodal_ratio)
            n_txt = cat_target - n_mm

            if not pools["mm"]:
                n_txt += n_mm
                n_mm = 0

            # Select items
            mm_items = random.choices(pools["mm"], k=n_mm) if n_mm > 0 and pools["mm"] else []
            txt_items = random.choices(pools["txt"], k=n_txt) if n_txt > 0 and pools["txt"] else []

            # Create batch items
            batch_items = []
            for c, img_idx in mm_items:
                batch_items.append(
                    {
                        "chunk": c,
                        "use_image": True,
                        "image_index": img_idx,
                        "category": category,
                    }
                )
            for c in txt_items:
                batch_items.append(
                    {
                        "chunk": c,
                        "use_image": False,
                        "image_index": None,
                        "category": category,
                    }
                )

            # Assign question types
            type_dist = self._get_question_type_distribution(len(batch_items))
            types_list = []
            for qt, count in type_dist.items():
                types_list.extend([qt] * count)
            random.shuffle(types_list)

            for item, qtype in zip(batch_items, types_list):
                item["question_type"] = qtype
                all_specs.append(item)

        return all_specs

    async def _batch_generate_questions_async(
        self,
        sample_specs: list[dict],
        progress_callbacks: dict,
    ) -> list[dict]:
        """Generate questions for all sample specs."""
        question_inputs = []
        valid_specs = []

        for spec in sample_specs:
            chunk = spec["chunk"]
            question_type = spec["question_type"]
            use_image = spec.get("use_image", False)
            image_index = spec.get("image_index")

            # Get target image
            target_image = None
            if use_image and image_index is not None and 0 <= image_index < len(chunk.images):
                img = chunk.images[image_index]
                if img.transcription:
                    target_image = img

            if use_image and not target_image:
                use_image = False

            # Build context
            context = build_context(
                chunk_text=chunk.text,
                previous_text=chunk.previous_chunk_text,
                next_text=chunk.next_chunk_text,
                code_context=chunk.extended_code_context,
                image_description=target_image.transcription if target_image else "",
                max_length=self.gen_config.max_context_length,
            )

            # Get prompt
            prompt_template = self.prompts.get_question_prompt(question_type)
            user_prompt = prompt_template.format(context=context)
            system_prompt = self.prompts.get_question_system_prompt(use_image=use_image)

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]
            question_inputs.append(messages)

            valid_specs.append(
                {
                    **spec,
                    "use_image": use_image,
                    "target_image": target_image,
                    "context": context,
                }
            )

        # Generate questions
        if progress_callbacks.get("set_questions_total"):
            progress_callbacks["set_questions_total"](len(question_inputs))

        question_client = self.model_registry.get_llm_client(self.gen_config.question_model)
        questions = await question_client.generate_batch_async(
            question_inputs,
            max_concurrent=self.gen_config.llm_concurrency,
            progress_callback=progress_callbacks.get("questions"),
        )

        # Combine with specs
        result = []
        for spec, question in zip(valid_specs, questions):
            question = question.strip()
            if question:
                spec["question"] = question
                result.append(spec)

        return result

    async def _batch_generate_tests_async(
        self,
        questions_data: list[dict],
        _progress_callbacks: dict,
    ) -> list[dict]:
        """Generate unit tests for code types."""
        code_items = []
        for item in questions_data:
            if item["question_type"] in (
                QuestionType.FUNCTION_COMPLETION,
                QuestionType.CODE_GENERATION,
            ):
                code_items.append(item)

        if not code_items:
            return questions_data

        # Extract entry points from questions
        test_tasks = []
        for item in code_items:
            entry_point = extract_entry_point_from_prompt(item["question"])
            if not entry_point:
                # Try to find function name in "function named `name`" pattern
                match = re.search(r"function\s+named\s+[`'\"]?(\w+)[`'\"]?", item["question"], re.I)
                if match:
                    entry_point = match.group(1)

            item["entry_point"] = entry_point or "solution"

            # For test generation, we need a reference solution first
            # We'll generate tests based on the question/task description
            test_tasks.append(
                {
                    "task_description": item["question"],
                    "reference_code": "",  # Will be filled after first answer attempt
                    "entry_point": item["entry_point"],
                }
            )

        # Generate tests using the test generator
        tests = await self.test_generator.generate_batch_tests_async(
            test_tasks,
            max_concurrent=self.gen_config.llm_concurrency,
        )

        # Add test results to items
        for item, test_result in zip(code_items, tests):
            if test_result.is_valid:
                item["test_code"] = test_result.test_code
            else:
                item["test_code"] = None
                item["test_error"] = test_result.validation_error

        return questions_data

    async def _batch_generate_answers_async(
        self,
        tests_data: list[dict],
        progress_callbacks: dict,
    ) -> list[dict]:
        """Generate answers for all items."""
        answer_inputs = []
        valid_items = []

        for item in tests_data:
            question_type = item["question_type"]
            question = item["question"]
            context = item["context"]
            test_code = item.get("test_code")
            use_image = item.get("use_image", False)

            # Get appropriate answer prompt
            if question_type in (QuestionType.FUNCTION_COMPLETION, QuestionType.CODE_GENERATION):
                if test_code:
                    user_prompt = self.prompts.answer_with_test_prompt.format(
                        question=question,
                        context=context,
                        test_code=test_code,
                    )
                else:
                    # No test generated, use regular prompt
                    user_prompt = self.prompts.answer_without_test_prompt.format(
                        question=question,
                        context=context,
                    )
            else:
                user_prompt = self.prompts.answer_without_test_prompt.format(
                    question=question,
                    context=context,
                )

            system_prompt = self.prompts.get_answer_system_prompt(use_image=use_image)

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]
            answer_inputs.append(messages)
            valid_items.append(item)

        # Generate answers
        if progress_callbacks.get("set_answers_total"):
            progress_callbacks["set_answers_total"](len(answer_inputs))

        answer_client = self.model_registry.get_llm_client(self.gen_config.answer_model)
        answers = await answer_client.generate_batch_async(
            answer_inputs,
            max_concurrent=self.gen_config.llm_concurrency,
            progress_callback=progress_callbacks.get("answers"),
        )

        # Combine with items
        for item, answer in zip(valid_items, answers):
            item["answer"] = answer.strip()

        return [item for item in valid_items if item.get("answer")]

    async def _validate_code_samples_async(
        self,
        answers_data: list[dict],
        _progress_callbacks: dict,
    ) -> tuple[list[Sample], list[dict]]:
        """Validate code samples against their tests."""
        samples = []
        failures = []

        for item in answers_data:
            question_type = item["question_type"]
            test_code = item.get("test_code")
            entry_point = item.get("entry_point")
            answer = item["answer"]

            if question_type in (QuestionType.FUNCTION_COMPLETION, QuestionType.CODE_GENERATION):
                if test_code and entry_point:
                    # Validate against test
                    passed, corrected_answer, attempts = (
                        await self.test_validator.validate_and_correct_async(
                            answer, test_code, entry_point, item["question"]
                        )
                    )

                    if passed:
                        sample = self._create_sample(item, corrected_answer, test_code, entry_point)
                        samples.append(sample)
                    else:
                        failures.append(
                            {
                                "question": item["question"],
                                "answer": answer,
                                "test_code": test_code,
                                "entry_point": entry_point,
                                "category": item["category"],
                                "question_type": question_type.value,
                                "error": "Failed to pass unit test after corrections",
                                "attempts": attempts,
                            }
                        )
                else:
                    # No test, just create sample (will be verified by code verifier if enabled)
                    sample = self._create_sample(item, answer, None, entry_point)
                    samples.append(sample)
            else:
                # QA type - no test validation
                sample = self._create_sample(item, answer, None, None)
                samples.append(sample)

        return samples, failures

    async def _verify_qa_samples_async(
        self,
        samples: list[Sample],
        _progress_callbacks: dict,
    ) -> tuple[list[Sample], list[dict]]:
        """Verify QA samples with code (syntax/execution only)."""
        verified = []
        failures = []

        qa_samples = [s for s in samples if s.question_type == "qa"]
        other_samples = [s for s in samples if s.question_type != "qa"]

        for sample in qa_samples:
            result = await self.code_verifier.verify_and_correct_sample_async(
                sample.answer, sample.question
            )

            if result.is_valid:
                if result.corrected_code:
                    sample.answer = result.corrected_code
                verified.append(sample)
            else:
                failures.append(
                    {
                        "question": sample.question,
                        "answer": sample.answer,
                        "category": sample.category,
                        "question_type": sample.question_type,
                        "error_type": result.error_type,
                        "error_message": result.error_message,
                    }
                )

        return other_samples + verified, failures

    def _create_sample(
        self,
        item: dict,
        answer: str,
        test_code: Optional[str],
        entry_point: Optional[str],
    ) -> Sample:
        """Create a Sample from generation item."""
        chunk = item["chunk"]
        target_image = item.get("target_image")

        return Sample(
            question=item["question"],
            answer=answer,
            category=item["category"],
            question_type=item["question_type"].value,
            test_code=test_code,
            entry_point=entry_point,
            image_path=target_image.resolved_path if target_image else None,
            source_path=str(chunk.source_path),
            metadata=chunk.metadata,
        )

    async def _curate_samples_async(
        self,
        samples: list[Sample],
        progress_callbacks: dict,
    ) -> tuple[list[Sample], list[dict]]:
        """Quality validation for all samples."""
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

            messages = [
                Message(role="system", content=self.prompts.sample_curation_system),
                Message(role="user", content=user_prompt),
            ]
            curation_inputs.append(messages)

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
                        "category": sample.category,
                        "question_type": sample.question_type,
                        "image_path": sample.image_path,
                        "rejection_reason": reason,
                    }
                )

        print(f"  ✓ Curation: {len(validated)} passed, {len(rejected)} rejected")
        return validated, rejected

    def _parse_curation_response(self, response: str) -> tuple[str, str]:
        """Parse curation response."""
        response = response.strip()
        response_upper = response.upper()

        decision = None
        reason = "No reason provided"

        lines = response.split("\n")
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_upper = line_stripped.upper()

            if "DECISION" in line_upper:
                if ":" in line_stripped:
                    decision_text = line_stripped.split(":", 1)[1].strip().upper()
                else:
                    decision_text = line_stripped.upper()

                if "PASS" in decision_text or "YES" in decision_text:
                    decision = "PASS"
                elif "REJECT" in decision_text or "NO" in decision_text:
                    decision = "REJECT"

            elif "REASON" in line_upper:
                if ":" in line_stripped:
                    reason = line_stripped.split(":", 1)[1].strip()
                elif i + 1 < len(lines):
                    reason = lines[i + 1].strip()

        if decision is None:
            if response_upper.startswith("PASS") or response_upper.startswith("YES"):
                decision = "PASS"
            elif response_upper.startswith("REJECT") or response_upper.startswith("NO"):
                decision = "REJECT"

        if decision is None:
            pass_count = response_upper.count("PASS") + response_upper.count("YES")
            reject_count = response_upper.count("REJECT") + response_upper.count("NO")

            if pass_count > reject_count:
                decision = "PASS"
            elif reject_count > pass_count:
                decision = "REJECT"

        if decision is None:
            decision = "REJECT"
            reason = "Unclear response format"

        return decision, reason

    def _get_question_type_distribution(self, total_samples: int) -> dict[QuestionType, int]:
        """Calculate distribution of question types based on weights."""
        question_types = self.gen_config.question_types
        weights = self.gen_config.question_type_weights

        if not question_types:
            return {}

        active_weights = {qt: weights.get(qt, 1.0) for qt in question_types}
        total_weight = sum(active_weights.values())

        if total_weight == 0:
            active_weights = {qt: 1.0 for qt in question_types}
            total_weight = len(question_types)

        distribution = {}
        current_total = 0

        sorted_types = sorted(active_weights.items(), key=lambda x: x[1], reverse=True)

        for i, (qt, weight) in enumerate(sorted_types):
            if i == len(sorted_types) - 1:
                count = total_samples - current_total
            else:
                count = int(total_samples * (weight / total_weight))

            distribution[qt] = count
            current_total += count

        return distribution

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
        }
        metadata = {
            "generated_count": len(samples),
            "target_samples": self.gen_config.target_samples,
            "attempt": attempt + 1,
        }
        self.checkpoint_manager.save_checkpoint(checkpoint_state, metadata)

    async def _cleanup_async_clients(self):
        """Close all async HTTP clients."""
        clients = [
            self.model_registry.get_llm_client(self.gen_config.question_model),
            self.model_registry.get_llm_client(self.gen_config.answer_model),
            self.model_registry.get_llm_client(self.gen_config.curate_model),
        ]

        for client in clients:
            await client.aclose()

        if self.gen_config.vision_model:
            try:
                vision_client = self.model_registry.get_vlm_client(self.gen_config.vision_model)
                await vision_client.aclose()
            except (AttributeError, KeyError, ValueError):
                pass
