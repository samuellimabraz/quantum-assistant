"""Main synthetic data generation pipeline with async batch processing."""

import asyncio
import random

from synthetic_data.config import PipelineConfig, QuestionType
from synthetic_data.extractors.chunker import Chunk
from synthetic_data.models import Message, ModelRegistry, Sample
from synthetic_data.utils import Deduplicator
from synthetic_data.utils.code_verifier import CodeVerifier
from synthetic_data.generators.category import CategoryManager
from synthetic_data.generators.prompts import PromptSet


class GenerationPipeline:
    """Pipeline for generating synthetic dataset samples."""

    def __init__(
        self,
        config: PipelineConfig,
        model_registry: ModelRegistry,
        category_manager: CategoryManager,
    ):
        """
        Initialize generation pipeline.

        Args:
            config: Pipeline configuration
            model_registry: Model registry for LLM/VLM access
            category_manager: Category manager
        """
        self.config = config
        self.gen_config = config.generation
        self.model_registry = model_registry
        self.category_manager = category_manager

        # Create prompt set from config
        self.prompts = PromptSet(
            question_generation=config.prompts.question_generation,
            question_generation_system=getattr(config.prompts, "question_generation_system", ""),
            answer_generation=config.prompts.answer_generation,
            answer_generation_system=getattr(config.prompts, "answer_generation_system", ""),
            summary_generation=config.prompts.summary_generation,
            caption_generation=config.prompts.caption_generation,
            code_generation=config.prompts.code_generation,
            content_quality_check=config.prompts.content_quality_check,
            content_filter_system=getattr(config.prompts, "content_filter_system", ""),
            image_quality_check=config.prompts.image_quality_check,
            image_filter_system=getattr(config.prompts, "image_filter_system", ""),
            category_classification=config.prompts.category_classification,
            category_classification_system=getattr(
                config.prompts, "category_classification_system", ""
            ),
            sample_curation=getattr(config.prompts, "sample_curation", ""),
            sample_curation_system=getattr(config.prompts, "sample_curation_system", ""),
        )

        random.seed(config.seed)

        # Initialize deduplicator if enabled
        self.deduplicator = None
        if self.gen_config.enable_deduplication:
            self.deduplicator = Deduplicator(self.gen_config.similarity_threshold)

        # Initialize code verifier if enabled
        self.code_verifier = None
        if self.gen_config.enable_code_verification:
            correction_client = model_registry.get_llm_client(self.gen_config.answer_model)
            self.code_verifier = CodeVerifier(
                llm_client=correction_client,
                max_iterations=self.gen_config.code_verification_max_iterations,
                timeout_seconds=self.gen_config.code_verification_timeout,
            )

    def generate_samples(
        self, chunks_by_category: dict[str, list[Chunk]], progress_callbacks: dict = None
    ) -> list[Sample]:
        """
        Generate synthetic samples from content chunks.

        Args:
            chunks_by_category: Chunks organized by category
            progress_callbacks: Dict with callbacks for 'questions', 'answers', 'curation'

        Returns:
            List of generated samples
        """
        # Run all generation in single async context to avoid event loop issues
        all_samples = asyncio.run(
            self._generate_all_samples_async(chunks_by_category, progress_callbacks)
        )

        # Deduplicate if enabled
        if self.deduplicator:
            print("\nDeduplicating samples...")
            all_samples = self.deduplicator.deduplicate(all_samples)

        return all_samples

    async def _generate_all_samples_async(
        self, chunks_by_category: dict[str, list[Chunk]], progress_callbacks: dict = None
    ) -> list[Sample]:
        """Generate all samples in a single async context with full batching and retry loop."""

        final_samples = []
        target_samples = self.gen_config.target_samples
        max_attempts = 5
        attempt = 0

        # Pre-calculate chunk pools per category
        chunk_pools = {}
        for cat, chunks in chunks_by_category.items():
            mm_chunks = [
                c for c in chunks if c.images and any(img.transcription for img in c.images)
            ]
            txt_chunks = chunks  # All chunks can be text chunks
            chunk_pools[cat] = {"mm": mm_chunks, "txt": txt_chunks}

        while len(final_samples) < target_samples and attempt < max_attempts:
            needed = target_samples - len(final_samples)
            # Add buffer for rejections (20% buffer or at least 5 samples)
            batch_target = int(needed * 1.2) + 5

            print(
                f"\n[Attempt {attempt+1}/{max_attempts}] Generating batch of ~{batch_target} samples (Needed: {needed})"
            )

            # Get distribution for this batch
            target_distribution = self.category_manager.get_target_distribution(
                batch_target, chunks_by_category
            )

            # Prepare sample specifications
            all_sample_specs = []

            for category, cat_target in target_distribution.items():
                if cat_target == 0:
                    continue

                pools = chunk_pools.get(category)
                if not pools:
                    continue

                # Determine MM vs Text split for this category in this batch
                # We aim for the configured ratio
                n_mm = int(cat_target * self.gen_config.multimodal_ratio)
                n_txt = cat_target - n_mm

                # Adjust if pools are empty
                if not pools["mm"]:
                    n_txt += n_mm
                    n_mm = 0

                # Select chunks
                selected_mm_chunks = []
                if n_mm > 0 and pools["mm"]:
                    # Sampling with replacement if needed
                    selected_mm_chunks = random.choices(pools["mm"], k=n_mm)

                selected_txt_chunks = []
                if n_txt > 0 and pools["txt"]:
                    selected_txt_chunks = random.choices(pools["txt"], k=n_txt)

                # Create batch items
                batch_items = []
                for c in selected_mm_chunks:
                    batch_items.append({"chunk": c, "use_image": True, "category": category})
                for c in selected_txt_chunks:
                    batch_items.append({"chunk": c, "use_image": False, "category": category})

                # Now assign question types
                # We need to ensure CAPTION types get use_image=True
                # And others get distributed

                # Get distribution of types for this category's batch
                type_dist = self._get_question_type_distribution(len(batch_items))
                types_list = []
                for qt, count in type_dist.items():
                    types_list.extend([qt] * count)
                random.shuffle(types_list)

                # Assign types to items
                # Priority: CAPTION must go to items with use_image=True

                # Separate items
                mm_items = [x for x in batch_items if x["use_image"]]
                txt_items = [x for x in batch_items if not x["use_image"]]

                final_cat_specs = []

                # Iterate through types and assign to compatible items
                for qtype in types_list:
                    if qtype == QuestionType.CAPTION:
                        if mm_items:
                            item = mm_items.pop()
                            item["question_type"] = qtype
                            final_cat_specs.append(item)
                        else:
                            # No MM items left for caption, swap to another type or skip?
                            # Swap to QA
                            if txt_items:
                                item = txt_items.pop()
                                item["question_type"] = QuestionType.QA
                                final_cat_specs.append(item)
                            elif mm_items:  # Should not happen given check above
                                item = mm_items.pop()
                                item["question_type"] = QuestionType.QA
                                final_cat_specs.append(item)
                    else:
                        if txt_items:
                            item = txt_items.pop()
                            item["question_type"] = qtype
                            final_cat_specs.append(item)
                        elif mm_items:
                            item = mm_items.pop()
                            item["question_type"] = qtype
                            final_cat_specs.append(item)

                all_sample_specs.extend(final_cat_specs)

            if not all_sample_specs:
                print("No samples to generate in this batch.")
                break

            print(f"Generating {len(all_sample_specs)} samples...")

            # Generate batch
            batch_samples = await self._batch_generate_all_samples(
                all_sample_specs, progress_callbacks
            )

            if self.code_verifier:
                print(f"Verifying code in {len(batch_samples)} samples...")
                verified_samples, code_failures = await self._verify_code_batch(batch_samples)

                if code_failures:
                    print(
                        f"  ⚠ Code verification: {len(code_failures)} samples failed verification"
                    )
                    # Save failed samples for debugging
                    if progress_callbacks and "save_code_failures" in progress_callbacks:
                        progress_callbacks["save_code_failures"](code_failures)

                batch_samples = verified_samples

            # Curate batch
            if self.gen_config.enable_curate_filtering:

                curated_samples, rejected = await self._curate_samples_async(
                    batch_samples,
                    progress_callbacks.get("curation") if progress_callbacks else None,
                )

                if progress_callbacks and "save_rejected" in progress_callbacks:
                    progress_callbacks["save_rejected"](rejected)

                final_samples.extend(curated_samples)
            else:
                final_samples.extend(batch_samples)

            print(f"Progress: {len(final_samples)}/{target_samples} samples generated.")
            attempt += 1

            # If we have enough, trim excess
            if len(final_samples) >= target_samples:
                final_samples = final_samples[:target_samples]
                break

        await self._cleanup_async_clients()

        return final_samples

    async def _batch_generate_all_samples(
        self, sample_specs: list[dict], progress_callbacks: dict = None
    ) -> list[Sample]:
        """
        Generate all samples with full batching across categories.

        Args:
            sample_specs: List of sample specifications with chunk, category, question_type, use_image
            progress_callbacks: Dict of callbacks

        Returns:
            List of generated samples
        """
        # Step 1: Prepare question generation inputs
        question_inputs = []
        spec_metadata = []

        for i, spec in enumerate(sample_specs):
            chunk = spec["chunk"]
            question_type = spec["question_type"]
            use_image = spec.get("use_image", False)

            images_with_transcription = [img for img in chunk.images if img.transcription]

            # Validation: if use_image is True, we must have images
            if use_image and not images_with_transcription:
                # Fallback to text-only if something went wrong
                use_image = False

            # Get appropriate prompt template
            prompt_template = self.prompts.get_question_prompt(question_type)

            # Build context - IMAGE FIRST if available, then text context
            context_parts = []

            # 1. Add image description FIRST if using image
            if use_image and images_with_transcription:
                context_parts.append(
                    f"[Image description]\n{images_with_transcription[0].transcription}\n"
                )

            # 2. Add text context
            if chunk.previous_chunk_text:
                context_parts.append(f"[Previous context]\n{chunk.previous_chunk_text}\n")

            context_parts.append(chunk.text[: self.gen_config.max_context_length])

            if chunk.next_chunk_text:
                context_parts.append(f"\n[Next context]\n{chunk.next_chunk_text}")

            # 3. Add all code from document
            if chunk.extended_code_context:
                context_parts.append(
                    f"\n[All code in document]\n```python\n{chunk.extended_code_context}\n```"
                )

            context = "\n".join(context_parts)

            # Special handling for caption - image description is required
            if question_type == QuestionType.CAPTION:
                if not images_with_transcription:
                    continue  # Skip caption questions without images
                user_prompt = prompt_template.format(
                    image_description=images_with_transcription[0].transcription, context=context
                )
            else:
                user_prompt = prompt_template.format(context=context)

            system_prompt = self.prompts.get_question_system_prompt(use_image=use_image)

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]
            question_inputs.append(messages)

            spec_metadata.append(
                {
                    **spec,
                    "use_image": use_image,
                    "images_with_transcription": images_with_transcription,
                }
            )

        # Step 2: Generate all questions in batch
        question_client = self.model_registry.get_llm_client(self.gen_config.question_model)

        if progress_callbacks and "set_questions_total" in progress_callbacks:
            progress_callbacks["set_questions_total"](len(question_inputs))

        questions = await question_client.generate_batch_async(
            question_inputs,
            max_concurrent=self.gen_config.llm_concurrency,
            progress_callback=progress_callbacks.get("questions") if progress_callbacks else None,
        )

        # Step 3: Prepare answer generation inputs
        answer_inputs = []
        valid_specs = []

        for i, (question, meta) in enumerate(zip(questions, spec_metadata)):
            question = question.strip()
            if not question:
                continue

            chunk = meta["chunk"]

            # Build context for answer generation - IMAGE FIRST if available
            context_parts = []

            # 1. Add image description FIRST if available
            if meta["use_image"] and meta["images_with_transcription"]:
                context_parts.append(
                    f"[Image description]\n{meta['images_with_transcription'][0].transcription}\n"
                )

            # 2. Add text context
            if chunk.previous_chunk_text:
                context_parts.append(f"[Previous context]\n{chunk.previous_chunk_text}\n")

            context_parts.append(chunk.text[: self.gen_config.max_context_length])

            if chunk.next_chunk_text:
                context_parts.append(f"\n[Next context]\n{chunk.next_chunk_text}")

            # 3. Add all code from document
            if chunk.extended_code_context:
                context_parts.append(
                    f"\n[All code in document - use this for reference]\n```python\n{chunk.extended_code_context}\n```"
                )

            context = "\n".join(context_parts)

            user_prompt = self.prompts.answer_generation.format(
                question=question,
                context=context,
            )

            # Get appropriate system prompt (enhanced if using image)
            system_prompt = self.prompts.get_answer_system_prompt(use_image=meta["use_image"])

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]
            answer_inputs.append(messages)
            valid_specs.append((i, meta, question))

        # Step 4: Generate all answers in batch
        answer_client = self.model_registry.get_llm_client(self.gen_config.answer_model)

        if progress_callbacks and "set_answers_total" in progress_callbacks:
            progress_callbacks["set_answers_total"](len(answer_inputs))

        answers = await answer_client.generate_batch_async(
            answer_inputs,
            max_concurrent=self.gen_config.llm_concurrency,
            progress_callback=progress_callbacks.get("answers") if progress_callbacks else None,
        )

        # Step 5: Build initial samples
        initial_samples = []
        for answer_idx, (_, meta, question) in enumerate(valid_specs):
            answer = answers[answer_idx].strip()
            if not answer:
                continue

            chunk = meta["chunk"]
            question_type = meta["question_type"]

            image_path = None
            if meta["use_image"] and meta["images_with_transcription"]:
                image_path = meta["images_with_transcription"][0].resolved_path

            sample = Sample(
                question=question,
                answer=answer,
                category=meta["category"],
                question_type=question_type.value,
                image_path=image_path,
                source_path=str(chunk.source_path),
                metadata=chunk.metadata,
            )
            initial_samples.append(sample)

        return initial_samples

    async def _cleanup_async_clients(self):
        """Close all async HTTP clients to prevent event loop issues."""
        # Get all clients from the model registry
        question_client = self.model_registry.get_llm_client(self.gen_config.question_model)
        answer_client = self.model_registry.get_llm_client(self.gen_config.answer_model)
        curate_client = self.model_registry.get_llm_client(self.gen_config.curate_model)

        # Close async clients
        await question_client.aclose()
        await answer_client.aclose()
        await curate_client.aclose()

        # Close VLM client if it exists
        if self.gen_config.vision_model:
            try:
                vision_client = self.model_registry.get_vlm_client(self.gen_config.vision_model)
                await vision_client.aclose()
            except (AttributeError, KeyError, ValueError):
                pass  # Vision model may not be configured

    def _get_question_type_distribution(self, total_samples: int) -> dict[QuestionType, int]:
        """Calculate equal distribution of question types."""
        question_types = self.gen_config.question_types
        num_types = len(question_types)

        if num_types == 0:
            return {}

        # Equal distribution across all types
        base_count = total_samples // num_types
        remainder = total_samples % num_types

        distribution = {}
        for i, qt in enumerate(question_types):
            # Give remainder samples to first few types
            distribution[qt] = base_count + (1 if i < remainder else 0)

        return distribution

    def _parse_curation_response(self, response: str) -> tuple[str, str]:
        """
        Parse curation response.

        Args:
            response: Model response

        Returns:
            Tuple of (decision, reason) where decision is "PASS" or "REJECT"
        """
        response = response.strip()
        response_upper = response.upper()

        # Initialize defaults
        decision = None
        reason = "No reason provided"

        # Try structured format
        lines = response.split("\n")
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_upper = line_stripped.upper()

            if "DECISION" in line_upper or "DEC" in line_upper:
                # Extract decision value after colon or space
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
                else:
                    if i + 1 < len(lines):
                        reason = lines[i + 1].strip()

        # Fallback 1: Check if entire response starts with PASS/REJECT/YES/NO
        if decision is None:
            if response_upper.startswith("PASS") or response_upper.startswith("YES"):
                decision = "PASS"
            elif response_upper.startswith("REJECT") or response_upper.startswith("NO"):
                decision = "REJECT"

        # Fallback 2: Search for PASS/REJECT/YES/NO anywhere
        if decision is None:
            pass_count = response_upper.count("PASS") + response_upper.count("YES")
            reject_count = response_upper.count("REJECT") + response_upper.count("NO")

            if pass_count > reject_count:
                decision = "PASS"
            elif reject_count > pass_count:
                decision = "REJECT"

        # Default to REJECT if still unclear
        if decision is None:
            decision = "REJECT"
            reason = "Unclear response format"

        return decision, reason

    async def _curate_samples_async(
        self, samples: list[Sample], progress_callback=None
    ) -> tuple[list[Sample], list[dict]]:
        """
        Comprehensive quality validation for all sample types.

        Checks correctness, quality, relevance, and educational value.

        Args:
            samples: Initial samples to validate
            progress_callback: Optional callback function(completed_count)

        Returns:
            Tuple of (validated samples that passed, rejected samples with reasons)
        """
        quality_inputs = []
        curation_prompt_template = self.prompts.sample_curation

        for sample in samples:
            user_prompt = curation_prompt_template.format(
                question=sample.question,
                answer=sample.answer,
                question_type=sample.question_type,
                has_image="yes" if sample.image_path else "no",
            )

            messages = [
                Message(role="system", content=self.prompts.sample_curation_system),
                Message(role="user", content=user_prompt),
            ]
            quality_inputs.append(messages)

        # Batch quality validation
        curate_client = self.model_registry.get_llm_client(self.gen_config.curate_model)
        quality_responses = await curate_client.generate_batch_async(
            quality_inputs,
            max_concurrent=self.gen_config.llm_concurrency,
            temperature=0.1,
            progress_callback=progress_callback,
        )

        # Filter samples and collect detailed rejection info
        validated_samples = []
        rejected_samples = []

        for sample, response in zip(samples, quality_responses):
            decision, reason = self._parse_curation_response(response)

            if decision == "PASS":
                validated_samples.append(sample)
            else:
                rejected_samples.append(
                    {
                        "question": sample.question,
                        "answer": sample.answer,
                        "category": sample.category,
                        "question_type": sample.question_type,
                        "image_path": sample.image_path,
                        "source_path": sample.source_path,
                        "rejection_reason": reason,
                        "full_response": response,
                    }
                )

        passed = len(validated_samples)
        rejected = len(rejected_samples)
        print(f"  ✓ Quality validation: {passed} passed, {rejected} rejected")

        return validated_samples, rejected_samples

    async def _verify_code_batch(self, samples: list[Sample]) -> tuple[list[Sample], list[dict]]:
        """
        Verify and correct code in a batch of samples.

        Args:
            samples: List of samples to verify

        Returns:
            Tuple of (verified samples, failed samples)
        """
        verified_samples = []
        failed_samples = []
        corrected_count = 0

        for sample in samples:
            # Only verify samples that likely contain code
            if sample.question_type in ["code", "qa"]:
                result = self.code_verifier.verify_and_correct_sample(
                    sample.answer, question=sample.question
                )

                if result.is_valid:
                    # Update sample if code was corrected
                    if result.corrected_code:
                        sample.answer = result.corrected_code
                        corrected_count += 1
                    verified_samples.append(sample)
                else:
                    failed_samples.append(
                        {
                            "question": sample.question,
                            "answer": sample.answer,
                            "category": sample.category,
                            "question_type": sample.question_type,
                            "source_path": sample.source_path,
                            "error_type": result.error_type,
                            "error_message": result.error_message,
                            "iterations_used": result.iterations_used,
                        }
                    )
            else:
                # Non-code samples pass through
                verified_samples.append(sample)

        if corrected_count > 0:
            print(f"  ✓ Code verification: {corrected_count} samples auto-corrected")

        return verified_samples, failed_samples
