"""Main synthetic data generation pipeline with async batch processing."""

import asyncio
import random

from synthetic_data.config import PipelineConfig, QuestionType
from synthetic_data.extractors.chunker import Chunk
from synthetic_data.models import Message, ModelRegistry, Sample
from synthetic_data.utils import Deduplicator
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
            answer_generation=config.prompts.answer_generation,
            summary_generation=config.prompts.summary_generation,
            caption_generation=config.prompts.caption_generation,
            code_generation=config.prompts.code_generation,
            content_quality_check=config.prompts.content_quality_check,
            image_quality_check=config.prompts.image_quality_check,
            category_classification=config.prompts.category_classification,
            qa_curation=getattr(
                config.prompts, "qa_curation", ""
            ),  # Handle older configs without this field
        )

        random.seed(config.seed)

        # Initialize deduplicator if enabled
        self.deduplicator = None
        if self.gen_config.enable_deduplication:
            self.deduplicator = Deduplicator(self.gen_config.similarity_threshold)

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
        """Generate all samples in a single async context with full batching."""
        # Get chunk-aware distribution (no weights, purely based on chunk availability)
        target_distribution = self.category_manager.get_target_distribution(
            self.gen_config.target_samples, chunks_by_category
        )

        print("\n[Sample distribution (chunk-aware)]")
        print("Category -> Target samples:")
        for cat in sorted(target_distribution.keys()):
            target = target_distribution.get(cat, 0)
            chunks = len(chunks_by_category.get(cat, []))
            if target > 0:
                print(f"  {cat}: {target} samples (from {chunks} chunks)")

        # Prepare all sample specifications across all categories
        print("\nPreparing sample generation plan...")
        all_sample_specs = []

        for category, target_count in target_distribution.items():
            category_chunks = chunks_by_category.get(category, [])

            if not category_chunks or target_count == 0:
                continue

            # Get question type distribution for this category
            question_types = self._get_question_type_distribution(target_count)

            # Create list of question types to cycle through
            qtype_list = []
            for qtype, type_count in question_types.items():
                qtype_list.extend([qtype] * type_count)

            random.shuffle(qtype_list)
            qtype_list = qtype_list[:target_count]

            # Create sample specifications
            for qtype in qtype_list:
                chunk = random.choice(category_chunks)
                all_sample_specs.append(
                    {
                        "chunk": chunk,
                        "category": category,
                        "question_type": qtype,
                    }
                )

        print(f"Total samples to generate: {len(all_sample_specs)}")

        # Generate all samples in batches
        all_samples = await self._batch_generate_all_samples(all_sample_specs, progress_callbacks)

        # Close async clients to avoid event loop issues
        await self._cleanup_async_clients()

        return all_samples

    async def _batch_generate_all_samples(
        self, sample_specs: list[dict], progress_callbacks: dict = None
    ) -> list[Sample]:
        """
        Generate all samples with full batching across categories.

        Args:
            sample_specs: List of sample specifications with chunk, category, question_type

        Returns:
            List of generated samples
        """
        total_samples = len(sample_specs)
        print(f"\nGenerating {total_samples} samples in batches...")

        # Step 1: Determine which samples will be multimodal
        # Ensure we hit the target multimodal ratio across all samples
        target_multimodal = int(total_samples * self.gen_config.multimodal_ratio)

        # Identify samples that can be multimodal (have images)
        multimodal_candidates = []
        non_multimodal = []

        for i, spec in enumerate(sample_specs):
            chunk = spec["chunk"]
            images_with_transcription = [img for img in chunk.images if img.transcription]

            # Caption questions MUST have images
            if spec["question_type"] == QuestionType.CAPTION:
                if images_with_transcription:
                    multimodal_candidates.append(i)
                # Skip caption questions without images
            elif images_with_transcription:
                multimodal_candidates.append(i)
            else:
                non_multimodal.append(i)

        # Randomly select which samples will use images to hit target ratio
        random.shuffle(multimodal_candidates)
        multimodal_indices = set(multimodal_candidates[:target_multimodal])

        # Step 2: Prepare question generation inputs
        question_inputs = []
        spec_metadata = []

        for i, spec in enumerate(sample_specs):
            chunk = spec["chunk"]
            question_type = spec["question_type"]

            images_with_transcription = [img for img in chunk.images if img.transcription]

            # Determine if this sample should use image based on our pre-selection
            use_image = i in multimodal_indices

            # Skip caption questions without images
            if question_type == QuestionType.CAPTION and not images_with_transcription:
                continue

            prompt_template = self.prompts.get_question_prompt(question_type)

            # Build context with code if available
            context = chunk.text[: self.gen_config.max_context_length]
            if chunk.code_blocks:
                code = "\n\n".join(chunk.code_blocks[:2])
                context = f"{context}\n\nCode example:\n```python\n{code}\n```"

            kwargs = {"context": context}

            if question_type == QuestionType.CAPTION:
                kwargs["image_description"] = images_with_transcription[0].transcription
            elif use_image and images_with_transcription:
                kwargs["image_description"] = images_with_transcription[0].transcription

            user_prompt = prompt_template.format(**kwargs)

            # Add system prompt for question generation
            messages = [
                Message(
                    role="system",
                    content="You generate natural, direct questions as a real user would ask. Be concise and objective.",
                ),
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

        # Set total and get progress callback for questions
        if progress_callbacks:
            set_total = progress_callbacks.get("set_questions_total")
            if set_total:
                set_total(len(question_inputs))
            question_progress = progress_callbacks.get("questions")
        else:
            question_progress = None

        questions = await question_client.generate_batch_async(
            question_inputs,
            max_concurrent=self.gen_config.llm_concurrency,
            progress_callback=question_progress,
        )

        print(f"  ✓ Generated {len(questions)} questions")

        # Step 3: Prepare answer generation inputs
        answer_inputs = []
        valid_specs = []

        for i, (question, meta) in enumerate(zip(questions, spec_metadata)):
            question = question.strip()
            if not question:
                continue

            chunk = meta["chunk"]
            question_type = meta["question_type"]

            # Build context with code if available
            context = chunk.text[: self.gen_config.max_context_length]
            if chunk.code_blocks:
                code = "\n\n".join(chunk.code_blocks)
                context = f"{context}\n\nCode example:\n```python\n{code}\n```"

            # Add image context if available
            if meta["use_image"] and meta["images_with_transcription"]:
                context = f"{context}\n\nImage description: {meta['images_with_transcription'][0].transcription}"

            user_prompt = self.prompts.answer_generation.format(
                question=question,
                context=context,
            )

            messages = [Message(role="user", content=user_prompt)]
            answer_inputs.append(messages)
            valid_specs.append((i, meta, question))

        # Step 4: Generate all answers in batch
        answer_client = self.model_registry.get_llm_client(self.gen_config.answer_model)

        # Set total and get progress callback for answers
        if progress_callbacks:
            set_total = progress_callbacks.get("set_answers_total")
            if set_total:
                set_total(len(answer_inputs))
            answer_progress = progress_callbacks.get("answers")
        else:
            answer_progress = None

        answers = await answer_client.generate_batch_async(
            answer_inputs,
            max_concurrent=self.gen_config.llm_concurrency,
            progress_callback=answer_progress,
        )

        print(f"  ✓ Generated {len(answers)} answers")

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

        # Step 6: Curate all samples in batch
        if initial_samples:
            # Set total and get progress callback for curation
            if progress_callbacks:
                set_total = progress_callbacks.get("set_curation_total")
                if set_total:
                    set_total(len(initial_samples))
                curation_progress = progress_callbacks.get("curation")
            else:
                curation_progress = None

            curated_samples = await self._curate_samples_async(initial_samples, curation_progress)
            return curated_samples

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
            except Exception:
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

    async def _curate_samples_async(
        self, samples: list[Sample], progress_callback=None
    ) -> list[Sample]:
        """
        Filter samples using binary quality check.

        Args:
            samples: Initial samples to filter
            progress_callback: Optional callback function(completed_count)

        Returns:
            Filtered samples that passed quality check
        """
        # Prepare quality check prompts
        quality_inputs = []
        curation_prompt_template = self.prompts.qa_curation

        for sample in samples:
            user_prompt = curation_prompt_template.format(
                question=sample.question,
                answer=sample.answer,
            )

            messages = [
                Message(
                    role="system",
                    content="Reasoning: low. You are a quality control filter. Respond only with PASS or REJECT.",
                ),
                Message(role="user", content=user_prompt),
            ]
            quality_inputs.append(messages)

        # Batch quality check using curate model
        curate_client = self.model_registry.get_llm_client(self.gen_config.curate_model)
        quality_responses = await curate_client.generate_batch_async(
            quality_inputs,
            max_concurrent=self.gen_config.llm_concurrency,
            temperature=0.0,
            progress_callback=progress_callback,
        )

        # Filter samples based on responses
        filtered_samples = []
        passed_count = 0
        rejected_count = 0

        for sample, response in zip(samples, quality_responses):
            decision = response.strip().upper()
            if "PASS" in decision:
                filtered_samples.append(sample)
                passed_count += 1
            else:
                rejected_count += 1

        print(f"  ✓ Quality filter: {passed_count} passed, {rejected_count} rejected")

        return filtered_samples
