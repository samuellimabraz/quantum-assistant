"""Main synthetic data generation pipeline with async batch processing."""

import asyncio
import random

from tqdm import tqdm

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
        )

        random.seed(config.seed)

        # Initialize deduplicator if enabled
        self.deduplicator = None
        if self.gen_config.enable_deduplication:
            self.deduplicator = Deduplicator(self.gen_config.similarity_threshold)

    def generate_samples(self, chunks_by_category: dict[str, list[Chunk]]) -> list[Sample]:
        """
        Generate synthetic samples from content chunks.

        Args:
            chunks_by_category: Chunks organized by category

        Returns:
            List of generated samples
        """
        # Run all generation in single async context to avoid event loop issues
        all_samples = asyncio.run(self._generate_all_samples_async(chunks_by_category))

        # Deduplicate if enabled
        if self.deduplicator:
            print("\nDeduplicating samples...")
            all_samples = self.deduplicator.deduplicate(all_samples)

        return all_samples

    async def _generate_all_samples_async(
        self, chunks_by_category: dict[str, list[Chunk]]
    ) -> list[Sample]:
        """Generate all samples in a single async context."""
        target_distribution = self.category_manager.get_target_distribution(
            self.gen_config.target_samples
        )

        all_samples = []

        for category, target_count in target_distribution.items():
            category_chunks = chunks_by_category.get(category, [])

            if not category_chunks:
                print(f"Warning: No chunks for category '{category}', skipping.")
                continue

            if target_count == 0:
                print(f"Warning: No samples allocated for category '{category}', skipping.")
                continue

            print(f"\nGenerating {target_count} samples for category: {category}")

            category_samples = await self._generate_category_samples_async(
                category, category_chunks, target_count
            )
            all_samples.extend(category_samples)

        return all_samples

    async def _generate_category_samples_async(
        self, category: str, chunks: list[Chunk], target_count: int
    ) -> list[Sample]:
        """Generate samples for a specific category using async batching."""
        samples = []

        question_types = self._get_question_type_distribution(target_count)
        difficulties = self._get_difficulty_distribution(target_count)

        batch_size = self.gen_config.llm_batch_size
        progress_bar = tqdm(total=target_count, desc=f"  {category}")

        for qtype, type_count in question_types.items():
            for difficulty, diff_count in difficulties.items():
                target = int((type_count / target_count) * diff_count)
                type_samples = []

                while len(type_samples) < target:
                    batch_chunks = random.choices(
                        chunks, k=min(batch_size, target - len(type_samples))
                    )

                    # Generate all questions in batch
                    try:
                        batch_samples = await self._generate_batch_samples_async(
                            batch_chunks, category, qtype, difficulty
                        )
                        type_samples.extend(batch_samples)
                        progress_bar.update(len(batch_samples))
                    except Exception as e:
                        print(f"\nBatch error: {e}, falling back to sequential")
                        # Fallback to sequential processing
                        for chunk in batch_chunks:
                            try:
                                sample = self._generate_sample(chunk, category, qtype, difficulty)
                                if sample:
                                    type_samples.append(sample)
                                    progress_bar.update(1)
                            except Exception as e2:
                                print(f"\nError: {e2}")
                                continue

                samples.extend(type_samples)

        progress_bar.close()
        return samples

    async def _generate_batch_samples_async(
        self,
        chunks: list[Chunk],
        category: str,
        question_type: QuestionType,
        difficulty: str,
    ) -> list[Sample]:
        """Generate multiple samples in parallel using async batching."""
        # Prepare all prompts for question generation
        question_inputs = []
        chunk_metadata = []

        for chunk in chunks:
            images_with_transcription = [img for img in chunk.images if img.transcription]
            use_image = (
                len(images_with_transcription) > 0
                and random.random() < self.gen_config.multimodal_ratio
            )

            prompt_template = self.prompts.get_question_prompt(question_type)
            context = chunk.text[: self.gen_config.max_context_length]

            kwargs = {"context": context, "difficulty": difficulty}

            if question_type == QuestionType.CODE and chunk.code_blocks:
                code = "\n\n".join(chunk.code_blocks[:2])
                kwargs["code_context"] = f"Example code:\n```python\n{code}\n```\n\n"
            else:
                kwargs["code_context"] = ""

            if use_image and images_with_transcription:
                kwargs["image_description"] = images_with_transcription[0].transcription

            user_prompt = prompt_template.format(**kwargs)
            messages = [Message(role="user", content=user_prompt)]
            question_inputs.append(messages)

            chunk_metadata.append(
                {
                    "chunk": chunk,
                    "use_image": use_image,
                    "images_with_transcription": images_with_transcription,
                }
            )

        # Generate questions in batch
        question_client = self.model_registry.get_llm_client(self.gen_config.question_model)
        questions = await question_client.generate_batch_async(
            question_inputs, max_concurrent=self.gen_config.llm_concurrency
        )

        # Prepare answer generation inputs
        answer_inputs = []
        valid_indices = []

        for i, (question, meta) in enumerate(zip(questions, chunk_metadata)):
            question = question.strip()
            if question:
                chunk = meta["chunk"]
                context = chunk.text[: self.gen_config.max_context_length]

                code_context = ""
                if chunk.code_blocks and question_type == QuestionType.CODE:
                    code = "\n\n".join(chunk.code_blocks)
                    code_context = f"Reference code:\n```python\n{code}\n```"

                user_prompt = self.prompts.answer_generation.format(
                    question=question,
                    context=context,
                    code_context=code_context,
                )

                messages = [Message(role="user", content=user_prompt)]
                answer_inputs.append(messages)
                valid_indices.append(i)

        # Generate answers in batch
        answer_client = self.model_registry.get_llm_client(self.gen_config.answer_model)
        answers = await answer_client.generate_batch_async(
            answer_inputs, max_concurrent=self.gen_config.llm_concurrency
        )

        # Build samples
        samples = []
        for answer_idx, chunk_idx in enumerate(valid_indices):
            question = questions[chunk_idx].strip()
            answer = answers[answer_idx].strip()
            meta = chunk_metadata[chunk_idx]
            chunk = meta["chunk"]

            if not answer:
                continue

            image_path = None
            if meta["use_image"] and meta["images_with_transcription"]:
                image_path = meta["images_with_transcription"][0].resolved_path

            code_context = None
            if chunk.code_blocks and question_type == QuestionType.CODE:
                code_context = "\n\n".join(chunk.code_blocks[:2])

            sample = Sample(
                question=question,
                answer=answer,
                category=category,
                question_type=question_type.value,
                difficulty=difficulty,
                image_path=image_path,
                code_context=code_context,
                source_path=str(chunk.source_path),
                metadata=chunk.metadata,
            )
            samples.append(sample)

        return samples

    def _generate_sample(
        self, chunk: Chunk, category: str, question_type: QuestionType, difficulty: str
    ) -> Sample | None:
        """Generate a single sample."""
        # Use image only if it has transcription
        images_with_transcription = [img for img in chunk.images if img.transcription]
        use_image = (
            len(images_with_transcription) > 0
            and random.random() < self.gen_config.multimodal_ratio
        )

        question = self._generate_question(chunk, question_type, difficulty, use_image)
        if not question:
            return None

        answer = self._generate_answer(chunk, question, question_type)
        if not answer:
            return None

        image_path = None
        if use_image and images_with_transcription:
            # Use the first image with transcription
            image_path = images_with_transcription[0].resolved_path

        code_context = None
        if chunk.code_blocks and question_type == QuestionType.CODE:
            code_context = "\n\n".join(chunk.code_blocks[:2])

        return Sample(
            question=question,
            answer=answer,
            category=category,
            question_type=question_type.value,
            difficulty=difficulty,
            image_path=image_path,
            code_context=code_context,
            source_path=str(chunk.source_path),
            metadata=chunk.metadata,
        )

    def _generate_question(
        self, chunk: Chunk, question_type: QuestionType, difficulty: str, use_image: bool
    ) -> str:
        """Generate a question using the question model."""
        prompt_template = self.prompts.get_question_prompt(question_type)

        context = chunk.text[: self.gen_config.max_context_length]

        kwargs = {"context": context, "difficulty": difficulty}

        if question_type == QuestionType.CODE and chunk.code_blocks:
            code = "\n\n".join(chunk.code_blocks[:2])
            kwargs["code_context"] = f"Example code:\n```python\n{code}\n```\n\n"
        else:
            kwargs["code_context"] = ""

        if use_image:
            # Get first image with transcription
            images_with_transcription = [img for img in chunk.images if img.transcription]
            if images_with_transcription:
                kwargs["image_description"] = images_with_transcription[0].transcription

        user_prompt = prompt_template.format(**kwargs)

        client = self.model_registry.get_llm_client(self.gen_config.question_model)
        messages = [Message(role="user", content=user_prompt)]

        return client.generate(messages).strip()

    def _generate_answer(self, chunk: Chunk, question: str, question_type: QuestionType) -> str:
        """Generate an answer using the answer model."""
        context = chunk.text[: self.gen_config.max_context_length]

        code_context = ""
        if chunk.code_blocks and question_type == QuestionType.CODE:
            code = "\n\n".join(chunk.code_blocks)
            code_context = f"Reference code:\n```python\n{code}\n```"

        user_prompt = self.prompts.answer_generation.format(
            question=question,
            context=context,
            code_context=code_context,
        )

        client = self.model_registry.get_llm_client(self.gen_config.answer_model)
        messages = [Message(role="user", content=user_prompt)]

        return client.generate(messages).strip()

    def _get_question_type_distribution(self, total_samples: int) -> dict[QuestionType, int]:
        """Calculate distribution of question types."""
        weights = self.gen_config.question_type_weights or {}
        question_types = self.gen_config.question_types

        if not weights:
            weights = {qt: 1.0 for qt in question_types}

        for qt in question_types:
            if qt not in weights:
                weights[qt] = 1.0

        total_weight = sum(weights.values())

        distribution = {}
        for qt in question_types:
            proportion = weights[qt] / total_weight
            distribution[qt] = int(total_samples * proportion)

        allocated = sum(distribution.values())
        if allocated < total_samples:
            max_weight_type = max(weights, key=weights.get)
            distribution[max_weight_type] += total_samples - allocated

        return distribution

    def _get_difficulty_distribution(self, total_samples: int) -> dict[str, int]:
        """Calculate distribution of difficulty levels."""
        weights = self.gen_config.difficulty_weights or {}
        difficulties = [d.value for d in self.gen_config.difficulty_levels]

        if not weights:
            weights = {d: 1.0 for d in difficulties}

        total_weight = sum(weights.values())

        distribution = {}
        for diff in difficulties:
            proportion = weights.get(diff, 1.0) / total_weight
            distribution[diff] = int(total_samples * proportion)

        allocated = sum(distribution.values())
        if allocated < total_samples:
            distribution[difficulties[0]] += total_samples - allocated

        return distribution
