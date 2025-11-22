"""Category management and classification."""

import asyncio

from synthetic_data.config import CategoryConfig
from synthetic_data.extractors.chunker import Chunk
from synthetic_data.models import LLMClient, Message


class CategoryManager:
    """Manage categories and classify content."""

    def __init__(self, categories: list[CategoryConfig], llm_client: LLMClient | None = None):
        """
        Initialize category manager.

        Args:
            categories: List of category configurations
            llm_client: Optional LLM client for intelligent classification
        """
        self.categories = {cat.name: cat for cat in categories}
        self.llm_client = llm_client

    def classify_chunk(
        self,
        chunk: Chunk,
        prompt_template: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Classify a chunk into a category.

        Args:
            chunk: Content chunk to classify
            prompt_template: Optional prompt template for LLM classification
            system_prompt: Optional system prompt template

        Returns:
            Category name
        """
        # Use LLM classification if available
        if self.llm_client and prompt_template:
            return self._llm_classify(chunk, prompt_template, system_prompt)

        # Fallback to keyword-based classification
        return self._keyword_classify(chunk)

    def _llm_classify(
        self, chunk: Chunk, prompt_template: str, system_prompt: str | None = None
    ) -> str:
        """Classify using LLM."""
        categories_desc = "\n".join(
            [f"- {name}: {cat.description}" for name, cat in self.categories.items()]
        )

        if system_prompt:
            system_content = system_prompt.format(categories=categories_desc)
        else:
            system_content = "You are a content classifier. Return only the category name."

        user_prompt = prompt_template.format(content=chunk.text[:1500])

        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_prompt),
        ]

        try:
            response = self.llm_client.generate(messages, max_tokens=500, temperature=0.1)
            category = response.strip().lower().replace("-", "_").replace(" ", "_")

            # Validate response
            if category in self.categories:
                return category
        except Exception as e:
            print(f"LLM classification failed: {e}, falling back to keyword matching")

        return self._keyword_classify(chunk)

    def _keyword_classify(self, chunk: Chunk) -> str:
        """Classify using keyword matching (fallback)."""
        text_lower = chunk.text.lower()
        source_path_lower = str(chunk.source_path).lower()

        scores = {}
        for name, category in self.categories.items():
            score = 0
            keywords = category.description.lower().split()

            for keyword in keywords:
                score += text_lower.count(keyword) * 2
                if keyword in source_path_lower:
                    score += 5

            scores[name] = score

        if max(scores.values()) > 0:
            return max(scores, key=scores.get)

        return list(self.categories.keys())[0]

    def get_target_distribution(
        self, total_samples: int, chunks_by_category: dict[str, list[Chunk]]
    ) -> dict[str, int]:
        """
        Calculate target sample distribution across categories.
        Aims for equal distribution among categories that have chunks.

        Args:
            total_samples: Total number of samples to generate
            chunks_by_category: Chunks organized by category

        Returns:
            Dict mapping category names to target sample counts
        """
        # Identify categories that have at least one chunk
        active_categories = [
            name
            for name in self.categories
            if chunks_by_category.get(name) and len(chunks_by_category[name]) > 0
        ]

        if not active_categories:
            return {name: 0 for name in self.categories}

        # Equal distribution across active categories
        num_active = len(active_categories)
        base_count = total_samples // num_active
        remainder = total_samples % num_active

        distribution = {name: 0 for name in self.categories}

        for i, name in enumerate(active_categories):
            count = base_count
            if i < remainder:
                count += 1
            distribution[name] = count

        return distribution

    async def classify_chunks_batch_async(
        self,
        chunks: list[Chunk],
        prompt_template: str,
        system_prompt: str | None = None,
        batch_size: int = 32,
        max_concurrent: int = 10,
        progress_callback=None,
    ) -> list[str]:
        """
        Classify multiple chunks in batch using async processing.

        Args:
            chunks: List of content chunks to classify
            prompt_template: Prompt template for LLM classification
            system_prompt: Optional system prompt template
            batch_size: Size of batches for processing
            max_concurrent: Maximum concurrent LLM requests
            progress_callback: Optional callback function(completed_count) for progress tracking

        Returns:
            List of category names corresponding to each chunk
        """
        if not self.llm_client:
            return [self._keyword_classify(chunk) for chunk in chunks]

        categories_desc = "\n".join(
            [f"- {name}: {cat.description}" for name, cat in self.categories.items()]
        )

        if system_prompt:
            system_content = system_prompt.format(categories=categories_desc)
        else:
            system_content = "You are a content classifier. Return only the category name."

        # Prepare all classification prompts
        all_messages = []
        for chunk in chunks:
            user_prompt = prompt_template.format(content=chunk.text[:1500])

            messages = [
                Message(role="system", content=system_content),
                Message(role="user", content=user_prompt),
            ]
            all_messages.append(messages)

        # Classify in batch
        responses = await self.llm_client.generate_batch_async(
            all_messages,
            max_concurrent=max_concurrent,
            temperature=0.1,
            max_tokens=500,
            progress_callback=progress_callback,
        )

        # Parse and validate responses
        categories = []
        for i, response in enumerate(responses):
            category = response.strip().lower().replace("-", "_").replace(" ", "_")
            if category in self.categories:
                categories.append(category)
            else:
                categories.append(self._keyword_classify(chunks[i]))

        return categories

    def organize_by_category(
        self,
        chunks: list[Chunk],
        prompt_template: str | None = None,
        system_prompt: str | None = None,
        batch_size: int = 32,
        max_concurrent: int = 10,
        progress_callback=None,
    ) -> dict[str, list[Chunk]]:
        """
        Organize chunks by category using batch classification.

        Args:
            chunks: List of content chunks
            prompt_template: Optional prompt template for classification
            system_prompt: Optional system prompt template
            batch_size: Size of batches for processing
            max_concurrent: Maximum concurrent LLM requests
            progress_callback: Optional callback function(completed_count) for progress tracking

        Returns:
            Dict mapping category names to lists of chunks
        """
        organized = {name: [] for name in self.categories}

        if self.llm_client and prompt_template:
            categories = asyncio.run(
                self._classify_and_cleanup_async(
                    chunks,
                    prompt_template,
                    system_prompt,
                    batch_size,
                    max_concurrent,
                    progress_callback,
                )
            )
            for chunk, category in zip(chunks, categories):
                organized[category].append(chunk)
        else:
            for chunk in chunks:
                category = self._keyword_classify(chunk)
                organized[category].append(chunk)

        return organized

    async def _classify_and_cleanup_async(
        self,
        chunks: list[Chunk],
        prompt_template: str,
        system_prompt: str | None,
        batch_size: int,
        max_concurrent: int,
        progress_callback=None,
    ) -> list[str]:
        """Classify chunks and cleanup async client."""
        categories = await self.classify_chunks_batch_async(
            chunks, prompt_template, system_prompt, batch_size, max_concurrent, progress_callback
        )
        if self.llm_client:
            await self.llm_client.aclose()
        return categories
