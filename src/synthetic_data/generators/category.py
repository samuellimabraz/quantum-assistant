"""Category classification for generated samples."""

import asyncio
from typing import Optional

from synthetic_data.config import CategoryConfig
from synthetic_data.models import LLMClient, Message, Sample


class CategoryClassifier:
    """Classify generated samples into categories.

    Performs post-generation classification based on the actual
    question and answer content, which is more accurate than
    classifying raw chunks.
    """

    def __init__(self, categories: list[CategoryConfig], llm_client: Optional[LLMClient] = None):
        """Initialize category classifier.

        Args:
            categories: List of category configurations
            llm_client: LLM client for classification
        """
        self.categories = {cat.name: cat for cat in categories}
        self.llm_client = llm_client

    def classify_sample(
        self,
        sample: Sample,
        prompt_template: str,
        system_prompt: str,
    ) -> str:
        """Classify a sample into a category.

        Args:
            sample: Generated sample to classify
            prompt_template: Prompt template for classification
            system_prompt: System prompt with category descriptions

        Returns:
            Category name
        """
        if not self.llm_client:
            return self._keyword_classify(sample)

        categories_desc = self._format_categories()
        system_content = system_prompt.format(categories=categories_desc)

        user_prompt = prompt_template.format(
            question=sample.question[:1000],
            answer=sample.answer[:1000],
            question_type=sample.question_type,
        )

        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_prompt),
        ]

        try:
            response = self.llm_client.generate(messages, max_tokens=100, temperature=0.1)
            category = response.strip().lower().replace("-", "_").replace(" ", "_")

            if category in self.categories:
                return category
        except Exception:
            pass

        return self._keyword_classify(sample)

    async def classify_samples_async(
        self,
        samples: list[Sample],
        prompt_template: str,
        system_prompt: str,
        max_concurrent: int = 20,
        progress_callback=None,
    ) -> list[str]:
        """Classify multiple samples concurrently.

        Args:
            samples: List of samples to classify
            prompt_template: Prompt template for classification
            system_prompt: System prompt with category descriptions
            max_concurrent: Maximum concurrent requests
            progress_callback: Optional callback(completed_count)

        Returns:
            List of category names
        """
        if not self.llm_client:
            return [self._keyword_classify(s) for s in samples]

        categories_desc = self._format_categories()
        system_content = system_prompt.format(categories=categories_desc)

        # Prepare messages
        batch_messages = []
        for sample in samples:
            user_prompt = prompt_template.format(
                question=sample.question[:1000],
                answer=sample.answer[:1000],
                question_type=sample.question_type,
            )
            batch_messages.append([
                Message(role="system", content=system_content),
                Message(role="user", content=user_prompt),
            ])

        # Batch classify
        responses = await self.llm_client.generate_batch_async(
            batch_messages,
            max_concurrent=max_concurrent,
            temperature=0.1,
            max_tokens=100,
            progress_callback=progress_callback,
        )

        # Parse responses
        categories = []
        for i, response in enumerate(responses):
            category = response.strip().lower().replace("-", "_").replace(" ", "_")
            if category in self.categories:
                categories.append(category)
            else:
                categories.append(self._keyword_classify(samples[i]))

        return categories

    def _format_categories(self) -> str:
        """Format categories for prompt."""
        return "\n".join(
            f"- {name}: {cat.description}" for name, cat in self.categories.items()
        )

    def _keyword_classify(self, sample: Sample) -> str:
        """Classify using keyword matching (fallback)."""
        text = f"{sample.question} {sample.answer}".lower()

        scores = {}
        for name, category in self.categories.items():
            score = 0
            keywords = category.description.lower().split()

            for keyword in keywords:
                if len(keyword) > 3:
                    score += text.count(keyword) * 2

            scores[name] = score

        if max(scores.values()) > 0:
            return max(scores, key=scores.get)

        return list(self.categories.keys())[0]

    async def aclose(self):
        """Close async client."""
        if self.llm_client:
            await self.llm_client.aclose()
