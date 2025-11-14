"""Category management and classification."""

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
        self.total_weight = sum(cat.weight for cat in categories)
        self.llm_client = llm_client

    def classify_chunk(self, chunk: Chunk, prompt_template: str | None = None) -> str:
        """
        Classify a chunk into a category.

        Args:
            chunk: Content chunk to classify
            prompt_template: Optional prompt template for LLM classification

        Returns:
            Category name
        """
        # Use LLM classification if available
        if self.llm_client and prompt_template:
            return self._llm_classify(chunk, prompt_template)

        # Fallback to keyword-based classification
        return self._keyword_classify(chunk)

    def _llm_classify(self, chunk: Chunk, prompt_template: str) -> str:
        """Classify using LLM."""
        categories_desc = "\n".join(
            [f"- {name}: {cat.description}" for name, cat in self.categories.items()]
        )

        user_prompt = prompt_template.format(
            categories=categories_desc,
            content=chunk.text[:1500],
        )

        messages = [
            Message(
                role="system",
                content="Reasoning: low. You are a content classifier. Return only the category name.",
            ),
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

            scores[name] = score * category.weight

        if max(scores.values()) > 0:
            return max(scores, key=scores.get)

        return list(self.categories.keys())[0]

    def get_target_distribution(self, total_samples: int) -> dict[str, int]:
        """
        Calculate target sample distribution across categories.

        Args:
            total_samples: Total number of samples to generate

        Returns:
            Dict mapping category names to target sample counts
        """
        distribution = {}

        for name, category in self.categories.items():
            proportion = category.weight / self.total_weight
            distribution[name] = int(total_samples * proportion)

        # Adjust for rounding errors
        total_allocated = sum(distribution.values())
        if total_allocated < total_samples:
            highest_weight_cat = max(
                self.categories.keys(),
                key=lambda k: self.categories[k].weight,
            )
            distribution[highest_weight_cat] += total_samples - total_allocated

        return distribution

    def organize_by_category(
        self, chunks: list[Chunk], prompt_template: str | None = None
    ) -> dict[str, list[Chunk]]:
        """
        Organize chunks by category.

        Args:
            chunks: List of content chunks
            prompt_template: Optional prompt template for classification

        Returns:
            Dict mapping category names to lists of chunks
        """
        organized = {name: [] for name in self.categories}

        for chunk in chunks:
            category = self.classify_chunk(chunk, prompt_template)
            organized[category].append(chunk)

        return organized
