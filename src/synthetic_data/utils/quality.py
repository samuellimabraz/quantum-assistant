"""Content quality filtering."""

from synthetic_data.extractors.chunker import Chunk
from synthetic_data.models import LLMClient, Message
from synthetic_data.parsers.base import ImageReference


class QualityFilter:
    """Filter low-quality content chunks and images."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize quality filter.

        Args:
            llm_client: LLM client for content and image quality checks
        """
        self.llm_client = llm_client

    def is_quality_content(self, chunk: Chunk, prompt_template: str) -> bool:
        """
        Check if content chunk is high quality for training.

        Args:
            chunk: Content chunk to check
            prompt_template: Prompt template for quality check

        Returns:
            True if content is high quality
        """
        if len(chunk.text) < 50:
            return False

        user_prompt = prompt_template.format(content=chunk.text)

        messages = [
            Message(
                role="system",
                content="Reasoning: low. You are a content quality evaluator. Respond with only 'yes' or 'no'.",
            ),
            Message(role="user", content=user_prompt),
        ]

        try:
            response = self.llm_client.generate(messages, max_tokens=500, temperature=0.1)
            return response.strip().lower().startswith("yes")
        except Exception as e:
            print(f"Quality check failed: {e}")
            return True

    def is_quality_image(self, image_ref: ImageReference, prompt_template: str) -> bool:
        """
        Check if image is relevant for training based on its transcription.

        Args:
            image_ref: Image reference with transcription to check
            prompt_template: Prompt template for quality check

        Returns:
            True if image is high quality and relevant
        """
        # Skip if no transcription available
        if not image_ref.transcription:
            return False

        # Skip if no resolved path
        if not image_ref.resolved_path:
            return False

        # Build context for quality check
        context_parts = []
        if image_ref.alt_text:
            context_parts.append(f"Alt text: {image_ref.alt_text}")
        if image_ref.caption:
            context_parts.append(f"Caption: {image_ref.caption}")

        context = "\n".join(context_parts) if context_parts else "No additional context"

        user_prompt = prompt_template.format(
            transcription=image_ref.transcription,
            context=context,
        )

        messages = [
            Message(
                role="system",
                content="Reasoning: low. You are an image quality evaluator. Respond with only 'yes' or 'no'.",
            ),
            Message(role="user", content=user_prompt),
        ]

        try:
            response = self.llm_client.generate(messages, max_tokens=100, temperature=0.1)
            return response.strip().lower().startswith("yes")
        except Exception as e:
            print(f"Image quality check failed: {e}")
            return True
