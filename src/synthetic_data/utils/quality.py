"""Content quality filtering with batch support."""

from typing import List, Tuple

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
            response = self.llm_client.generate(messages, max_tokens=500, temperature=0.1)
            return response.strip().lower().startswith("yes")
        except Exception as e:
            print(f"Image quality check failed: {e}")
            return True

    async def filter_chunks_batch_async(
        self,
        chunks: List[Chunk],
        content_prompt: str,
        image_prompt: str,
        batch_size: int = 10,
        max_concurrent: int = 20,
        progress_callback=None,
    ) -> List[Tuple[Chunk, bool]]:
        """
        Filter chunks in batch using async processing.

        Args:
            chunks: List of chunks to filter
            content_prompt: Prompt template for content quality
            image_prompt: Prompt template for image quality
            batch_size: Batch size for processing
            max_concurrent: Max concurrent requests
            progress_callback: Optional callback function(completed_count) for progress tracking

        Returns:
            List of (chunk, passed_filter) tuples
        """
        # Prepare all content quality check messages
        content_messages = []
        for chunk in chunks:
            if len(chunk.text) < 50:
                content_messages.append(None)
            else:
                user_prompt = content_prompt.format(content=chunk.text)
                messages = [
                    Message(
                        role="system",
                        content="Reasoning: low. You are a content quality evaluator. Respond with only 'yes' or 'no'.",
                    ),
                    Message(role="user", content=user_prompt),
                ]
                content_messages.append(messages)

        # Batch check content quality
        content_results = []
        for i in range(0, len(content_messages), batch_size):
            batch = content_messages[i : i + batch_size]
            batch_to_process = [msg for msg in batch if msg is not None]

            if batch_to_process:
                try:
                    batch_progress = None
                    if progress_callback:
                        batch_start = i

                        def batch_progress_callback(completed):
                            progress_callback(batch_start + completed)

                        batch_progress = batch_progress_callback

                    responses = await self.llm_client.generate_batch_async(
                        batch_to_process,
                        max_tokens=500,
                        temperature=0.1,
                        max_concurrent=max_concurrent,
                        progress_callback=batch_progress,
                    )

                    # Map responses back to original positions
                    response_idx = 0
                    for msg in batch:
                        if msg is None:
                            content_results.append(False)
                        else:
                            response = responses[response_idx]
                            content_results.append(response.strip().lower().startswith("yes"))
                            response_idx += 1
                except Exception as e:
                    print(f"Batch content quality check failed: {e}")
                    content_results.extend([True] * len(batch))
            else:
                content_results.extend([False] * len(batch))

        results = []
        for i, (chunk, content_passed) in enumerate(zip(chunks, content_results)):
            if not content_passed:
                results.append((chunk, False))
            elif chunk.is_multimodal and chunk.images:
                # Filter images
                filtered_chunk = Chunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    code_blocks=chunk.code_blocks,
                    images=[],  # Will add filtered images
                    source_path=chunk.source_path,
                    metadata=chunk.metadata,
                )

                for img in chunk.images:
                    if img.transcription and img.resolved_path:
                        if self.is_quality_image(img, image_prompt):
                            filtered_chunk.images.append(img)

                results.append((filtered_chunk, True))
            else:
                results.append((chunk, True))

        return results
