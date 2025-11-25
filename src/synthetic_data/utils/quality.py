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

    def _parse_filter_response(self, response: str) -> Tuple[bool, str]:
        """
        Parse filter response with robust handling of various formats.

        Args:
            response: Model response (may vary in format)

        Returns:
            Tuple of (passed, reason)
        """
        response = response.strip()
        response_lower = response.lower()

        decision = None
        reason = "No reason provided"

        # Try structured format: "DECISION: yes/no\nREASON: ..."
        lines = response.split("\n")
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            if "decision" in line_lower or "dec" in line_lower:
                # Extract decision value after colon or space
                if ":" in line_stripped:
                    decision_text = line_stripped.split(":", 1)[1].strip().lower()
                else:
                    # No colon - try to extract yes/no from line
                    decision_text = line_stripped.lower()

                # Look for yes/no in the decision text
                if "yes" in decision_text:
                    decision = "yes"
                elif "no" in decision_text:
                    decision = "no"

            # Look for reason
            elif "reason" in line_lower:
                if ":" in line_stripped:
                    reason = line_stripped.split(":", 1)[1].strip()
                else:
                    # Reason might be on next line
                    if i + 1 < len(lines):
                        reason = lines[i + 1].strip()

        # Fallback 1: Check if entire response starts with yes/no
        if decision is None:
            if response_lower.startswith("yes"):
                decision = "yes"
            elif response_lower.startswith("no"):
                decision = "no"

        # Fallback 2: Search for yes/no anywhere in response
        if decision is None:
            # Count occurrences
            yes_count = response_lower.count("yes")
            no_count = response_lower.count("no")

            if yes_count > no_count:
                decision = "yes"
            elif no_count > yes_count:
                decision = "no"

        # Default to "no" (reject) if still unclear
        if decision is None:
            decision = "no"
            reason = "Unclear response format"

        passed = decision == "yes"
        return passed, reason

    def is_quality_content(
        self, chunk: Chunk, prompt_template: str, system_prompt: str | None = None
    ) -> bool:
        """
        Check if content chunk is high quality for training.

        Args:
            chunk: Content chunk to check
            prompt_template: Prompt template for quality check
            system_prompt: Optional system prompt

        Returns:
            True if content is high quality
        """
        if len(chunk.text) < 50:
            return False

        user_prompt = prompt_template.format(content=chunk.text)

        system_content = (
            system_prompt
            if system_prompt
            else "You are a content quality evaluator. Respond with only 'yes' or 'no'."
        )

        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_prompt),
        ]

        try:
            response = self.llm_client.generate(messages, max_tokens=1024, temperature=0.1)
            return response.strip().lower().startswith("yes")
        except Exception as e:
            print(f"Quality check failed: {e}")
            return True

    def is_quality_image(
        self, image_ref: ImageReference, prompt_template: str, system_prompt: str | None = None
    ) -> bool:
        """
        Check if image is relevant for training based on its transcription.

        Args:
            image_ref: Image reference with transcription to check
            prompt_template: Prompt template for quality check
            system_prompt: Optional system prompt

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

        system_content = (
            system_prompt
            if system_prompt
            else "You are an image quality evaluator. Respond with only 'yes' or 'no'."
        )

        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_prompt),
        ]

        try:
            response = self.llm_client.generate(messages, max_tokens=1024, temperature=0.1)
            return response.strip().lower().startswith("yes")
        except Exception as e:
            print(f"Image quality check failed: {e}")
            return True

    async def filter_chunks_batch_async(
        self,
        chunks: List[Chunk],
        content_prompt: str,
        image_prompt: str,
        content_system_prompt: str | None = None,
        image_system_prompt: str | None = None,
        batch_size: int = 10,
        max_concurrent: int = 20,
        progress_callback=None,
        checkpoint_callback=None,
        save_debug: bool = True,
    ) -> Tuple[List[Tuple[Chunk, bool]], List[dict]]:
        """
        Filter chunks in batch using async processing with checkpoint support.

        Args:
            chunks: List of chunks to filter
            content_prompt: Prompt template for content quality
            image_prompt: Prompt template for image quality
            content_system_prompt: Optional system prompt for content filtering
            image_system_prompt: Optional system prompt for image filtering
            batch_size: Batch size for processing
            max_concurrent: Max concurrent requests
            progress_callback: Optional callback function(completed_count) for progress tracking
            checkpoint_callback: Optional callback function(results, debug_info) for checkpoint saving
            save_debug: Whether to save debug information

        Returns:
            Tuple of (List of (chunk, passed_filter) tuples, debug_info list)
        """
        debug_info = []
        results = []

        content_system_content = (
            content_system_prompt
            if content_system_prompt
            else "You are a content quality evaluator. Respond with only 'yes' or 'no'."
        )

        content_messages = []
        for chunk in chunks:
            if len(chunk.text) < 50:
                content_messages.append(None)
            else:
                user_prompt = content_prompt.format(content=chunk.text)
                messages = [
                    Message(role="system", content=content_system_content),
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
                        max_tokens=1024,
                        temperature=0.1,
                        max_concurrent=max_concurrent,
                        progress_callback=batch_progress,
                    )

                    # Map responses back to original positions
                    response_idx = 0
                    for batch_idx, msg in enumerate(batch):
                        chunk_idx = i + batch_idx
                        if msg is None:
                            content_results.append(False)
                        else:
                            response = responses[response_idx]
                            passed, reason = self._parse_filter_response(response)
                            content_results.append(passed)

                            # Save debug info
                            if save_debug and chunk_idx < len(chunks):
                                debug_info.append(
                                    {
                                        "type": "content",
                                        "chunk_id": chunks[chunk_idx].chunk_id,
                                        "source": str(chunks[chunk_idx].source_path),
                                        "decision": "PASS" if passed else "REJECT",
                                        "reason": reason,
                                        "content_preview": chunks[chunk_idx].text[:200],
                                        "full_response": response,
                                    }
                                )

                            response_idx += 1
                except Exception as e:
                    print(f"Batch content quality check failed: {e}")
                    content_results.extend([True] * len(batch))
            else:
                content_results.extend([False] * len(batch))

        unique_images = {}  # image_path -> (ImageReference, chunk_text)
        for chunk_idx, chunk in enumerate(chunks):
            if content_results[chunk_idx] and chunk.is_multimodal and chunk.images:
                for img in chunk.images:
                    if img.transcription and img.resolved_path:
                        if img.resolved_path not in unique_images:

                            context_preview = (
                                chunk.text[:500] if len(chunk.text) > 500 else chunk.text
                            )
                            unique_images[img.resolved_path] = (img, context_preview, chunk_idx)

        # Batch check all unique images
        image_results = {}  # image_path -> (passed, reason)
        if unique_images:
            image_system_content = (
                image_system_prompt
                if image_system_prompt
                else "You are an image quality evaluator. Respond with only 'yes' or 'no'."
            )

            image_messages = []
            image_paths = []

            for img_path, (img, chunk_context, chunk_idx) in unique_images.items():
                context_parts = []
                if img.alt_text:
                    context_parts.append(f"Alt text: {img.alt_text}")
                if img.caption:
                    context_parts.append(f"Caption: {img.caption}")
                if img.context:
                    context_parts.append(f"Image context: {img.context}")
                
                context_parts.append(f"Surrounding text: {chunk_context}")

                context = "\n".join(context_parts) if context_parts else "No additional context"

                user_prompt = image_prompt.format(
                    transcription=img.transcription,
                    context=context,
                )

                messages = [
                    Message(role="system", content=image_system_content),
                    Message(role="user", content=user_prompt),
                ]
                image_messages.append(messages)
                image_paths.append((img_path, img, chunk_idx))

            # Batch process image quality checks
            try:
                image_responses = await self.llm_client.generate_batch_async(
                    image_messages,
                    max_tokens=1024,
                    temperature=0.1,
                    max_concurrent=max_concurrent,
                )

                for idx, (img_path, img, chunk_idx) in enumerate(image_paths):
                    response = image_responses[idx]
                    passed, reason = self._parse_filter_response(response)
                    image_results[img_path] = (passed, reason)

                    if save_debug:
                        debug_info.append(
                            {
                                "type": "image",
                                "chunk_id": chunks[chunk_idx].chunk_id,
                                "source": str(chunks[chunk_idx].source_path),
                                "image_path": img_path,
                                "decision": "PASS" if passed else "REJECT",
                                "reason": reason,
                                "transcription_preview": img.transcription[:200],
                                "full_response": response,
                            }
                        )
            except Exception as e:
                print(f"Batch image quality check failed: {e}")
                for img_path in unique_images.keys():
                    image_results[img_path] = (True, "Error during filtering")

        # Build final results with filtered images
        for chunk_idx, chunk in enumerate(chunks):
            content_passed = content_results[chunk_idx]

            if not content_passed:
                results.append((chunk, False))
            elif chunk.is_multimodal and chunk.images:
                # Filter images based on batch results
                filtered_chunk = Chunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    code_blocks=chunk.code_blocks,
                    images=[],
                    source_path=chunk.source_path,
                    metadata=chunk.metadata,
                    previous_chunk_text=chunk.previous_chunk_text,
                    next_chunk_text=chunk.next_chunk_text,
                    all_document_code=chunk.all_document_code,
                )

                for img in chunk.images:
                    if img.transcription and img.resolved_path:
                        passed, _ = image_results.get(img.resolved_path, (True, "Not filtered"))
                        if passed:
                            filtered_chunk.images.append(img)
                    else:
                        # Keep images without transcription/resolution
                        filtered_chunk.images.append(img)

                results.append((filtered_chunk, True))
            else:
                results.append((chunk, True))

            if (chunk_idx + 1) % batch_size == 0 and checkpoint_callback:
                checkpoint_callback(results, debug_info)

        return results, debug_info
