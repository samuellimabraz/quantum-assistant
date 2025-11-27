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
        self._last_debug_info = []  # Store debug info for external access

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
        self._last_debug_info = []  # Reset for this batch

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
                                chunk = chunks[chunk_idx]
                                debug_info.append(
                                    {
                                        "type": "content",
                                        "chunk_id": chunk.chunk_id,
                                        "source": str(chunk.source_path),
                                        "decision": "PASS" if passed else "REJECT",
                                        "reason": reason,
                                        "content_preview": chunk.text[:200],
                                        "has_images": len(chunk.images) > 0,
                                        "image_ids": [
                                            img.image_id for img in chunk.images if img.image_id
                                        ],
                                        "full_response": response,
                                    }
                                )

                            response_idx += 1
                except Exception as e:
                    print(f"Batch content quality check failed: {e}")
                    content_results.extend([True] * len(batch))
            else:
                content_results.extend([False] * len(batch))

        unique_images = {}  # image_path -> (ImageReference, chunk_text, chunk_idx)
        images_skipped_no_content_pass = 0
        images_skipped_no_transcription = 0
        images_skipped_no_resolved_path = 0

        for chunk_idx, chunk in enumerate(chunks):
            if not content_results[chunk_idx]:
                # Content failed - count images that won't be evaluated
                for img in chunk.images:
                    if img.transcription and img.resolved_path:
                        images_skipped_no_content_pass += 1
                continue

            if chunk.is_multimodal and chunk.images:
                for img in chunk.images:
                    if not img.transcription:
                        images_skipped_no_transcription += 1
                        continue
                    if not img.resolved_path:
                        images_skipped_no_resolved_path += 1
                        continue

                    if img.resolved_path not in unique_images:
                        context_preview = chunk.text[:500] if len(chunk.text) > 500 else chunk.text
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
                    context_parts.append(f"Alt text: {img.alt_text[:200]}")
                if img.caption:
                    context_parts.append(f"Caption: {img.caption[:200]}")
                if img.context:
                    # Limit context to prevent API errors (some contexts are huge)
                    context_limited = img.context[:500] if len(img.context) > 500 else img.context
                    context_parts.append(f"Image context: {context_limited}")

                context_parts.append(f"Surrounding text: {chunk_context}")

                context = "\n".join(context_parts) if context_parts else "No additional context"

                # Limit transcription to prevent API errors
                transcription_limited = (
                    img.transcription[:2000] if len(img.transcription) > 2000 else img.transcription
                )

                user_prompt = image_prompt.format(
                    transcription=transcription_limited,
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
                    max_tokens=2048,
                    temperature=0.1,
                    max_concurrent=max_concurrent,
                )

                for idx, (img_path, img, chunk_idx) in enumerate(image_paths):
                    response = image_responses[idx]

                    if response is None or (
                        isinstance(response, str) and response.startswith("ERROR:")
                    ):
                        if save_debug:
                            debug_info.append(
                                {
                                    "type": "image_error",
                                    "chunk_id": chunks[chunk_idx].chunk_id,
                                    "source": str(chunks[chunk_idx].source_path),
                                    "image_id": img.image_id,
                                    "image_path": img_path,
                                    "error": "API request failed",
                                    "transcription_length": (
                                        len(img.transcription) if img.transcription else 0
                                    ),
                                    "context_length": len(img.context) if img.context else 0,
                                }
                            )

                        image_results[img_path] = (True, "API error - defaulted to pass")
                        continue

                    passed, reason = self._parse_filter_response(response)
                    image_results[img_path] = (passed, reason)

                    if save_debug:
                        debug_info.append(
                            {
                                "type": "image",
                                "chunk_id": chunks[chunk_idx].chunk_id,
                                "source": str(chunks[chunk_idx].source_path),
                                "image_id": img.image_id,
                                "image_path": img_path,
                                "resolved_path": img.resolved_path,
                                "image_type": img.image_type.value if img.image_type else "unknown",
                                "alt_text": img.alt_text,
                                "decision": "PASS" if passed else "REJECT",
                                "reason": reason,
                                "transcription_preview": (
                                    img.transcription[:200] if img.transcription else None
                                ),
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
                    accumulated_code=chunk.accumulated_code,
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

        # Add summary statistics to debug info
        debug_info.append(
            {
                "type": "summary",
                "images_evaluated": len(unique_images),
                "images_skipped_no_content_pass": images_skipped_no_content_pass,
                "images_skipped_no_transcription": images_skipped_no_transcription,
                "images_skipped_no_resolved_path": images_skipped_no_resolved_path,
            }
        )

        # Store debug info for external access
        self._last_debug_info = debug_info

        return results, debug_info
