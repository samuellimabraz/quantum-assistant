"""Content quality filtering with optimized async processing."""

import asyncio
from typing import List, Tuple

from synthetic_data.extractors.chunker import Chunk
from synthetic_data.models import LLMClient, Message
from synthetic_data.parsers.base import ImageReference


class QualityFilter:
    """Filter low-quality content chunks and images with pipelined async processing."""

    def __init__(self, llm_client: LLMClient, max_concurrent: int = 20):
        """
        Initialize quality filter.

        Args:
            llm_client: LLM client for content and image quality checks
            max_concurrent: Maximum concurrent API requests
        """
        self.llm_client = llm_client
        self.max_concurrent = max_concurrent
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
        content_system_prompt: str | None = None,
        batch_size: int = 10,
        max_concurrent: int | None = None,
        progress_callback=None,
        checkpoint_callback=None,
        checkpoint_interval: int = 50,
        save_debug: bool = True,
    ) -> Tuple[List[Tuple[Chunk, bool]], List[dict]]:
        """
        Filter chunks for content quality

        Args:
            chunks: List of chunks to filter
            content_prompt: Prompt template for content quality
            content_system_prompt: Optional system prompt for content filtering
            batch_size: Deprecated, kept for compatibility
            max_concurrent: Max concurrent requests (uses instance default if None)
            progress_callback: Optional callback function(completed_count)
            checkpoint_callback: Optional callback function(results, debug_info)
            checkpoint_interval: Save checkpoint every N completed items
            save_debug: Whether to save debug information

        Returns:
            Tuple of (List of (chunk, passed_filter) tuples, debug_info list)
        """
        max_concurrent = max_concurrent or self.max_concurrent
        debug_info = []
        content_results = [None] * len(chunks)
        self._last_debug_info = []

        content_system = content_system_prompt or (
            "You are a content quality evaluator. Respond with only 'yes' or 'no'."
        )

        # Filter content concurrently
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = [0]
        last_checkpoint = [0]
        lock = asyncio.Lock()

        async def check_content(idx: int, chunk: Chunk) -> None:
            """Check content quality for a single chunk."""
            if len(chunk.text) < 50:
                content_results[idx] = (False, "Content too short")
                return

            messages = [
                Message(role="system", content=content_system),
                Message(role="user", content=content_prompt.format(content=chunk.text)),
            ]

            async with semaphore:
                try:
                    response = await self.llm_client.generate_async(
                        messages, max_tokens=1024, temperature=0.1
                    )
                    passed, reason = self._parse_filter_response(response)
                    content_results[idx] = (passed, reason)

                    if save_debug:
                        async with lock:
                            debug_info.append(
                                {
                                    "type": "content",
                                    "chunk_id": chunk.chunk_id,
                                    "source": str(chunk.source_path),
                                    "decision": "PASS" if passed else "REJECT",
                                    "reason": reason,
                                    "content_preview": chunk.text[:200],
                                    "has_images": len(chunk.images) > 0,
                                    "full_response": response,
                                }
                            )
                except Exception as e:
                    content_results[idx] = (True, f"Error: {e}")

            async with lock:
                completed[0] += 1
                if progress_callback:
                    progress_callback(completed[0])

        # Run all content checks concurrently
        content_tasks = [check_content(i, chunk) for i, chunk in enumerate(chunks)]
        await asyncio.gather(*content_tasks)

        # Build final results - keep chunks that passed content filter
        results = []
        for idx, chunk in enumerate(chunks):
            passed, _ = content_results[idx] or (False, "")
            results.append((chunk, passed))

        # Summary
        chunks_passed = sum(1 for _, passed in results if passed)
        chunks_rejected = len(chunks) - chunks_passed

        debug_info.append(
            {
                "type": "summary",
                "chunks_evaluated": len(chunks),
                "chunks_passed": chunks_passed,
                "chunks_rejected": chunks_rejected,
            }
        )

        if checkpoint_callback:
            checkpoint_callback(results, debug_info)

        self._last_debug_info = debug_info
        return results, debug_info
