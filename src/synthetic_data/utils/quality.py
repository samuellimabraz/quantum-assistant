"""Content quality filtering with optimized async processing."""

import asyncio
import logging
from typing import List, Tuple

from synthetic_data.extractors.chunker import Chunk
from synthetic_data.models import LLMClient, Message
from synthetic_data.parsers.base import ImageReference

logger = logging.getLogger(__name__)


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
        Parse filter response in expected format: DECISION: yes/no\\nREASON: explanation

        Args:
            response: Model response

        Returns:
            Tuple of (passed, reason)

        Raises:
            ValueError: If response format is invalid
        """
        response = response.strip()
        if not response:
            logger.warning("Received empty response from model")
            raise ValueError("Empty response from model")

        lines = [line.strip() for line in response.split("\n") if line.strip()]

        decision = None
        reason = None

        # Parse each line looking for DECISION and REASON
        for line in lines:
            line_lower = line.lower()

            # Match "DECISION: yes/no" or "Decision: yes/no"
            if line_lower.startswith("decision:") or line_lower.startswith("dec:"):
                decision_value = line.split(":", 1)[1].strip().lower()
                if "yes" in decision_value:
                    decision = True
                elif "no" in decision_value:
                    decision = False

            # Match "REASON: explanation" or "Reason: explanation"
            elif line_lower.startswith("reason:"):
                reason = line.split(":", 1)[1].strip()
                if reason and not reason[0].isalnum():
                    reason = reason.lstrip(":- ")

        # Validation
        if decision is None:
            # Try simple yes/no at start of response
            first_word = lines[0].lower().split()[0] if lines else ""
            if first_word == "yes":
                decision = True
                reason = response if not reason else reason
            elif first_word == "no":
                decision = False
                reason = response if not reason else reason
            else:
                logger.warning("Could not parse decision from response. First 200 chars: %s", response[:200])
                raise ValueError(f"Could not parse decision from response: {response[:200]}")

        if not reason or len(reason.strip()) < 3:
            # Extract full response as reason if not found
            reason = (
                " ".join(lines[1:])
                if len(lines) > 1
                else "Model provided decision without explanation"
            )
            logger.debug("Reason not found in expected format, using fallback: %s", reason[:100])

        return decision, reason

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
        skip_chunk_ids: set[int] | None = None,
        cached_results: dict[int, bool] | None = None,
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
            skip_chunk_ids: Set of chunk IDs to skip (already processed)
            cached_results: Dict of chunk_id -> passed for already-processed chunks
            save_debug: Whether to save debug information

        Returns:
            Tuple of (List of (chunk, passed_filter) tuples, debug_info list)
        """
        max_concurrent = max_concurrent or self.max_concurrent
        skip_chunk_ids = skip_chunk_ids or set()
        cached_results = cached_results or {}
        
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
            # Use cached result if available
            if chunk.chunk_id in cached_results:
                content_results[idx] = (cached_results[chunk.chunk_id], "Cached from checkpoint")
                async with lock:
                    completed[0] += 1
                    if progress_callback:
                        progress_callback(completed[0])
                return

            # Skip if in skip set
            if chunk.chunk_id in skip_chunk_ids:
                content_results[idx] = (True, "Already processed in previous run")
                async with lock:
                    completed[0] += 1
                    if progress_callback:
                        progress_callback(completed[0])
                return

            if len(chunk.text) < 50:
                content_results[idx] = (False, "Content too short")
                async with lock:
                    completed[0] += 1
                    if progress_callback:
                        progress_callback(completed[0])
                return

            messages = [
                Message(role="system", content=content_system),
                Message(role="user", content=content_prompt.format(content=chunk.text)),
            ]

            async with semaphore:
                try:
                    # Client handles retries and timeout internally
                    response = await self.llm_client.generate_async(
                        messages, max_tokens=1024, temperature=0.1, request_timeout=60.0
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
                                }
                            )

                except ValueError as e:
                    # Parsing error after retries - reject chunk
                    error_msg = str(e)
                    logger.warning("Parse error for chunk %s: %s", chunk.chunk_id, error_msg)
                    content_results[idx] = (False, f"Invalid response: {error_msg}")

                except (asyncio.TimeoutError, Exception) as e:
                    # Timeout or API error after retries - reject chunk
                    logger.error("Error for chunk %s: %s", chunk.chunk_id, str(e))
                    content_results[idx] = (False, f"Error after retries: {type(e).__name__}")

            async with lock:
                completed[0] += 1
                if progress_callback:
                    progress_callback(completed[0])

                # Save checkpoint periodically
                if checkpoint_callback and (
                    completed[0] - last_checkpoint[0] >= checkpoint_interval
                ):
                    # Build partial results for checkpoint
                    partial_results = [
                        (chunks[i], content_results[i][0] if content_results[i] else False)
                        for i in range(len(chunks))
                        if content_results[i] is not None
                    ]
                    checkpoint_callback(partial_results, debug_info)
                    last_checkpoint[0] = completed[0]

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

        # Final checkpoint
        if checkpoint_callback:
            checkpoint_callback(results, debug_info)

        self._last_debug_info = debug_info
        return results, debug_info
