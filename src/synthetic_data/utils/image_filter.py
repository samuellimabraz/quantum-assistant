"""Image quality filtering after transcription."""

import asyncio
import logging
import re

from synthetic_data.models import LLMClient, Message
from synthetic_data.parsers.base import Document, ImageReference

logger = logging.getLogger(__name__)


class ImageQualityFilter:
    """Filter low-quality images and remove their references from content."""

    def __init__(self, llm_client: LLMClient, max_concurrent: int = 20):
        """
        Initialize image quality filter.

        Args:
            llm_client: LLM client for image quality checks (handles retries internally)
            max_concurrent: Maximum concurrent API requests
        """
        self.llm_client = llm_client
        self.max_concurrent = max_concurrent

    def filter_document_images(
        self,
        document: Document,
        prompt_template: str,
        system_prompt: str | None = None,
    ) -> tuple[Document, list[ImageReference], list[dict]]:
        """
        Filter images in a document (sync wrapper).

        Args:
            document: Document with transcribed images
            prompt_template: Prompt template for quality check
            system_prompt: Optional system prompt

        Returns:
            Tuple of (filtered_document, removed_images, debug_info)
        """
        return asyncio.run(
            self.filter_document_images_async(document, prompt_template, system_prompt)
        )

    async def filter_document_images_async(
        self,
        document: Document,
        prompt_template: str,
        system_prompt: str | None = None,
    ) -> tuple[Document, list[ImageReference], list[dict]]:
        """
        Filter images in a document and remove references from content.

        Args:
            document: Document with transcribed images
            prompt_template: Prompt template for quality check
            system_prompt: Optional system prompt

        Returns:
            Tuple of (filtered_document, removed_images, debug_info)
        """
        if not document.images:
            return document, [], []

        # Filter images
        passed_images = []
        removed_images = []
        debug_info = []

        system_content = (
            system_prompt
            if system_prompt
            else "You are an image quality evaluator. Respond with only 'yes' or 'no'."
        )

        # Check each image
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def check_image(img_ref: ImageReference) -> tuple[ImageReference, bool, str]:
            """Check if image passes quality filter."""
            # Skip if no transcription or resolved path
            if not img_ref.transcription or not img_ref.resolved_path:
                return img_ref, False, "No transcription or resolved path"

            # Build context for quality check
            context_parts = []
            if img_ref.alt_text:
                context_parts.append(f"Alt text: {img_ref.alt_text}")
            if img_ref.caption:
                context_parts.append(f"Caption: {img_ref.caption}")

            context = "\n".join(context_parts) if context_parts else "No additional context"

            user_prompt = prompt_template.format(
                transcription=img_ref.transcription[:2000],
                context=context,
            )

            messages = [
                Message(role="system", content=system_content),
                Message(role="user", content=user_prompt),
            ]

            async with semaphore:
                try:
                    # Client handles retries and timeout internally
                    response = await self.llm_client.generate_async(
                        messages, max_tokens=1024, temperature=0.1, request_timeout=60.0
                    )
                    passed, reason = self._parse_filter_response(response)
                    return img_ref, passed, reason

                except ValueError as e:
                    # Parsing error after retries - reject image
                    error_msg = str(e)
                    logger.warning("Parse error for image %s: %s", img_ref.image_id, error_msg)
                    return img_ref, False, f"Invalid response: {error_msg}"

                except (asyncio.TimeoutError, Exception) as e:
                    # Timeout or API error after retries - reject image
                    logger.error("Error for image %s: %s", img_ref.image_id, str(e))
                    return img_ref, False, f"Error after retries: {type(e).__name__}"

        # Run all checks concurrently
        tasks = [check_image(img) for img in document.images]
        results = await asyncio.gather(*tasks)

        # Process results
        for img_ref, passed, reason in results:
            debug_info.append(
                {
                    "image_id": img_ref.image_id,
                    "image_path": img_ref.path,
                    "resolved_path": img_ref.resolved_path,
                    "decision": "PASS" if passed else "REJECT",
                    "reason": reason,
                    "transcription_preview": (
                        img_ref.transcription[:200] if img_ref.transcription else None
                    ),
                }
            )

            if passed:
                passed_images.append(img_ref)
            else:
                removed_images.append(img_ref)

        # Remove image markers from content for rejected images
        filtered_content = document.content
        for img_ref in removed_images:
            if img_ref.image_id:
                marker = f"[IMAGE:{img_ref.image_id}]"
                filtered_content = filtered_content.replace(marker, "")

        # Clean up extra whitespace
        filtered_content = re.sub(r"\n\n\n+", "\n\n", filtered_content)

        # Create filtered document
        filtered_document = Document(
            source_path=document.source_path,
            title=document.title,
            content=filtered_content,
            code_blocks=document.code_blocks,
            images=passed_images,
            metadata=document.metadata,
        )

        return filtered_document, removed_images, debug_info

    async def filter_documents_batch_async(
        self,
        documents: list[Document],
        prompt_template: str,
        system_prompt: str | None = None,
        progress_callback=None,
        checkpoint_callback=None,
        checkpoint_interval: int = 100,
        skip_image_ids: set[str] | None = None,
    ) -> tuple[list[Document], list[dict]]:
        """
        Filter images across multiple documents

        Args:
            documents: List of documents with transcribed images
            prompt_template: Prompt template for quality check
            system_prompt: Optional system prompt
            progress_callback: Optional callback function(completed_image_count)
            checkpoint_callback: Optional callback function(documents, debug_info)
            checkpoint_interval: Save checkpoint every N images completed
            skip_image_ids: Set of image IDs to skip (already processed)

        Returns:
            Tuple of (filtered_documents, debug_info)
        """
        skip_image_ids = skip_image_ids or set()
        system_content = (
            system_prompt
            if system_prompt
            else "You are an image quality evaluator. Respond with only 'yes' or 'no'."
        )

        # Collect all images from all documents with their document reference
        # Skip images that were already processed
        all_items = []
        for doc_idx, doc in enumerate(documents):
            for img_idx, img_ref in enumerate(doc.images):
                if img_ref.transcription and img_ref.resolved_path:
                    # Skip if already processed
                    if img_ref.image_id and img_ref.image_id in skip_image_ids:
                        continue
                    all_items.append((doc_idx, img_idx, img_ref))

        if not all_items:
            # No images to filter, return documents as-is
            return documents, []

        # Track filter decisions per image
        # Key: (doc_idx, img_idx) -> (passed, reason, debug_entry)
        filter_results: dict[tuple[int, int], tuple[bool, str, dict]] = {}

        completed_count = [0]
        last_checkpoint = [0]
        lock = asyncio.Lock()
        all_debug_info = []

        async def process_one(item: tuple) -> None:
            """Process a single image filter check."""
            doc_idx, img_idx, img_ref = item

            # Skip if no transcription or resolved path
            if not img_ref.transcription or not img_ref.resolved_path:
                passed, reason = False, "No transcription or resolved path"
            else:
                # Build context for quality check
                context_parts = []
                if img_ref.alt_text:
                    context_parts.append(f"Alt text: {img_ref.alt_text}")
                if img_ref.caption:
                    context_parts.append(f"Caption: {img_ref.caption}")

                context = "\n".join(context_parts) if context_parts else "No additional context"

                user_prompt = prompt_template.format(
                    transcription=img_ref.transcription[:2000],
                    context=context,
                )

                messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content=user_prompt),
                ]

                try:
                    # Client handles retries and timeout internally
                    response = await self.llm_client.generate_async(
                        messages, max_tokens=1024, temperature=0.1, request_timeout=60.0
                    )
                    passed, reason = self._parse_filter_response(response)

                except ValueError as e:
                    # Parsing error after retries - reject image
                    error_msg = str(e)
                    logger.warning("Parse error for image %s: %s", img_ref.image_id, error_msg)
                    passed, reason = False, f"Invalid response: {error_msg}"

                except (asyncio.TimeoutError, Exception) as e:
                    # Timeout or API error after retries - reject image
                    logger.error("Error for image %s: %s", img_ref.image_id, str(e))
                    passed, reason = False, f"Error after retries: {type(e).__name__}"

            debug_entry = {
                "image_id": img_ref.image_id,
                "image_path": img_ref.path,
                "resolved_path": img_ref.resolved_path,
                "decision": "PASS" if passed else "REJECT",
                "reason": reason,
                "transcription_preview": (
                    img_ref.transcription[:200] if img_ref.transcription else None
                ),
            }

            async with lock:
                filter_results[(doc_idx, img_idx)] = (passed, reason, debug_entry)
                all_debug_info.append(debug_entry)
                completed_count[0] += 1

                if progress_callback:
                    progress_callback(completed_count[0])

                # Save checkpoint periodically
                if checkpoint_callback and (
                    completed_count[0] - last_checkpoint[0] >= checkpoint_interval
                ):
                    # Build partial filtered documents for checkpoint
                    partial_docs = self._build_filtered_documents(
                        documents, filter_results, skip_image_ids
                    )
                    checkpoint_callback(partial_docs, all_debug_info)
                    last_checkpoint[0] = completed_count[0]

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_limit(item: tuple) -> None:
            async with semaphore:
                await process_one(item)

        # Process all images concurrently with timeout protection
        tasks = [process_with_limit(item) for item in all_items]
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logger.error("Error in batch processing: %s", str(e))
            # Continue to build results from what we have

        # Build final filtered documents
        filtered_documents = self._build_filtered_documents(
            documents, filter_results, skip_image_ids
        )

        # Final checkpoint
        if checkpoint_callback:
            checkpoint_callback(filtered_documents, all_debug_info)

        # Add summary
        total_images = sum(len(doc.images) for doc in documents)
        filtered_images = sum(len(doc.images) for doc in filtered_documents)
        removed_count = total_images - filtered_images

        all_debug_info.append(
            {
                "type": "summary",
                "total_images_before": total_images,
                "total_images_after": filtered_images,
                "images_removed": removed_count,
            }
        )

        return filtered_documents, all_debug_info

    def _build_filtered_documents(
        self,
        documents: list[Document],
        filter_results: dict[tuple[int, int], tuple[bool, str, dict]],
        skip_image_ids: set[str],
    ) -> list[Document]:
        """Build filtered documents based on filter results.

        Args:
            documents: Original documents
            filter_results: Dict of (doc_idx, img_idx) -> (passed, reason, debug_entry)
            skip_image_ids: Set of image IDs that were already processed (keep as-is)

        Returns:
            List of filtered documents with rejected images removed
        """
        filtered_documents = []

        for doc_idx, doc in enumerate(documents):
            passed_images = []
            removed_images = []

            for img_idx, img_ref in enumerate(doc.images):
                key = (doc_idx, img_idx)
                if key in filter_results:
                    # Newly processed in this run
                    passed, _, _ = filter_results[key]
                    if passed:
                        passed_images.append(img_ref)
                    else:
                        removed_images.append(img_ref)
                elif img_ref.image_id and img_ref.image_id in skip_image_ids:
                    # Already processed in previous run - keep it
                    passed_images.append(img_ref)
                else:
                    # Image wasn't processed (no transcription/path) - remove it
                    removed_images.append(img_ref)

            # Remove image markers from content for rejected images
            filtered_content = doc.content
            for img_ref in removed_images:
                if img_ref.image_id:
                    marker = f"[IMAGE:{img_ref.image_id}]"
                    filtered_content = filtered_content.replace(marker, "")

            # Clean up extra whitespace
            filtered_content = re.sub(r"\n\n\n+", "\n\n", filtered_content)

            # Create filtered document
            filtered_doc = Document(
                source_path=doc.source_path,
                title=doc.title,
                content=filtered_content,
                code_blocks=doc.code_blocks,
                images=passed_images,
                metadata=doc.metadata,
            )
            filtered_documents.append(filtered_doc)

        return filtered_documents

    def _parse_filter_response(self, response: str) -> tuple[bool, str]:
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
                # Remove any remaining prefix artifacts
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
                logger.warning(
                    "Could not parse decision from response. First 200 chars: %s", response[:200]
                )
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
