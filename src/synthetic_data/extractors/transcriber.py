"""Image transcription using vision models with async batch support."""

import asyncio
import hashlib
from pathlib import Path

from synthetic_data.models import VLMClient
from synthetic_data.parsers.base import Document, ImageReference, ImageType


class ImageTranscriber:
    """Transcribe images using VLM with async batch processing for high throughput."""

    def __init__(
        self,
        vlm_client: VLMClient,
        transcription_prompt: str,
        system_prompt: str = "",
        batch_size: int = 16,
        max_concurrent: int = 16,
        enable_classification: bool = True,
    ):
        """
        Initialize image transcriber.

        Args:
            vlm_client: Vision-language model client
            transcription_prompt: Prompt template for image transcription
            system_prompt: System prompt for transcription
            batch_size: Number of images to process in parallel batches
            max_concurrent: Maximum concurrent VLM requests
            enable_classification: Whether to classify images by type
        """
        self.vlm_client = vlm_client
        self.transcription_prompt = transcription_prompt
        self.system_prompt = system_prompt
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.enable_classification = enable_classification

    def transcribe_document_images(self, document: Document) -> None:
        """
        Transcribe all images in a document in-place (sync).

        Args:
            document: Document with images to transcribe
        """
        for img_ref in document.images:
            if img_ref.resolved_path:
                try:
                    transcription = self._transcribe_image(img_ref)
                    img_ref.transcription = transcription
                except Exception as e:
                    print(f"Warning: Failed to transcribe {img_ref.path}: {e}")
                    img_ref.transcription = None

    async def transcribe_document_images_async(
        self, document: Document, progress_callback=None
    ) -> None:
        """
        Transcribe all images in a document in-place using async batching.

        Args:
            document: Document with images to transcribe
            progress_callback: Optional callback function(completed_count) for progress tracking
        """
        images_to_transcribe = [img for img in document.images if img.resolved_path]

        if not images_to_transcribe:
            return

        # Prepare batch of (prompt, image_path) tuples
        prompts_and_images = []
        for img_ref in images_to_transcribe:
            try:
                prompt = self._prepare_prompt(img_ref)
                image_path = Path(img_ref.resolved_path).resolve()
                if image_path.exists():
                    prompts_and_images.append((img_ref, prompt, image_path))
            except Exception as e:
                print(f"Warning: Failed to prepare image {img_ref.path}: {e}")
                img_ref.transcription = None

        if not prompts_and_images:
            return

        # Collect all prompts for single batch call
        batch_prompts = [(prompt, img_path) for _, prompt, img_path in prompts_and_images]

        try:
            transcriptions = await self.vlm_client.generate_batch_with_images_async(
                batch_prompts,
                system_prompt=self.system_prompt,
                max_concurrent=self.max_concurrent,
                progress_callback=progress_callback,
            )

            # Assign transcriptions back to image references
            for i, transcription in enumerate(transcriptions):
                img_ref = prompts_and_images[i][0]
                img_ref.transcription = transcription.strip()

                # Generate unique ID if not present
                if not img_ref.image_id:
                    img_ref.image_id = self._generate_image_id(img_ref)
        except Exception as e:
            print(f"Warning: Batch transcription failed: {e}")

    def transcribe_batch_documents(self, documents: list[Document]) -> None:
        """
        Transcribe images across multiple documents using async batching (sync wrapper).

        Args:
            documents: List of documents with images to transcribe
        """
        asyncio.run(self.transcribe_batch_documents_async(documents))

    async def transcribe_batch_documents_async(
        self,
        documents: list[Document],
        progress_callback=None,
        checkpoint_callback=None,
        checkpoint_interval: int = 50,
        skip_image_ids: set[str] | None = None,
    ) -> None:
        """
        Transcribe images across multiple documents with full parallelization.

        Uses pipelined processing:
        - All images are processed and API calls made concurrently
        - Checkpoints are saved periodically based on completed count
        - Skips images already transcribed in previous runs

        Args:
            documents: List of documents with images to transcribe
            progress_callback: Optional callback function(completed_count) for progress tracking
            checkpoint_callback: Optional callback function(documents) for checkpoint saving
            checkpoint_interval: Save checkpoint every N images completed
            skip_image_ids: Set of image IDs to skip (already transcribed)
        """
        skip_image_ids = skip_image_ids or set()
        
        # Collect all images from all documents
        all_items = []

        for doc in documents:
            for img_ref in doc.images:
                # Skip if already transcribed or in skip set
                if img_ref.transcription or (img_ref.image_id and img_ref.image_id in skip_image_ids):
                    continue
                if not img_ref.resolved_path:
                    continue
                    
                try:
                    prompt = self._prepare_prompt(img_ref)
                    image_path = Path(img_ref.resolved_path).resolve()
                    if image_path.exists():
                        all_items.append((img_ref, prompt, image_path))
                except Exception as e:
                    print(f"Warning: Failed to prepare image {img_ref.path}: {e}")
                    img_ref.transcription = None

        if not all_items:
            return

        completed_count = [0]
        last_checkpoint = [0]
        lock = asyncio.Lock()

        async def process_one(item: tuple) -> None:
            """Process a single image with transcription and classification."""
            img_ref, prompt, image_path = item

            try:
                transcription = await self.vlm_client.generate_with_image_async(
                    prompt, image_path, system_prompt=self.system_prompt
                )
                img_ref.transcription = transcription.strip() if transcription else None

                if img_ref.transcription:
                    if not img_ref.image_id:
                        img_ref.image_id = self._generate_image_id(img_ref)

                    if self.enable_classification:
                        await self._classify_single_image_async(img_ref)

            except ValueError as e:
                print(f"Skipping {image_path.name}: {e}")
                img_ref.transcription = None
            except Exception as e:
                err_type = type(e).__name__
                print(f"Warning: Failed to transcribe {image_path.name}: {err_type}: {e}")
                img_ref.transcription = None

            async with lock:
                completed_count[0] += 1
                if progress_callback:
                    progress_callback(completed_count[0])

                # Save checkpoint periodically
                if checkpoint_callback and (
                    completed_count[0] - last_checkpoint[0] >= checkpoint_interval
                ):
                    checkpoint_callback(documents)
                    last_checkpoint[0] = completed_count[0]

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_limit(item: tuple) -> None:
            async with semaphore:
                await process_one(item)

        tasks = [process_with_limit(item) for item in all_items]
        await asyncio.gather(*tasks)

        # Final checkpoint
        if checkpoint_callback:
            checkpoint_callback(documents)

    async def _classify_single_image_async(self, img_ref: ImageReference) -> None:
        """Classify a single image by type using its transcription."""
        if not img_ref.transcription:
            return

        try:
            from synthetic_data.models import Message

            prompt = self._get_classification_prompt(img_ref.transcription)

            messages = [
                Message(
                    role="system",
                    content="You are an image classifier. Respond with ONLY the category name.",
                ),
                Message(role="user", content=prompt),
            ]

            response = await self.vlm_client.generate_async(messages, temperature=0.1)
            img_ref.image_type = self._parse_image_type(response.strip())

        except Exception:
            # Fallback to heuristic classification
            img_ref.image_type = self._heuristic_classify(img_ref.transcription)

    def _prepare_prompt(self, img_ref: ImageReference) -> str:
        """Prepare transcription prompt with context including code references.

        Provides rich context to improve transcription accuracy:
        - Alt text and caption for image identification
        - Surrounding text context for interpretation
        - Code blocks that may have generated the image (critical for output images)
        """
        context_parts = []

        # Basic image identification
        if img_ref.alt_text:
            context_parts.append(f"Alt text: {img_ref.alt_text}")
        if img_ref.caption:
            context_parts.append(f"Caption: {img_ref.caption}")

        # Surrounding text context (expanded from 300 to 600 chars for better understanding)
        if img_ref.context:
            context_parts.append(f"Surrounding text: {img_ref.context[:600]}")

        # Code context - critical for images that are outputs of code execution
        # This helps the transcriber understand what the visualization represents
        if hasattr(img_ref, "code_context") and img_ref.code_context:
            code_preview = img_ref.code_context[:2000]
            context_parts.append(
                f"Code that may have generated this image:\n```python\n{code_preview}\n```"
            )

        context_info = (
            "\n".join(context_parts) if context_parts else "No additional context available"
        )
        return self.transcription_prompt.format(context=context_info)

    def _transcribe_image(self, img_ref: ImageReference) -> str:
        """
        Generate detailed transcription for a single image (sync fallback).

        Args:
            img_ref: Image reference with resolved path

        Returns:
            Detailed description of the image content
        """
        image_path = Path(img_ref.resolved_path).resolve()

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        prompt = self._prepare_prompt(img_ref)

        try:
            response = self.vlm_client.generate_with_image(
                prompt, image_path, system_prompt=self.system_prompt
            )
            return response.strip()
        except Exception as e:
            raise RuntimeError(f"VLM transcription failed: {e}")

    def _generate_image_id(self, img_ref: ImageReference) -> str:
        """Generate unique ID for an image reference."""
        if img_ref.resolved_path:
            path_hash = hashlib.md5(img_ref.resolved_path.encode()).hexdigest()[:12]
            return f"img_{path_hash}"
        elif img_ref.path:
            path_hash = hashlib.md5(img_ref.path.encode()).hexdigest()[:12]
            return f"img_{path_hash}"
        else:
            random_hash = hashlib.md5(str(id(img_ref)).encode()).hexdigest()[:12]
            return f"img_{random_hash}"

    async def _classify_batch_images_async(
        self, batch_items: list[tuple[ImageReference, str, Path]]
    ) -> None:
        """Classify a batch of images by type using their transcriptions.

        Args:
            batch_items: List of (img_ref, prompt, image_path) tuples with transcriptions already set
        """
        if not batch_items:
            return

        classification_prompts = []
        for img_ref, _, _ in batch_items:
            if not img_ref.transcription:
                continue

            prompt = self._get_classification_prompt(img_ref.transcription)
            classification_prompts.append((img_ref, prompt))

        if not classification_prompts:
            return

        # Batch classify
        try:
            from synthetic_data.models import Message

            message_batches = [
                [
                    Message(
                        role="system",
                        content="You are an image classifier. Respond with ONLY the category name.",
                    ),
                    Message(role="user", content=prompt),
                ]
                for _, prompt in classification_prompts
            ]

            responses = await self.vlm_client.generate_batch_async(
                message_batches,
                max_concurrent=self.max_concurrent,
                temperature=0.1,
            )

            for i, response in enumerate(responses):
                img_ref = classification_prompts[i][0]
                img_type = self._parse_image_type(response.strip())
                img_ref.image_type = img_type

        except Exception as e:
            print(f"Warning: Batch image classification failed: {e}")
            # Fallback to heuristic classification
            for img_ref, _ in classification_prompts:
                img_ref.image_type = self._heuristic_classify(img_ref.transcription)

    def _get_classification_prompt(self, transcription: str) -> str:
        """Get prompt for classifying an image by its transcription."""
        return f"""Classify this quantum computing image into ONE of these categories:
- circuit: Quantum circuit diagrams with gates and qubits
- chart: Histograms, bar charts, line plots, measurement results
- bloch_sphere: Bloch sphere visualizations
- formula: Mathematical formulas and equations
- diagram: Technical diagrams, flowcharts, architecture diagrams
- table: Tables and structured data
- code_output: Code execution output visualizations
- decorative: Decorative or non-essential images (logos, icons, backgrounds)
- unknown: Cannot determine

Image description:
{transcription[:500]}

Respond with ONLY the category name (lowercase)."""

    def _parse_image_type(self, response: str) -> ImageType:
        """Parse image type from model response."""
        response_lower = response.lower().strip()

        # Direct mapping
        type_mapping = {
            "circuit": ImageType.CIRCUIT,
            "chart": ImageType.CHART,
            "bloch_sphere": ImageType.BLOCH_SPHERE,
            "formula": ImageType.FORMULA,
            "diagram": ImageType.DIAGRAM,
            "table": ImageType.TABLE,
            "code_output": ImageType.CODE_OUTPUT,
            "decorative": ImageType.DECORATIVE,
            "unknown": ImageType.UNKNOWN,
        }

        # Try exact match
        for key, img_type in type_mapping.items():
            if key in response_lower:
                return img_type

        # Fallback to heuristic
        return self._heuristic_classify(response_lower)

    def _heuristic_classify(self, transcription: str) -> ImageType:
        """Heuristic classification based on keywords in transcription."""
        transcription_lower = transcription.lower()

        # Circuit keywords
        if any(
            kw in transcription_lower
            for kw in [
                "circuit",
                "gate",
                "qubit",
                "quantum circuit",
                "hadamard",
                "cnot",
                "cx gate",
                "measurement",
                "barrier",
                "wire",
            ]
        ):
            return ImageType.CIRCUIT

        # Chart keywords
        if any(
            kw in transcription_lower
            for kw in [
                "histogram",
                "bar chart",
                "plot",
                "graph",
                "axis",
                "measurement result",
                "probability",
                "count",
                "distribution",
                "x-axis",
                "y-axis",
            ]
        ):
            return ImageType.CHART

        # Bloch sphere keywords
        if any(
            kw in transcription_lower
            for kw in ["bloch sphere", "bloch", "sphere", "state vector", "qubit state"]
        ):
            return ImageType.BLOCH_SPHERE

        # Formula keywords
        if any(
            kw in transcription_lower
            for kw in [
                "equation",
                "formula",
                "mathematical expression",
                "integral",
                "summation",
                "matrix",
                "operator",
                "hamiltonian",
            ]
        ):
            return ImageType.FORMULA

        # Table keywords
        if any(
            kw in transcription_lower for kw in ["table", "row", "column", "cell", "data table"]
        ):
            return ImageType.TABLE

        # Code output keywords
        if any(
            kw in transcription_lower
            for kw in ["output", "result", "execution", "console", "print"]
        ):
            return ImageType.CODE_OUTPUT

        # Decorative keywords
        if any(
            kw in transcription_lower
            for kw in ["logo", "icon", "decorative", "background", "banner"]
        ):
            return ImageType.DECORATIVE

        return ImageType.UNKNOWN
