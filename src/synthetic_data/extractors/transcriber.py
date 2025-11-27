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
    ) -> None:
        """
        Transcribe images across multiple documents using async batching with checkpoints.

        Args:
            documents: List of documents with images to transcribe
            progress_callback: Optional callback function(completed_count) for progress tracking
            checkpoint_callback: Optional callback function(documents) for checkpoint saving
            checkpoint_interval: Save checkpoint every N images
        """
        # Collect all images from all documents
        all_prompts_and_images = []

        for doc in documents:
            images_to_transcribe = [
                img for img in doc.images if img.resolved_path and not img.transcription
            ]
            for img_ref in images_to_transcribe:
                try:
                    prompt = self._prepare_prompt(img_ref)
                    image_path = Path(img_ref.resolved_path).resolve()
                    if image_path.exists():
                        all_prompts_and_images.append((img_ref, prompt, image_path))
                except Exception as e:
                    print(f"Warning: Failed to prepare image {img_ref.path}: {e}")
                    img_ref.transcription = None

        if not all_prompts_and_images:
            return

        # Process in batches with checkpoints between batches
        completed_count = 0

        try:
            for i in range(0, len(all_prompts_and_images), checkpoint_interval):
                batch_items = all_prompts_and_images[i : i + checkpoint_interval]
                batch_prompts = [(prompt, img_path) for _, prompt, img_path in batch_items]

                # Transcribe this batch
                transcriptions = await self.vlm_client.generate_batch_with_images_async(
                    batch_prompts,
                    system_prompt=self.system_prompt,
                    max_concurrent=self.max_concurrent,
                    progress_callback=None,  # Handle progress manually
                )

                for j, transcription in enumerate(transcriptions):
                    img_ref = batch_items[j][0]
                    img_ref.transcription = transcription.strip()

                    # Generate unique ID if not present
                    if not img_ref.image_id:
                        img_ref.image_id = self._generate_image_id(img_ref)

                    completed_count += 1

                    # Update progress after each image
                    if progress_callback:
                        progress_callback(completed_count)

                # Classify images if enabled
                if self.enable_classification:
                    await self._classify_batch_images_async(batch_items)

                # Save checkpoint after batch is complete and transcriptions are assigned
                if checkpoint_callback:
                    checkpoint_callback(documents)

        except Exception as e:
            print(f"Warning: Batch transcription failed: {e}")
            # Save checkpoint on error with whatever we've completed
            if checkpoint_callback:
                checkpoint_callback(documents)

    def _prepare_prompt(self, img_ref: ImageReference) -> str:
        """Prepare transcription prompt with context."""
        context_info = f"Alt text: {img_ref.alt_text}\n" if img_ref.alt_text else ""
        if img_ref.caption:
            context_info += f"Caption: {img_ref.caption}\n"
        if img_ref.context:
            context_info += f"Context: {img_ref.context[:300]}"

        return self.transcription_prompt.format(context=context_info.strip())

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
            # Use hash of resolved path for consistency
            path_hash = hashlib.md5(img_ref.resolved_path.encode()).hexdigest()[:12]
            return f"img_{path_hash}"
        elif img_ref.path:
            # Fallback to original path
            path_hash = hashlib.md5(img_ref.path.encode()).hexdigest()[:12]
            return f"img_{path_hash}"
        else:
            # Fallback to random ID
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

        # Build classification prompts
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

            # Assign classifications
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
