"""Image transcription using vision models with async batch support."""

import asyncio
from pathlib import Path

from synthetic_data.models import Message, VLMClient
from synthetic_data.parsers.base import Document, ImageReference


class ImageTranscriber:
    """Transcribe images using VLM with async batch processing for high throughput."""

    def __init__(
        self,
        vlm_client: VLMClient,
        transcription_prompt: str,
        batch_size: int = 16,
        max_concurrent: int = 16,
    ):
        """
        Initialize image transcriber.

        Args:
            vlm_client: Vision-language model client
            transcription_prompt: Prompt template for image transcription
            batch_size: Number of images to process in parallel batches
            max_concurrent: Maximum concurrent VLM requests
        """
        self.vlm_client = vlm_client
        self.transcription_prompt = transcription_prompt
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent

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
                max_concurrent=self.max_concurrent,
                progress_callback=progress_callback,
            )

            # Assign transcriptions back to image references
            for i, transcription in enumerate(transcriptions):
                img_ref = prompts_and_images[i][0]
                img_ref.transcription = transcription.strip()
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
        self, documents: list[Document], progress_callback=None
    ) -> None:
        """
        Transcribe images across multiple documents using async batching.

        Args:
            documents: List of documents with images to transcribe
            progress_callback: Optional callback function(completed_count) for progress tracking
        """
        # Collect all images from all documents
        all_prompts_and_images = []
        
        for doc in documents:
            images_to_transcribe = [img for img in doc.images if img.resolved_path]
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

        # Transcribe all images in one batch
        batch_prompts = [(prompt, img_path) for _, prompt, img_path in all_prompts_and_images]
        
        try:
            transcriptions = await self.vlm_client.generate_batch_with_images_async(
                batch_prompts,
                max_concurrent=self.max_concurrent,
                progress_callback=progress_callback,
            )

            # Assign transcriptions back to image references
            for i, transcription in enumerate(transcriptions):
                img_ref = all_prompts_and_images[i][0]
                img_ref.transcription = transcription.strip()
        except Exception as e:
            print(f"Warning: Batch transcription failed: {e}")

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
            response = self.vlm_client.generate_with_image(prompt, image_path)
            return response.strip()
        except Exception as e:
            raise RuntimeError(f"VLM transcription failed: {e}")
