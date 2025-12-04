"""Image processing utilities for fine-tuning data preparation."""

import hashlib
import io
from pathlib import Path

from PIL import Image

from .config import ImageConfig


class ImageProcessor:
    """Process and resize images for VLM fine-tuning.

    Handles image resizing with aspect ratio preservation, format conversion,
    and efficient caching to avoid reprocessing.
    """

    def __init__(self, config: ImageConfig, output_dir: Path):
        """Initialize image processor.

        Args:
            config: Image processing configuration
            output_dir: Directory to save processed images
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._processed_cache: dict[str, Path] = {}

    def process(self, image: Image.Image | None, original_path: str | None = None) -> Path | None:
        """Process an image: resize and save to output directory.

        Args:
            image: PIL Image object to process
            original_path: Original path for naming (optional)

        Returns:
            Path to processed image, or None if no image
        """
        if image is None:
            return None

        image_hash = self._compute_hash(image)
        if image_hash in self._processed_cache:
            return self._processed_cache[image_hash]

        processed = self._resize(image)
        processed = self._convert_mode(processed)
        output_path = self._save(processed, image_hash, original_path)

        self._processed_cache[image_hash] = output_path
        return output_path

    def _resize(self, image: Image.Image) -> Image.Image:
        """Resize image maintaining aspect ratio."""
        if not self.config.preserve_aspect_ratio:
            return image.resize(
                (self.config.max_size, self.config.max_size), Image.Resampling.LANCZOS
            )

        width, height = image.size
        max_dim = max(width, height)

        if max_dim <= self.config.max_size:
            return image

        scale = self.config.max_size / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _convert_mode(self, image: Image.Image) -> Image.Image:
        """Convert image to RGB mode for JPEG compatibility."""
        if self.config.format.upper() == "JPEG" and image.mode in ("RGBA", "P", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "RGBA":
                background.paste(image, mask=image.split()[3])
            elif image.mode == "P":
                image = image.convert("RGBA")
                if "A" in image.getbands():
                    background.paste(image, mask=image.split()[3])
                else:
                    background = image.convert("RGB")
            else:
                background.paste(image.convert("RGB"))
            return background
        elif image.mode not in ("RGB", "L"):
            return image.convert("RGB")
        return image

    def _save(self, image: Image.Image, image_hash: str, original_path: str | None) -> Path:
        """Save processed image to output directory."""
        extension = "jpg" if self.config.format.upper() == "JPEG" else self.config.format.lower()

        if original_path:
            stem = Path(original_path).stem
            filename = f"{stem}_{image_hash[:8]}.{extension}"
        else:
            filename = f"{image_hash[:16]}.{extension}"

        output_path = self.output_dir / filename

        save_kwargs = {}
        if self.config.format.upper() == "JPEG":
            save_kwargs["quality"] = self.config.quality
            save_kwargs["optimize"] = True

        image.save(output_path, format=self.config.format, **save_kwargs)
        return output_path

    def _compute_hash(self, image: Image.Image) -> str:
        """Compute hash of image content for deduplication."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return hashlib.sha256(buffer.getvalue()).hexdigest()

    @property
    def processed_count(self) -> int:
        """Return number of unique images processed."""
        return len(self._processed_cache)
