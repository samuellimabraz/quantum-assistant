"""Image format conversion utilities.

Provides high-quality SVG to PNG conversion using CairoSVG (preferred)
with Wand/ImageMagick as fallback. CairoSVG has superior SVG compliance,
especially for clip-paths and complex CSS styles used in Qiskit circuit diagrams.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

MAX_PROCESSABLE_PIXELS = 30_000_000


class SVGConverter:
    """Converts SVG images to PNG with high fidelity.

    Uses CairoSVG as primary converter (best SVG compliance),
    with Wand/ImageMagick as fallback. Preserves native SVG resolution.
    """

    def __init__(self, scale: float = 2.0):
        """Initialize SVG converter.

        Args:
            scale: Scale factor for SVG rendering (2.0 = 2x native resolution for quality)
        """
        self.scale = scale
        self._cairosvg_available: bool | None = None
        self._wand_available: bool | None = None

    @property
    def cairosvg_available(self) -> bool:
        """Check if CairoSVG is available."""
        if self._cairosvg_available is None:
            try:
                import cairosvg

                cairosvg.svg2png(bytestring=b'<svg width="1" height="1"></svg>')
                self._cairosvg_available = True
            except (ImportError, OSError, ValueError):
                self._cairosvg_available = False
        return self._cairosvg_available

    @property
    def wand_available(self) -> bool:
        """Check if Wand/ImageMagick is available."""
        if self._wand_available is None:
            try:
                from wand import image as wand_image_module

                _ = wand_image_module.Image
                self._wand_available = True
            except (ImportError, OSError, AttributeError):
                self._wand_available = False
        return self._wand_available

    def convert(self, svg_path: Path | str) -> Image.Image:
        """Convert SVG to PIL Image.

        Args:
            svg_path: Path to SVG file

        Returns:
            PIL Image in RGB or RGBA mode

        Raises:
            FileNotFoundError: If SVG file doesn't exist
            ValueError: If image is too large
            RuntimeError: If no converter is available
        """
        svg_path = Path(svg_path)
        if not svg_path.exists():
            raise FileNotFoundError(f"SVG file not found: {svg_path}")

        # Check file size - reject extremely large SVGs upfront
        file_size_mb = svg_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 10:
            raise ValueError(
                f"SVG file too large: {file_size_mb:.1f}MB (max 10MB). "
                f"Large SVGs often render poorly and timeout."
            )

        if self.cairosvg_available:
            img = self._convert_with_cairosvg(svg_path)

            if self._is_blank_image(img) and self.wand_available:
                return self._convert_with_wand(svg_path)
            return img

        if self.wand_available:
            return self._convert_with_wand(svg_path)

        raise RuntimeError(
            "No SVG converter available. Install one of:\n"
            "  - CairoSVG (recommended): pip install cairosvg && brew install cairo\n"
            "  - Wand: pip install Wand && brew install imagemagick"
        )

    def _is_blank_image(self, img: Image.Image) -> bool:
        """Check if image is blank/white (CairoSVG rendering issue).

        Args:
            img: PIL Image to check

        Returns:
            True if image appears to be blank
        """
        # Convert to RGB for consistent checking
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Sample center pixel and corners
        w, h = img.size
        samples = [
            img.getpixel((w // 2, h // 2)),  # center
            img.getpixel((w // 4, h // 4)),  # top-left quarter
            img.getpixel((3 * w // 4, 3 * h // 4)),  # bottom-right quarter
        ]

        # Check if all samples are white or near-white
        return all(all(c >= 250 for c in pixel) for pixel in samples)

    def _convert_with_cairosvg(self, svg_path: Path) -> Image.Image:
        """Convert SVG using CairoSVG at native resolution."""
        import cairosvg

        png_data = cairosvg.svg2png(
            url=str(svg_path),
            scale=self.scale,
            background_color="white",
        )

        return Image.open(io.BytesIO(png_data))

    def _convert_with_wand(self, svg_path: Path) -> Image.Image:
        """Convert SVG using Wand/ImageMagick (fallback).

        Note: Wand may not render all SVG features correctly,
        especially clip-paths and complex CSS styles.
        """
        from wand.color import Color
        from wand.image import Image as WandImage

        resolution = int(72 * self.scale)

        with WandImage(filename=str(svg_path), resolution=resolution) as wand_img:
            wand_img.background_color = Color("white")
            wand_img.alpha_channel = "remove"
            wand_img.format = "png"
            png_blob = wand_img.make_blob("png")

        return Image.open(io.BytesIO(png_blob))


class ImageLoader:
    """Unified image loader supporting multiple formats.

    Handles:
    - Standard formats (PNG, JPEG, etc.) via PIL
    - SVG via SVGConverter (CairoSVG or Wand)
    - AVIF via pillow-avif or Wand
    """

    def __init__(
        self,
        max_dimension: int = 1024,
        svg_scale: float = 2.0,
    ):
        """Initialize image loader.

        Args:
            max_dimension: Maximum width/height for output (maintains aspect ratio)
            svg_scale: Scale factor for SVG rendering (2.0 = 2x native for quality)
        """
        self.max_dimension = max_dimension
        self.svg_converter = SVGConverter(scale=svg_scale)

    def load(self, image_path: Path | str) -> Image.Image:
        """Load image from path and convert to standard RGB format.

        Args:
            image_path: Path to image file

        Returns:
            PIL Image in RGB mode, resized if needed

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image is too large to process
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        suffix = image_path.suffix.lower()

        if suffix == ".svg":
            img = self.svg_converter.convert(image_path)
        elif suffix == ".avif":
            img = self._load_avif(image_path)
        else:
            img = Image.open(image_path)

        # Check if image is too large BEFORE resizing
        # This prevents expensive operations on massive images
        if img.width * img.height > MAX_PROCESSABLE_PIXELS:
            raise ValueError(
                f"Image too large: {img.width}x{img.height} "
                f"({img.width * img.height / 1e6:.1f}M pixels, max {MAX_PROCESSABLE_PIXELS / 1e6:.1f}M). "
                f"Resize source image before processing."
            )

        # Resize if needed
        if img.width > self.max_dimension or img.height > self.max_dimension:
            img.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)

        img = self._ensure_rgb(img)

        return img

    def _load_avif(self, image_path: Path) -> Image.Image:
        """Load AVIF image."""
        try:
            import pillow_avif  # noqa: F401

            return Image.open(image_path)
        except ImportError:
            pass

        try:
            from wand.color import Color
            from wand.image import Image as WandImage

            with WandImage(filename=str(image_path)) as wand_img:
                if wand_img.width * wand_img.height > MAX_PROCESSABLE_PIXELS:
                    raise ValueError(f"AVIF too large: {wand_img.width}x{wand_img.height}")

                wand_img.background_color = Color("white")
                wand_img.alpha_channel = "remove"
                wand_img.format = "png"
                png_blob = wand_img.make_blob("png")

            return Image.open(io.BytesIO(png_blob))
        except ImportError as exc:
            raise ImportError(
                "No AVIF loader available. Install one of:\n"
                "  - pillow-avif: pip install pillow-avif-plugin\n"
                "  - Wand: pip install Wand && brew install imagemagick"
            ) from exc

    def _ensure_rgb(self, img: Image.Image) -> Image.Image:
        """Convert image to RGB mode."""
        if img.mode == "RGB":
            return img

        if img.mode == "RGBA":
            # Composite onto white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.split() and len(img.split()) > 3:
                background.paste(img, mask=img.split()[3])
            else:
                background.paste(img)
            return background

        if img.mode == "L":
            return img.convert("RGB")

        return img.convert("RGB")

    def load_as_base64(
        self,
        image_path: Path | str,
        output_format: str = "JPEG",
        quality: int = 95,
    ) -> str:
        """Load image and return as base64-encoded string.

        Args:
            image_path: Path to image file
            output_format: Output format (JPEG or PNG)
            quality: JPEG quality (1-100)

        Returns:
            Base64-encoded image string
        """
        img = self.load(image_path)

        buffer = io.BytesIO()
        if output_format.upper() == "JPEG":
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
        else:
            img.save(buffer, format=output_format)

        return base64.b64encode(buffer.getvalue()).decode()


# Module-level cache for singleton instances
class _ConverterCache:
    """Cache for converter singleton instances."""

    svg_converter: SVGConverter | None = None
    image_loader: ImageLoader | None = None


def get_svg_converter() -> SVGConverter:
    """Get default SVG converter instance."""
    if _ConverterCache.svg_converter is None:
        _ConverterCache.svg_converter = SVGConverter()
    return _ConverterCache.svg_converter


def get_image_loader() -> ImageLoader:
    """Get default image loader instance."""
    if _ConverterCache.image_loader is None:
        _ConverterCache.image_loader = ImageLoader()
    return _ConverterCache.image_loader
