"""Model client implementations with async batch support and optimized parallelization."""

import asyncio
import base64
import io
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image


Image.MAX_IMAGE_PIXELS = 500_000_000

MAX_PROCESSABLE_PIXELS = 30_000_000  


class ImageProcessingTimeout(Exception):
    """Raised when image processing times out."""

    pass


# macOS: Automatically configure library paths if using Homebrew
if sys.platform == "darwin":
    # Configure ImageMagick for Wand
    if "MAGICK_HOME" not in os.environ:
        os.environ["MAGICK_HOME"] = "/opt/homebrew"

    # Configure cairo library path
    try:
        cairo_path = (
            subprocess.check_output(["brew", "--prefix", "cairo"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        lib_path = f"{cairo_path}/lib"
        if os.path.exists(lib_path):
            dyld_path = os.environ.get("DYLD_LIBRARY_PATH", "")
            if lib_path not in dyld_path:
                os.environ["DYLD_LIBRARY_PATH"] = f"{lib_path}:{dyld_path}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

# Shared thread pool for CPU-bound image processing
_image_executor: ThreadPoolExecutor | None = None


def get_image_executor() -> ThreadPoolExecutor:
    """Get or create shared thread pool for image processing."""
    global _image_executor
    if _image_executor is None:
        # Use CPU count for image processing threads
        _image_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
    return _image_executor


@dataclass
class Message:
    """Chat message."""

    role: str
    content: str | list[dict]


class LLMClient:
    """Client for text-based language models with async batch support."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model_name: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 300.0,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        service_tier: str | None = "auto",
        max_connections: int = 100,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
    ):
        """Initialize LLM client."""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name or "default"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.service_tier = service_tier
        self.top_p = top_p
        self.min_p = min_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty

        # Optimized connection limits for high throughput
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections // 2,
        )
        self.client = httpx.Client(timeout=timeout, limits=limits)
        self.async_client = httpx.AsyncClient(timeout=timeout, limits=limits)

    def _build_payload(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict:
        """Build the API payload with all configured parameters."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if self.service_tier:
            payload["service_tier"] = self.service_tier
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.presence_penalty is not None:
            payload["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = self.repetition_penalty

        return payload

    def generate(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text completion synchronously."""
        url = f"{self.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = self._build_payload(messages, max_tokens, temperature)

        response = self.client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _ensure_async_client(self):
        """Ensure async client is available."""
        try:
            if self.async_client.is_closed:
                limits = httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=50,
                )
                self.async_client = httpx.AsyncClient(timeout=self.timeout, limits=limits)
        except AttributeError:
            pass

    async def generate_async(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text completion asynchronously with retry logic."""
        self._ensure_async_client()

        url = f"{self.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = self._build_payload(messages, max_tokens, temperature)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                else:
                    raise last_error

    async def generate_batch_async(
        self,
        batch: list[list[Message]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_concurrent: int = 20,
        progress_callback=None,
    ) -> list[str]:
        """
        Generate completions for multiple inputs with full parallelization.

        All requests run concurrently up to max_concurrent limit.
        No artificial batching - streaming results with immediate progress updates.

        Args:
            batch: List of message lists
            max_tokens: Override max tokens
            temperature: Override temperature
            max_concurrent: Maximum concurrent requests
            progress_callback: Optional callback function(completed_count)

        Returns:
            List of generated texts
        """
        if not batch:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(batch)
        completed = [0]
        lock = asyncio.Lock()

        async def generate_one(idx: int, messages: list[Message]) -> str:
            """Generate single completion with concurrency control."""
            async with semaphore:
                try:
                    result = await self.generate_async(messages, max_tokens, temperature)
                except Exception as e:
                    print(f"Error in batch item {idx}: {e}")
                    result = ""

            async with lock:
                results[idx] = result
                completed[0] += 1
                if progress_callback:
                    progress_callback(completed[0])

            return result

        # Launch all tasks concurrently
        tasks = [generate_one(i, msgs) for i, msgs in enumerate(batch)]
        await asyncio.gather(*tasks)

        return results

    def generate_batch(
        self,
        batch: list[list[Message]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_concurrent: int = 10,
    ) -> list[str]:
        """Generate completions for multiple inputs (sync wrapper)."""
        return asyncio.run(
            self.generate_batch_async(batch, max_tokens, temperature, max_concurrent)
        )

    async def aclose(self):
        """Close async HTTP client."""
        await self.async_client.aclose()

    def close(self):
        """Close HTTP clients."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class VLMClient(LLMClient):
    """Client for vision-language models with async batch support and optimized parallelization."""

    def _process_image(self, image_path: Path | str, save_debug: bool = False) -> str:
        """
        Process image to base64 JPEG (CPU-bound, run in thread pool for async).

        Args:
            image_path: Path to image file
            save_debug: If True, save converted image for debugging

        Returns:
            Base64 encoded JPEG image

        Raises:
            ValueError: If image is too large to process
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If image processing fails
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = None
        suffix = image_path.suffix.lower()

        try:
            if suffix == ".svg":
                img = self._load_svg_image(image_path)
            elif suffix == ".avif":
                img = self._load_avif_image(image_path)
            else:
                img = Image.open(image_path)

            total_pixels = img.width * img.height
            if total_pixels > MAX_PROCESSABLE_PIXELS:
                raise ValueError(
                    f"Image too large: {img.width}x{img.height} ({total_pixels/1e6:.1f}M pixels). "
                    f"Max: {MAX_PROCESSABLE_PIXELS/1e6:.0f}M pixels"
                )

            # Resize to target dimension
            max_dimension = 640
            if img.width > max_dimension or img.height > max_dimension:
                ratio = min(max_dimension / img.width, max_dimension / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to RGB
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            elif img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background

            if save_debug:
                debug_dir = Path("outputs/vlm_debug")
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / f"{image_path.stem}_processed.jpg"
                img.save(debug_path, format="JPEG", quality=95)
                print(f"  [Debug] Saved processed image: {debug_path}")

            # Encode to base64
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95, optimize=True)
            image_data = base64.b64encode(buffer.getvalue()).decode()

            return image_data

        finally:
            if img is not None:
                try:
                    img.close()
                except Exception:
                    pass

    def _load_svg_image(self, image_path: Path) -> Image.Image:
        """Load SVG image using Wand/ImageMagick."""
        try:
            from wand.image import Image as WandImage
            from wand.color import Color

            with WandImage(filename=str(image_path), resolution=300) as wand_img:
                if wand_img.width * wand_img.height > MAX_PROCESSABLE_PIXELS:
                    raise ValueError(f"SVG too large: {wand_img.width}x{wand_img.height}")

                wand_img.background_color = Color("white")
                wand_img.alpha_channel = "remove"
                wand_img.format = "png"
                png_blob = wand_img.make_blob("png")

            return Image.open(io.BytesIO(png_blob))

        except ImportError:
            raise ImportError(
                "Wand required for SVG. Install with: pip install Wand\n"
                "Also requires ImageMagick: brew install imagemagick (Mac) "
                "or apt-get install libmagickwand-dev (Linux)"
            )

    def _load_avif_image(self, image_path: Path) -> Image.Image:
        """Load AVIF image with PIL fallback to Wand.

        For large images that PIL can't handle, uses Wand to resize first.
        """
        try:
            import pillow_avif 

            img = Image.open(image_path)
            img.load()
            return img
        except ImportError:
            pass
        except Exception:
            pass

        try:
            from wand.image import Image as WandImage
            from wand.color import Color

            with WandImage(filename=str(image_path)) as wand_img:
                total_pixels = wand_img.width * wand_img.height

                if total_pixels > MAX_PROCESSABLE_PIXELS:
                    raise ValueError(
                        f"AVIF too large: {wand_img.width}x{wand_img.height} "
                        f"({total_pixels/1e6:.1f}M pixels)"
                    )

                # For large-ish images, resize in Wand first (faster than PIL resize)
                if total_pixels > 2_000_000:  # > 2MP
                    # Resize to max 1024px dimension for faster processing
                    max_dim = 1024
                    if wand_img.width > max_dim or wand_img.height > max_dim:
                        ratio = min(max_dim / wand_img.width, max_dim / wand_img.height)
                        new_width = int(wand_img.width * ratio)
                        new_height = int(wand_img.height * ratio)
                        wand_img.resize(new_width, new_height)

                wand_img.background_color = Color("white")
                wand_img.alpha_channel = "remove"
                wand_img.format = "png"
                png_blob = wand_img.make_blob("png")

            return Image.open(io.BytesIO(png_blob))

        except ImportError:
            raise ImportError(
                "Neither pillow-avif nor Wand could load AVIF. Install with:\n"
                "  pip install pillow-avif  # or\n"
                "  pip install Wand && brew install imagemagick"
            )
        except ValueError:
            raise

    async def _process_image_async(self, image_path: Path | str, timeout: float = 120.0) -> str:
        """Process image asynchronously using thread pool with timeout.

        Args:
            image_path: Path to image file
            timeout: Maximum seconds to wait for image processing (default 30s)

        Returns:
            Base64 encoded JPEG image

        Raises:
            ImageProcessingTimeout: If processing exceeds timeout
            ValueError: If image is too large
            RuntimeError: If image processing fails
        """
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(get_image_executor(), self._process_image, image_path),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise ImageProcessingTimeout(
                f"Image processing timed out after {timeout}s: {image_path}"
            )

    def generate_with_image(
        self,
        text: str,
        image_path: Path | str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        save_debug: bool = False,
    ) -> str:
        """Generate text completion with image input."""
        image_data = self._process_image(image_path, save_debug=save_debug)

        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        messages.append(
            Message(
                role="user",
                content=[
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            )
        )

        return self.generate(messages=messages, max_tokens=max_tokens, temperature=temperature)

    async def generate_with_image_async(
        self,
        text: str,
        image_path: Path | str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        save_debug: bool = False,
        image_data: str | None = None,
    ) -> str:
        """Generate text completion with image input asynchronously.

        Args:
            text: Text prompt
            image_path: Path to image file
            system_prompt: Optional system prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            save_debug: Save processed image for debugging
            image_data: Pre-processed base64 image data (skip processing if provided)
        """
        # Use pre-processed image data or process async
        if image_data is None:
            image_data = await self._process_image_async(image_path)

        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        messages.append(
            Message(
                role="user",
                content=[
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            )
        )

        return await self.generate_async(
            messages=messages, max_tokens=max_tokens, temperature=temperature
        )

    async def generate_batch_with_images_async(
        self,
        prompts: list[tuple[str, Path]],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_concurrent: int = 16,
        progress_callback=None,
    ) -> list[str]:
        """
        Generate completions for multiple image+text inputs with pipelined processing.

        Uses a two-stage pipeline:
        1. Image processing runs in thread pool (CPU-bound)
        2. API calls run concurrently (I/O-bound)

        Images are pre-processed and queued for API calls, maximizing throughput.

        Args:
            prompts: List of (text, image_path) tuples
            system_prompt: Optional system prompt to use for all requests
            max_tokens: Override max tokens
            temperature: Override temperature
            max_concurrent: Maximum concurrent API requests
            progress_callback: Optional callback function(completed_count)

        Returns:
            List of generated texts
        """
        if not prompts:
            return []

        api_semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(prompts)
        completed_count = [0]
        lock = asyncio.Lock()

        async def process_and_generate(idx: int, text: str, image_path: Path) -> str:
            """Process image and generate completion."""
            result = ""
            try:
                # Process image with timeout
                image_data = await self._process_image_async(image_path, timeout=120.0)

                # Acquire API semaphore and make request
                async with api_semaphore:
                    result = await self.generate_with_image_async(
                        text,
                        image_path,
                        system_prompt,
                        max_tokens,
                        temperature,
                        image_data=image_data,  # Skip re-processing
                    )
            except ImageProcessingTimeout as e:
                print(f"Timeout processing image {image_path.name}: {e}")
            except ValueError as e:
                # Image too large or invalid - skip gracefully
                print(f"Skipping image {image_path.name}: {e}")
            except Exception as e:
                print(f"Error processing image {image_path.name}: {type(e).__name__}: {e}")

            async with lock:
                results[idx] = result
                completed_count[0] += 1
                if progress_callback:
                    progress_callback(completed_count[0])

            return result

        # Launch all tasks - image processing and API calls are pipelined
        tasks = [process_and_generate(i, text, Path(img)) for i, (text, img) in enumerate(prompts)]
        await asyncio.gather(*tasks)

        return results

    def generate_batch_with_images(
        self,
        prompts: list[tuple[str, Path]],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_concurrent: int = 4,
    ) -> list[str]:
        """Generate completions for multiple image+text inputs (sync wrapper)."""
        return asyncio.run(
            self.generate_batch_with_images_async(
                prompts, system_prompt, max_tokens, temperature, max_concurrent
            )
        )
