"""Model client implementations with async batch support."""

import asyncio
import base64
import io
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

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

        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

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

        payload = {
            "model": self.model_name,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
        }

        if self.service_tier:
            payload["service_tier"] = self.service_tier

        response = self.client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def generate_async(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text completion asynchronously with retry logic."""
        try:
            if self.async_client.is_closed:
                self.async_client = httpx.AsyncClient(timeout=self.timeout)
        except AttributeError:
            pass

        url = f"{self.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
        }

        if self.service_tier:
            payload["service_tier"] = self.service_tier

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
        max_concurrent: int = 10,
        progress_callback=None,
    ) -> list[str]:
        """
        Generate completions for multiple inputs concurrently.

        Args:
            batch: List of message lists
            max_tokens: Override max tokens
            temperature: Override temperature
            max_concurrent: Maximum concurrent requests
            progress_callback: Optional callback function(completed_count) called after each completion

        Returns:
            List of generated texts
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        completed_count = 0
        results = [None] * len(batch)
        lock = asyncio.Lock()

        async def generate_with_limit(idx, messages):
            nonlocal completed_count
            async with semaphore:
                try:
                    result = await self.generate_async(messages, max_tokens, temperature)
                except Exception as e:
                    print(f"Error processing batch item {idx}: {e}")
                    result = ""

                async with lock:
                    results[idx] = result
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count)
                return result

        tasks = [generate_with_limit(i, messages) for i, messages in enumerate(batch)]
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
    """Client for vision-language models with async batch support."""

    def _process_image(self, image_path: Path | str, save_debug: bool = False) -> str:
        """
        Process image to base64 JPEG.

        Args:
            image_path: Path to image file
            save_debug: If True, save converted image for debugging

        Returns:
            Base64 encoded JPEG image
        """
        image_path = Path(image_path)

        if image_path.suffix.lower() == ".svg":
            try:
                from wand.image import Image as WandImage
                from wand.color import Color

                with WandImage(filename=str(image_path), resolution=300) as wand_img:
                    wand_img.background_color = Color("white")
                    wand_img.alpha_channel = "remove"

                    wand_img.format = "png"
                    png_blob = wand_img.make_blob("png")

                    img = Image.open(io.BytesIO(png_blob))

            except ImportError:
                raise ImportError(
                    "Wand required for SVG. Install with: pip install Wand\n"
                    "Also requires ImageMagick: brew install imagemagick (Mac) or apt-get install libmagickwand-dev (Linux)"
                )
        else:
            if str(image_path).lower().endswith(".avif"):
                try:
                    import pillow_avif
                except ImportError:
                    raise ImportError(
                        "pillow-avif required for AVIF. Install with: pip install pillow-avif"
                    )
            img = Image.open(image_path)

        # Resize large images
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
        img.close()

        return image_data

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
    ) -> str:
        """Generate text completion with image input asynchronously."""
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

        return await self.generate_async(
            messages=messages, max_tokens=max_tokens, temperature=temperature
        )

    async def generate_batch_with_images_async(
        self,
        prompts: list[tuple[str, Path]],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_concurrent: int = 8,
        progress_callback=None,
    ) -> list[str]:
        """
        Generate completions for multiple image+text inputs concurrently.

        Args:
            prompts: List of (text, image_path) tuples
            system_prompt: Optional system prompt to use for all requests
            max_tokens: Override max tokens
            temperature: Override temperature
            max_concurrent: Maximum concurrent requests (lower for VLM)
            progress_callback: Optional callback function(completed_count) called after each completion

        Returns:
            List of generated texts
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        completed_count = 0
        results = [None] * len(prompts)
        lock = asyncio.Lock()

        async def generate_with_limit(idx, text, image_path):
            nonlocal completed_count
            async with semaphore:
                try:
                    result = await self.generate_with_image_async(
                        text, image_path, system_prompt, max_tokens, temperature
                    )
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    result = ""

                async with lock:
                    results[idx] = result
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count)
                return result

        tasks = [generate_with_limit(i, text, img) for i, (text, img) in enumerate(prompts)]
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
