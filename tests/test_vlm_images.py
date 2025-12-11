import os
from pathlib import Path

import pytest
from PIL import Image

from synthetic_data.utils.image_converter import ImageLoader


class TestImageLoader:

    def test_max_dimension_640(self, tmp_path: Path):
        large_image = Image.new("RGB", (2000, 1500), color="blue")
        test_path = tmp_path / "large_image.png"
        large_image.save(test_path)

        loader = ImageLoader(max_dimension=640)
        result = loader.load(test_path)

        assert result.width <= 640
        assert result.height <= 640
        assert result.mode == "RGB"

    def test_max_dimension_1024(self, tmp_path: Path):
        large_image = Image.new("RGB", (2000, 1500), color="green")
        test_path = tmp_path / "large_image.png"
        large_image.save(test_path)

        loader = ImageLoader(max_dimension=1024)
        result = loader.load(test_path)

        assert result.width <= 1024
        assert result.height <= 1024
        assert result.mode == "RGB"

    def test_small_image_unchanged(self, tmp_path: Path):
        small_image = Image.new("RGB", (300, 200), color="red")
        test_path = tmp_path / "small_image.png"
        small_image.save(test_path)

        loader = ImageLoader(max_dimension=640)
        result = loader.load(test_path)

        assert result.width == 300
        assert result.height == 200

    def test_aspect_ratio_preserved(self, tmp_path: Path):
        wide_image = Image.new("RGB", (2000, 500), color="yellow")
        test_path = tmp_path / "wide_image.png"
        wide_image.save(test_path)

        loader = ImageLoader(max_dimension=640)
        result = loader.load(test_path)

        original_ratio = 2000 / 500
        result_ratio = result.width / result.height
        assert abs(original_ratio - result_ratio) < 0.01

    def test_rgba_to_rgb_conversion(self, tmp_path: Path):
        rgba_image = Image.new("RGBA", (400, 300), color=(255, 0, 0, 128))
        test_path = tmp_path / "rgba_image.png"
        rgba_image.save(test_path)

        loader = ImageLoader(max_dimension=640)
        result = loader.load(test_path)

        assert result.mode == "RGB"

    def test_grayscale_to_rgb(self, tmp_path: Path):
        gray_image = Image.new("L", (400, 300), color=128)
        test_path = tmp_path / "gray_image.png"
        gray_image.save(test_path)

        loader = ImageLoader(max_dimension=640)
        result = loader.load(test_path)

        assert result.mode == "RGB"

    def test_load_as_base64_jpeg(self, tmp_path: Path):
        test_image = Image.new("RGB", (400, 300), color="purple")
        test_path = tmp_path / "test_image.png"
        test_image.save(test_path)

        loader = ImageLoader(max_dimension=640)
        base64_str = loader.load_as_base64(test_path, output_format="JPEG", quality=95)

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

        import base64
        import io

        decoded = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(decoded))
        assert img.format == "JPEG"

    def test_file_not_found(self):
        loader = ImageLoader(max_dimension=640)

        with pytest.raises(FileNotFoundError):
            loader.load(Path("/nonexistent/path/image.png"))


class TestVLMClientImageProcessing:

    def test_vlm_client_uses_640_dimension(self):
        from synthetic_data.models import VLMClient

        client = VLMClient(
            base_url="http://localhost:8000",
            api_key="test",
            model_name="test-model",
        )

        assert client.max_dimension == 640
        assert client.image_loader.max_dimension == 640

    def test_vlm_client_custom_dimension(self):
        from synthetic_data.models import VLMClient

        client = VLMClient(
            base_url="http://localhost:8000",
            api_key="test",
            model_name="test-model",
            max_dimension=512,
        )

        assert client.max_dimension == 512
        assert client.image_loader.max_dimension == 512


@pytest.mark.skipif(
    not os.getenv("VISION_MODEL_BASE_URL"),
    reason="Vision model credentials not configured",
)
class TestVLMIntegration:

    def test_vlm_single_image(self):
        from dotenv import load_dotenv

        from synthetic_data.models import VLMClient

        load_dotenv()

        client = VLMClient(
            base_url=os.getenv("VISION_MODEL_BASE_URL"),
            api_key=os.getenv("VISION_MODEL_API_KEY"),
            model_name=os.getenv("VISION_MODEL_NAME"),
            max_tokens=4096,
            temperature=0.1,
        )

        test_image = Path("assets/tests/90c68fb2-6ed5-41f3-a4f8-f73e92367c4c-0.svg")
        if not test_image.exists():
            pytest.skip(f"Test image not found: {test_image}")

        response = client.generate_with_image(
            text="Briefly describe this quantum circuit.",
            image_path=test_image,
            max_tokens=256,
        )

        client.close()

        assert response
        assert len(response) > 20
        assert "circuit" in response.lower() or "quantum" in response.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
