import os
from pathlib import Path

from dotenv import load_dotenv

from synthetic_data.models import VLMClient


def test_vlm_with_images():
    """Test VLM client with various image formats."""
    load_dotenv()

    base_url = os.getenv("VISION_MODEL_BASE_URL")
    api_key = os.getenv("VISION_MODEL_API_KEY")
    model_name = os.getenv("VISION_MODEL_NAME")

    if not base_url or not api_key:
        print("Vision model credentials not found in environment")
        return

    client = VLMClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        max_tokens=100,
        temperature=0.3,
    )

    test_images = Path("assets/tests")
    if not test_images.exists():
        print(f"Test images directory not found: {test_images}")
        return

    image_files = list(test_images.glob("*"))

    print(f"Found {len(image_files)} test images")
    print("=" * 60)

    for img_path in image_files:
        if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".svg", ".avif"]:
            continue

        print(f"\nTesting: {img_path.name}")
        print(f"  Format: {img_path.suffix}")
        print(f"  Size: {img_path.stat().st_size / 1024:.1f} KB")

        try:
            prompt = """Provide a detailed, objective and technical description of this image related to quantum computing, mathematics, or physics.
    If the image is not related to quantum computing, mathematics, or physics, describe in simple terms what it is.
    Focus on:
    - Key visual elements (circuits, gates, qubits, states, diagrams, formulas, graphs, etc.), how they interact
    - Mathematical notation or formulas if present
    - Any labels, legends, or annotations that help understand the image
    - Charts or graphs description if present

    The idea is that it should be possible to understand the image without looking at it."""
            response = client.generate_with_image(
                text=f"{prompt}",
                image_path=img_path,
                max_tokens=4096,
            )

            print(f"  Response: {response}")
            print("  ✓ Success")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    client.close()
    print("\n" + "=" * 60)
    print("VLM image test completed!")


if __name__ == "__main__":
    test_vlm_with_images()
