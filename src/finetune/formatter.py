"""ms-swift format converter for fine-tuning datasets."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SwiftFormatConfig


@dataclass
class SwiftSample:
    """A sample in ms-swift format.

    Attributes:
        messages: List of message dictionaries with role and content
        images: List of image paths (for multimodal samples)
    """

    messages: list[dict[str, str]]
    images: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"messages": self.messages}
        if self.images:
            result["images"] = self.images
        return result


class SwiftFormatter:
    """Format samples for ms-swift fine-tuning framework.

    Converts samples from the quantum-assistant dataset format to the
    ms-swift expected JSONL format with proper message structure.

    ms-swift format:
    ```json
    {
        "messages": [
            {"role": "system", "content": "<system-prompt>"},
            {"role": "user", "content": "<image><query>"},
            {"role": "assistant", "content": "<response>"}
        ],
        "images": ["path/to/image.jpg"]
    }
    ```
    """

    def __init__(self, config: SwiftFormatConfig):
        """Initialize formatter.

        Args:
            config: Swift format configuration
        """
        self.config = config

    def format_sample(
        self,
        question: str,
        answer: str,
        question_type: str,
        image_path: Path | None = None,
    ) -> SwiftSample:
        """Format a single sample to ms-swift format.

        Args:
            question: The input question/prompt
            answer: The reference answer
            question_type: Type of question (function_completion, code_generation, qa)
            image_path: Path to processed image (if multimodal)

        Returns:
            SwiftSample object ready for serialization
        """
        messages = []

        if self.config.include_system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        user_content = self._build_user_content(
            question, question_type, has_image=image_path is not None
        )
        messages.append({"role": "user", "content": user_content})

        messages.append({"role": "assistant", "content": answer})

        images = [str(image_path)] if image_path else []

        return SwiftSample(messages=messages, images=images)

    def _build_user_content(self, question: str, question_type: str, has_image: bool) -> str:
        """Build user message content with image placeholder if needed.

        Args:
            question: The input question
            question_type: Type of question
            has_image: Whether the sample has an image

        Returns:
            Formatted user content string
        """
        if has_image:
            return f"{self.config.image_placeholder}\n{question}"
        return question

    def write_jsonl(self, samples: list[SwiftSample], output_path: Path) -> int:
        """Write samples to JSONL file.

        Args:
            samples: List of SwiftSample objects
            output_path: Path to output JSONL file

        Returns:
            Number of samples written
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

        return len(samples)
