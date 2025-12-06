"""Shared types for generation stages."""

from dataclasses import dataclass
from typing import Optional

from synthetic_data.config import QuestionType
from synthetic_data.extractors.chunker import Chunk
from synthetic_data.parsers.base import ImageReference


@dataclass
class InputCandidate:
    """A candidate input generated from a chunk."""

    chunk: Chunk
    question: str
    question_type: QuestionType
    test_code: Optional[str] = None
    entry_point: Optional[str] = None
    target_image: Optional[ImageReference] = None
    context: str = ""
    score: float = 0.0
    rejection_reason: Optional[str] = None

    @property
    def is_multimodal(self) -> bool:
        return self.target_image is not None

    @property
    def is_valid(self) -> bool:
        return self.rejection_reason is None and bool(self.question)

