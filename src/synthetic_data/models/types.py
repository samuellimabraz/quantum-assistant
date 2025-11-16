"""Type definitions for synthetic data generation."""

from dataclasses import dataclass, field


@dataclass
class Sample:
    """Generated dataset sample."""

    question: str
    answer: str
    category: str
    question_type: str

    image_path: str | None = None
    source_path: str | None = None
    metadata: dict = field(default_factory=dict)
