"""Type definitions for synthetic data generation."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Sample:
    """Generated dataset sample.
    
    Represents a single training sample for quantum computing VLM fine-tuning.
    
    For function_completion and code_generation types:
    - test_code: Unit test that validates the answer code
    - entry_point: Function name that the test calls
    - Answer must pass the test to be valid
    
    For qa type:
    - No test required
    - Code in answer is verified for syntax/execution only
    
    Attributes:
        question: The input prompt/question
        answer: The reference solution/answer
        category: One of 14 quantum computing categories
        question_type: function_completion, code_generation, or qa
        test_code: Unit test code (for code types only)
        entry_point: Function name being tested (for code types only)
        image_path: Path to associated image (for multimodal samples)
        source_path: Path to source document
        metadata: Additional metadata
    """

    question: str
    answer: str
    category: str
    question_type: str

    # For function_completion and code_generation types
    test_code: Optional[str] = None
    entry_point: Optional[str] = None

    # Multimodal support
    image_path: Optional[str] = None

    # Provenance
    source_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_multimodal(self) -> bool:
        """Check if sample has an associated image."""
        return self.image_path is not None

    @property
    def has_test(self) -> bool:
        """Check if sample has a unit test."""
        return self.test_code is not None and len(self.test_code.strip()) > 0

    @property
    def is_code_type(self) -> bool:
        """Check if sample is a code-focused type requiring unit test."""
        return self.question_type in ("function_completion", "code_generation")

    def to_dict(self) -> dict:
        """Convert sample to dictionary for serialization."""
        return {
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "question_type": self.question_type,
            "test_code": self.test_code,
            "entry_point": self.entry_point,
            "image_path": self.image_path,
            "source_path": self.source_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Sample":
        """Create sample from dictionary."""
        return cls(
            question=data["question"],
            answer=data["answer"],
            category=data["category"],
            question_type=data["question_type"],
            test_code=data.get("test_code"),
            entry_point=data.get("entry_point"),
            image_path=data.get("image_path"),
            source_path=data.get("source_path"),
            metadata=data.get("metadata", {}),
        )
