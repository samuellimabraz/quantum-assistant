"""Synthetic data generation."""

from .allocation import (
    AllocationConfig,
    AllocationMetrics,
    AllocationResult,
    Allocator,
    DiversityTracker,
    SampleTask,
    TypeAllocationConfig,
)
from .category import CategoryClassifier
from .prompts import PromptSet, build_context
from .types import InputCandidate
from .sessions import AnswerBatchProcessor, AnswerResult, AnswerSession
from .stages import (
    AnswerStage,
    ClassifyStage,
    CurateStage,
    FilterCandidatesStage,
    PlanStage,
)

__all__ = [
    # Allocation
    "AllocationConfig",
    "AllocationMetrics",
    "AllocationResult",
    "Allocator",
    "DiversityTracker",
    "SampleTask",
    "TypeAllocationConfig",
    # Types
    "InputCandidate",
    # Generation stages
    "AnswerStage",
    "ClassifyStage",
    "CurateStage",
    "FilterCandidatesStage",
    "PlanStage",
    # Classification
    "CategoryClassifier",
    # Prompts
    "PromptSet",
    "build_context",
    # Sessions
    "AnswerSession",
    "AnswerBatchProcessor",
    "AnswerResult",
]
