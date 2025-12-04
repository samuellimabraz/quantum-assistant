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
from .pipeline import GenerationPipeline
from .planner import InputCandidate, InputPlanner
from .prompts import PromptSet, build_context
from .sessions import AnswerBatchProcessor, AnswerResult, AnswerSession

__all__ = [
    # Allocation
    "AllocationConfig",
    "AllocationMetrics",
    "AllocationResult",
    "Allocator",
    "DiversityTracker",
    "SampleTask",
    "TypeAllocationConfig",
    # Pipeline
    "CategoryClassifier",
    "GenerationPipeline",
    "InputPlanner",
    "InputCandidate",
    "PromptSet",
    "build_context",
    # Sessions
    "AnswerSession",
    "AnswerBatchProcessor",
    "AnswerResult",
]
