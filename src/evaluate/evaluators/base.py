"""Base evaluator interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """Results from evaluating a single sample."""

    task_id: str
    success: bool
    predictions: list[str]
    ground_truth: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class AggregatedResults:
    """Aggregated evaluation results across all samples."""

    total_samples: int
    successful: int
    failed: int
    metrics: dict[str, float]
    per_sample_results: list[EvaluationResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        return self.successful / self.total_samples if self.total_samples > 0 else 0.0


class Evaluator(ABC):
    """Abstract base class for evaluators."""

    @abstractmethod
    def evaluate_sample(self, sample: dict[str, Any], predictions: list[str]) -> EvaluationResult:
        """
        Evaluate predictions for a single sample.

        Args:
            sample: The input sample containing task information
            predictions: List of predicted outputs (for pass@k evaluation)

        Returns:
            EvaluationResult containing metrics and success status
        """
        pass

    @abstractmethod
    def aggregate_results(self, results: list[EvaluationResult]) -> AggregatedResults:
        """
        Aggregate individual evaluation results.

        Args:
            results: List of individual evaluation results

        Returns:
            AggregatedResults containing aggregated metrics
        """
        pass
