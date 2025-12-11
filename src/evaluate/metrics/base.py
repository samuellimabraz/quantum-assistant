"""Base metric interface."""

from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def compute(self, predictions: Any, references: Any, **kwargs) -> float:
        """
        Compute the metric.

        Args:
            predictions: Model predictions
            references: Ground truth references
            **kwargs: Additional metric-specific parameters

        Returns:
            Metric score as a float
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        pass
