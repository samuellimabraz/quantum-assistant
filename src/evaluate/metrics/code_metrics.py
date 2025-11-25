"""Code-specific metrics for evaluation."""

import math
from typing import Any

from evaluate.metrics.base import Metric


class PassAtK(Metric):
    """
    Pass@k metric for code generation.

    Measures the probability that at least one of the top k generated
    solutions passes all test cases.
    """

    def __init__(self, k: int = 1):
        """
        Initialize Pass@k metric.

        Args:
            k: Number of samples to consider
        """
        self.k = k

    @property
    def name(self) -> str:
        """Return metric name."""
        return f"pass@{self.k}"

    def compute(
        self, predictions: list[bool], references: Any = None, n: int | None = None, **kwargs
    ) -> float:
        """
        Compute Pass@k.

        Args:
            predictions: List of boolean pass/fail results for generated code
            references: Not used for this metric
            n: Total number of samples generated (if different from len(predictions))
            **kwargs: Additional parameters

        Returns:
            Pass@k score
        """
        if not predictions:
            return 0.0

        n = n or len(predictions)
        c = sum(predictions)  # Number of correct solutions

        if n < self.k:
            return float(c > 0)

        # Use the unbiased estimator from the HumanEval paper
        # pass@k = 1 - (n-c choose k) / (n choose k)
        if c == 0:
            return 0.0

        # If we must pick more samples than there are incorrect ones,
        # we're guaranteed to get at least one correct
        if self.k > n - c:
            return 1.0

        return 1.0 - math.prod((n - c - i) / (n - i) for i in range(self.k))

    @staticmethod
    def compute_pass_at_k(n: int, c: int, k: int) -> float:
        """
        Static method to compute pass@k given n and c.

        Args:
            n: Total number of samples
            c: Number of correct samples
            k: k value for pass@k

        Returns:
            Pass@k score
        """
        if n < k:
            return float(c > 0)
        if c == 0:
            return 0.0

        # If we must pick more samples than there are incorrect ones,
        # we're guaranteed to get at least one correct
        if k > n - c:
            return 1.0

        return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


class ExecutionAccuracy(Metric):
    """
    Execution accuracy metric.

    Measures the percentage of code samples that execute successfully
    and pass all test cases.
    """

    @property
    def name(self) -> str:
        """Return metric name."""
        return "execution_accuracy"

    def compute(self, predictions: list[bool], references: Any = None, **kwargs) -> float:
        """
        Compute execution accuracy.

        Args:
            predictions: List of boolean pass/fail results
            references: Not used for this metric
            **kwargs: Additional parameters

        Returns:
            Accuracy as a float between 0 and 1
        """
        if not predictions:
            return 0.0

        return sum(predictions) / len(predictions)


class SyntaxErrorRate(Metric):
    """Measure the rate of syntax errors in generated code."""

    @property
    def name(self) -> str:
        """Return metric name."""
        return "syntax_error_rate"

    def compute(self, predictions: list[dict[str, Any]], references: Any = None, **kwargs) -> float:
        """
        Compute syntax error rate.

        Args:
            predictions: List of execution results with error information
            references: Not used
            **kwargs: Additional parameters

        Returns:
            Syntax error rate as a float between 0 and 1
        """
        if not predictions:
            return 0.0

        syntax_errors = sum(
            1 for pred in predictions if pred.get("error", "").startswith("SyntaxError")
        )

        return syntax_errors / len(predictions)
