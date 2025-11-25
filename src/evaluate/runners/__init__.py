"""Evaluation runners."""

from evaluate.runners.qiskit_humaneval import QiskitHumanEvalRunner
from evaluate.runners.synthetic import SyntheticDatasetRunner

__all__ = ["QiskitHumanEvalRunner", "SyntheticDatasetRunner"]

