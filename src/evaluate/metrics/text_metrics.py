"""Text-based metrics for evaluation."""

import re
from collections import Counter
from typing import Any

from evaluate.metrics.base import Metric


class ExactMatch(Metric):
    """Exact match metric."""

    @property
    def name(self) -> str:
        """Return metric name."""
        return "exact_match"

    def compute(self, predictions: list[str], references: list[str], **kwargs) -> float:
        """
        Compute exact match accuracy.

        Args:
            predictions: List of predicted strings
            references: List of reference strings
            **kwargs: Additional parameters

        Returns:
            Exact match accuracy as a float between 0 and 1
        """
        if not predictions or not references:
            return 0.0

        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        matches = sum(pred.strip() == ref.strip() for pred, ref in zip(predictions, references))
        return matches / len(predictions)


class BLEU(Metric):
    """BLEU score for text similarity."""

    def __init__(self, n: int = 4):
        """
        Initialize BLEU metric.

        Args:
            n: Maximum n-gram order (default: 4)
        """
        self.n = n

    @property
    def name(self) -> str:
        """Return metric name."""
        return f"bleu_{self.n}"

    def compute(self, predictions: list[str], references: list[str], **kwargs) -> float:
        """
        Compute BLEU score.

        Args:
            predictions: List of predicted strings
            references: List of reference strings
            **kwargs: Additional parameters

        Returns:
            BLEU score as a float between 0 and 1
        """
        if not predictions or not references:
            return 0.0

        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        # Simple BLEU implementation (can be replaced with external library)
        scores = []
        for pred, ref in zip(predictions, references):
            score = self._compute_bleu_single(pred, ref)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def _compute_bleu_single(self, prediction: str, reference: str) -> float:
        """Compute BLEU for a single prediction-reference pair."""
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        if not pred_tokens or not ref_tokens:
            return 0.0

        # Brevity penalty
        bp = 1.0 if len(pred_tokens) > len(ref_tokens) else (
            pow(2.718281828, 1 - len(ref_tokens) / len(pred_tokens))
        )

        # N-gram precisions (only compute up to the length of the shortest sequence)
        max_n = min(self.n, len(pred_tokens), len(ref_tokens))
        precisions = []
        
        for n in range(1, max_n + 1):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)

            if not pred_ngrams:
                precisions.append(0.0)
                continue

            matches = sum((pred_ngrams & ref_ngrams).values())
            precision = matches / sum(pred_ngrams.values())
            precisions.append(precision)

        # If no valid precisions, return 0
        if not precisions or any(p == 0 for p in precisions):
            return 0.0

        # Geometric mean of precisions
        if len(precisions) == 1:
            geo_mean = precisions[0]
        else:
            # Product of all precisions raised to 1/len(precisions)
            product = 1.0
            for p in precisions:
                product *= p
            geo_mean = pow(product, 1.0 / len(precisions))

        return bp * geo_mean

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization."""
        return re.findall(r'\w+|[^\w\s]', text.lower())

    @staticmethod
    def _get_ngrams(tokens: list[str], n: int) -> Counter:
        """Extract n-grams from tokens."""
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        return Counter(ngrams)


class ROUGE(Metric):
    """ROUGE-L score for text similarity."""

    @property
    def name(self) -> str:
        """Return metric name."""
        return "rouge_l"

    def compute(self, predictions: list[str], references: list[str], **kwargs) -> float:
        """
        Compute ROUGE-L score.

        Args:
            predictions: List of predicted strings
            references: List of reference strings
            **kwargs: Additional parameters

        Returns:
            ROUGE-L F1 score as a float between 0 and 1
        """
        if not predictions or not references:
            return 0.0

        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        scores = []
        for pred, ref in zip(predictions, references):
            score = self._compute_rouge_l_single(pred, ref)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def _compute_rouge_l_single(self, prediction: str, reference: str) -> float:
        """Compute ROUGE-L for a single prediction-reference pair."""
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        if not pred_tokens or not ref_tokens:
            return 0.0

        # Compute LCS length
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)

        if lcs_length == 0:
            return 0.0

        # Compute precision and recall
        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(ref_tokens)

        # Compute F1 score
        if precision + recall == 0:
            return 0.0

        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization."""
        return re.findall(r'\w+|[^\w\s]', text.lower())

    @staticmethod
    def _lcs_length(seq1: list, seq2: list) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

