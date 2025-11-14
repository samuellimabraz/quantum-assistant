"""Deduplication utilities."""

from difflib import SequenceMatcher

from synthetic_data.models import Sample


class Deduplicator:
    """Remove duplicate or highly similar samples."""

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize deduplicator.

        Args:
            similarity_threshold: Threshold for considering samples similar (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold

    def deduplicate(self, samples: list[Sample]) -> list[Sample]:
        """
        Remove duplicate samples.

        Args:
            samples: List of samples to deduplicate

        Returns:
            Deduplicated list of samples
        """
        if not samples:
            return samples

        unique_samples = []
        seen_questions = []

        for sample in samples:
            if not self._is_duplicate(sample.question, seen_questions):
                unique_samples.append(sample)
                seen_questions.append(sample.question)

        removed = len(samples) - len(unique_samples)
        if removed > 0:
            print(f"Removed {removed} duplicate/similar samples")

        return unique_samples

    def _is_duplicate(self, question: str, seen_questions: list[str]) -> bool:
        """Check if question is duplicate or too similar to existing ones."""
        for seen in seen_questions:
            similarity = self._similarity(question, seen)
            if similarity >= self.similarity_threshold:
                return True
        return False

    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity ratio between two strings."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
