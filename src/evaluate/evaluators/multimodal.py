"""Multimodal evaluator for synthetic dataset."""

from typing import Any

from evaluate.evaluators.base import AggregatedResults, Evaluator, EvaluationResult
from evaluate.execution.sandbox import CodeExecutor
from evaluate.metrics.code_metrics import PassAtK
from evaluate.metrics.text_metrics import BLEU, ROUGE, ExactMatch


class MultimodalEvaluator(Evaluator):
    """Evaluator for multimodal synthetic dataset with various question types."""

    def __init__(self, timeout: int = 30):
        """
        Initialize multimodal evaluator.

        Args:
            timeout: Execution timeout for code samples
        """
        self.executor = CodeExecutor(timeout=timeout)
        self.exact_match = ExactMatch()
        self.bleu = BLEU(n=4)
        self.rouge = ROUGE()
        self.pass_at_1 = PassAtK(k=1)

    def evaluate_sample(self, sample: dict[str, Any], predictions: list[str]) -> EvaluationResult:
        """
        Evaluate predictions for a multimodal sample.

        Args:
            sample: Sample with 'question_type', 'answer', and optional test info
            predictions: List of model predictions

        Returns:
            EvaluationResult with appropriate metrics based on question type
        """
        task_id = sample.get("task_id", sample.get("id", "unknown"))
        question_type = sample.get("question_type", "qa")
        ground_truth = sample.get("answer", "")

        # Choose evaluation strategy based on question type
        if question_type in ["code", "function_completion"]:
            return self._evaluate_code(task_id, sample, predictions, ground_truth)
        else:
            return self._evaluate_text(task_id, question_type, predictions, ground_truth)

    def _evaluate_code(
        self, task_id: str, sample: dict, predictions: list[str], ground_truth: str
    ) -> EvaluationResult:
        """Evaluate code generation predictions."""
        test_code = sample.get("test")

        if test_code:
            # Execute code if test is available
            entry_point = sample.get("entry_point")
            execution_results = []

            for code in predictions:
                result = self.executor.execute(code, test_code, entry_point)
                execution_results.append(result.success)

            success = any(execution_results)
            metrics = {
                "pass@1": self.pass_at_1.compute(execution_results),
                "pass_rate": sum(execution_results) / len(execution_results),
            }

            return EvaluationResult(
                task_id=task_id,
                success=success,
                predictions=predictions,
                ground_truth=ground_truth,
                metrics=metrics,
                metadata={
                    "execution_results": execution_results,
                    "question_type": "code",
                },
            )
        else:
            # Fall back to text-based evaluation
            return self._evaluate_text(task_id, "code", predictions, ground_truth)

    def _evaluate_text(
        self, task_id: str, question_type: str, predictions: list[str], ground_truth: str
    ) -> EvaluationResult:
        """Evaluate text-based predictions."""
        # Use first prediction for text metrics
        prediction = predictions[0] if predictions else ""

        metrics = {}

        # Compute text similarity metrics
        try:
            metrics["bleu"] = self.bleu.compute([prediction], [ground_truth])
            metrics["rouge_l"] = self.rouge.compute([prediction], [ground_truth])
            metrics["exact_match"] = self.exact_match.compute([prediction], [ground_truth])
        except Exception as e:
            metrics["error"] = str(e)

        # Success is based on a combination of metrics
        # For QA and summary, we consider ROUGE-L > 0.3 as reasonable
        # For caption, we consider BLEU > 0.2 as reasonable
        success = False
        if question_type in ["qa", "summary"]:
            success = metrics.get("rouge_l", 0.0) > 0.3
        elif question_type == "caption":
            success = metrics.get("bleu", 0.0) > 0.2
        else:
            success = metrics.get("exact_match", 0.0) > 0.5

        return EvaluationResult(
            task_id=task_id,
            success=success,
            predictions=predictions,
            ground_truth=ground_truth,
            metrics=metrics,
            metadata={"question_type": question_type},
        )

    def aggregate_results(self, results: list[EvaluationResult]) -> AggregatedResults:
        """
        Aggregate evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            AggregatedResults with metrics grouped by question type
        """
        if not results:
            return AggregatedResults(
                total_samples=0,
                successful=0,
                failed=0,
                metrics={},
                per_sample_results=[],
            )

        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        # Group results by question type
        by_type = {}
        for result in results:
            qtype = result.metadata.get("question_type", "unknown")
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(result)

        # Aggregate metrics overall and by type
        aggregated_metrics = {}

        # Overall metrics
        all_metrics = {}
        for metric_name in ["pass@1", "pass_rate", "bleu", "rouge_l", "exact_match"]:
            values = [r.metrics.get(metric_name) for r in results if metric_name in r.metrics]
            if values:
                all_metrics[metric_name] = sum(values) / len(values)

        aggregated_metrics["overall"] = all_metrics

        # Per-type metrics
        for qtype, type_results in by_type.items():
            type_metrics = {}
            for metric_name in ["pass@1", "pass_rate", "bleu", "rouge_l", "exact_match"]:
                values = [
                    r.metrics.get(metric_name) for r in type_results if metric_name in r.metrics
                ]
                if values:
                    type_metrics[metric_name] = sum(values) / len(values)

            type_metrics["count"] = len(type_results)
            type_metrics["success_rate"] = sum(1 for r in type_results if r.success) / len(
                type_results
            )
            aggregated_metrics[f"by_type.{qtype}"] = type_metrics

        return AggregatedResults(
            total_samples=len(results),
            successful=successful,
            failed=failed,
            metrics=aggregated_metrics,
            per_sample_results=results,
            metadata={"question_types": list(by_type.keys())},
        )

