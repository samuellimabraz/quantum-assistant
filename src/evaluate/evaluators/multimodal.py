"""Multimodal evaluator for synthetic dataset samples."""

from typing import Any

from evaluate.evaluators.base import AggregatedResults, Evaluator, EvaluationResult
from evaluate.execution.sandbox import CodeExecutor
from evaluate.metrics.code_metrics import PassAtK
from evaluate.metrics.text_metrics import BLEU, ROUGE


class QuestionType:
    """Valid question types in synthetic dataset."""

    FUNCTION_COMPLETION = "function_completion"
    CODE_GENERATION = "code_generation"
    QA = "qa"

    CODE_TYPES = {FUNCTION_COMPLETION, CODE_GENERATION}
    ALL_TYPES = {FUNCTION_COMPLETION, CODE_GENERATION, QA}


class MultimodalEvaluator(Evaluator):
    """
    Evaluator for multimodal synthetic dataset samples.

    Handles three question types:
    - function_completion: Code stub completion with unit test verification
    - code_generation: Full code generation with unit test verification
    - qa: Text-based Q&A with similarity metrics (BLEU, ROUGE-L)
    """

    def __init__(self, timeout: int = 30, k_values: list[int] | None = None):
        """
        Initialize multimodal evaluator.

        Args:
            timeout: Execution timeout for code samples
            k_values: K values for Pass@k computation
        """
        self.executor = CodeExecutor(timeout=timeout)
        self.k_values = k_values or [1]
        self.bleu = BLEU(n=4)
        self.rouge = ROUGE()

    def evaluate_sample(self, sample: dict[str, Any], predictions: list[str]) -> EvaluationResult:
        """
        Evaluate predictions for a sample.

        Args:
            sample: Sample dictionary with question_type, answer, test_code, entry_point
            predictions: List of model predictions

        Returns:
            EvaluationResult with metrics based on question type
        """
        task_id = sample.get("task_id", sample.get("id", "unknown"))
        question_type = sample.get("question_type", sample.get("type", "qa"))
        ground_truth = sample.get("answer", "")

        if question_type in QuestionType.CODE_TYPES:
            return self._evaluate_code(task_id, sample, predictions, ground_truth)
        else:
            return self._evaluate_text(task_id, question_type, predictions, ground_truth)

    def _evaluate_code(
        self,
        task_id: str,
        sample: dict[str, Any],
        predictions: list[str],
        ground_truth: str,
    ) -> EvaluationResult:
        """Evaluate code generation predictions with test execution."""
        test_code = sample.get("test_code", "")
        entry_point = sample.get("entry_point", "")

        if not test_code:
            # No test code - fall back to syntax check
            return self._evaluate_syntax_only(task_id, predictions, ground_truth)

        execution_results = []
        for code in predictions:
            result = self.executor.execute(code, test_code, entry_point)
            execution_results.append(result.success)

        passed_count = sum(execution_results)
        success = passed_count > 0

        metrics = {
            "pass_rate": passed_count / len(predictions) if predictions else 0.0,
        }

        # Compute Pass@k metrics
        for k in self.k_values:
            if len(predictions) >= k:
                metrics[f"pass@{k}"] = PassAtK.compute_pass_at_k(
                    n=len(predictions),
                    c=passed_count,
                    k=k,
                )

        return EvaluationResult(
            task_id=task_id,
            success=success,
            predictions=predictions,
            ground_truth=ground_truth,
            metrics=metrics,
            metadata={
                "question_type": sample.get("question_type", "code_generation"),
                "execution_results": execution_results,
                "num_passed": passed_count,
            },
        )

    def _evaluate_syntax_only(
        self,
        task_id: str,
        predictions: list[str],
        ground_truth: str,
    ) -> EvaluationResult:
        """Evaluate code with syntax check only (no test code available)."""
        syntax_results = []
        for code in predictions:
            try:
                compile(code, "<string>", "exec")
                syntax_results.append(True)
            except SyntaxError:
                syntax_results.append(False)

        passed_count = sum(syntax_results)
        success = passed_count > 0

        metrics = {
            "syntax_pass_rate": passed_count / len(predictions) if predictions else 0.0,
        }

        return EvaluationResult(
            task_id=task_id,
            success=success,
            predictions=predictions,
            ground_truth=ground_truth,
            metrics=metrics,
            metadata={
                "question_type": "code",
                "syntax_results": syntax_results,
                "evaluation_mode": "syntax_only",
            },
        )

    def _evaluate_text(
        self,
        task_id: str,
        question_type: str,
        predictions: list[str],
        ground_truth: str,
    ) -> EvaluationResult:
        """Evaluate text-based predictions with similarity metrics."""
        prediction = predictions[0] if predictions else ""

        metrics = {}
        try:
            metrics["bleu"] = self.bleu.compute([prediction], [ground_truth])
            metrics["rouge_l"] = self.rouge.compute([prediction], [ground_truth])
        except Exception as e:
            metrics["error"] = str(e)

        # Success based on ROUGE-L threshold
        success = metrics.get("rouge_l", 0.0) > 0.3

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

        # Group by question type
        by_type: dict[str, list[EvaluationResult]] = {}
        for result in results:
            qtype = result.metadata.get("question_type", "unknown")
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(result)

        aggregated_metrics: dict[str, Any] = {}

        # Overall metrics
        overall: dict[str, float] = {}

        # Aggregate Pass@k for code types
        code_results = [
            r for r in results
            if r.metadata.get("question_type") in QuestionType.CODE_TYPES
        ]
        if code_results:
            for k in self.k_values:
                key = f"pass@{k}"
                values = [r.metrics.get(key) for r in code_results if key in r.metrics]
                if values:
                    overall[key] = sum(values) / len(values)

        # Aggregate text metrics for QA
        qa_results = [
            r for r in results
            if r.metadata.get("question_type") == QuestionType.QA
        ]
        if qa_results:
            for metric_name in ["bleu", "rouge_l"]:
                values = [r.metrics.get(metric_name) for r in qa_results if metric_name in r.metrics]
                if values:
                    overall[metric_name] = sum(values) / len(values)

        aggregated_metrics["overall"] = overall

        # Per-type metrics
        for qtype, type_results in by_type.items():
            type_metrics: dict[str, float] = {
                "count": float(len(type_results)),
                "success_rate": sum(1 for r in type_results if r.success) / len(type_results),
            }

            if qtype in QuestionType.CODE_TYPES:
                for k in self.k_values:
                    key = f"pass@{k}"
                    values = [r.metrics.get(key) for r in type_results if key in r.metrics]
                    if values:
                        type_metrics[key] = sum(values) / len(values)
            else:
                for metric_name in ["bleu", "rouge_l"]:
                    values = [r.metrics.get(metric_name) for r in type_results if metric_name in r.metrics]
                    if values:
                        type_metrics[metric_name] = sum(values) / len(values)

            aggregated_metrics[f"by_type.{qtype}"] = type_metrics

        return AggregatedResults(
            total_samples=len(results),
            successful=successful,
            failed=failed,
            metrics=aggregated_metrics,
            per_sample_results=results,
            metadata={"question_types": list(by_type.keys())},
        )
