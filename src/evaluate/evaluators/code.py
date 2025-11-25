"""Code generation evaluator."""

from typing import Any

from evaluate.evaluators.base import AggregatedResults, Evaluator, EvaluationResult
from evaluate.execution.sandbox import CodeExecutor
from evaluate.metrics.code_metrics import ExecutionAccuracy, PassAtK


class CodeEvaluator(Evaluator):
    """Evaluator for code generation tasks with Pass@k metrics."""

    def __init__(self, k_values: list[int] | None = None, timeout: int = 30):
        """
        Initialize code evaluator.

        Args:
            k_values: List of k values for Pass@k metrics (default: [1, 5, 10])
            timeout: Execution timeout in seconds
        """
        self.k_values = k_values or [1, 5, 10]
        self.executor = CodeExecutor(timeout=timeout)
        self.pass_at_k_metrics = {k: PassAtK(k=k) for k in self.k_values}
        self.accuracy_metric = ExecutionAccuracy()

    def evaluate_sample(self, sample: dict[str, Any], predictions: list[str]) -> EvaluationResult:
        """
        Evaluate code predictions for a single sample.

        Args:
            sample: Sample containing 'test' and 'entry_point' keys
            predictions: List of generated code solutions

        Returns:
            EvaluationResult with execution results and Pass@k metrics
        """
        task_id = sample.get("task_id", "unknown")
        test_code = sample.get("test", "")
        entry_point = sample.get("entry_point")

        if not test_code:
            return EvaluationResult(
                task_id=task_id,
                success=False,
                predictions=predictions,
                error="No test code provided",
            )

        # Execute each prediction
        execution_results = []
        for code in predictions:
            result = self.executor.execute(code, test_code, entry_point)
            execution_results.append(result.success)

        # Compute metrics
        success = any(execution_results)
        metrics = {}

        # Compute pass@k for different k values
        for k in self.k_values:
            if len(predictions) >= k:
                metrics[f"pass@{k}"] = self.pass_at_k_metrics[k].compute(
                    execution_results, n=len(predictions)
                )

        # Also store the raw pass rate
        metrics["pass_rate"] = sum(execution_results) / len(execution_results)

        return EvaluationResult(
            task_id=task_id,
            success=success,
            predictions=predictions,
            ground_truth=sample.get("canonical_solution"),
            metrics=metrics,
            metadata={
                "execution_results": execution_results,
                "num_predictions": len(predictions),
                "num_passed": sum(execution_results),
            },
        )

    def aggregate_results(self, results: list[EvaluationResult]) -> AggregatedResults:
        """
        Aggregate evaluation results across samples.

        Args:
            results: List of individual evaluation results

        Returns:
            AggregatedResults with aggregated Pass@k metrics
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

        # Aggregate metrics
        aggregated_metrics = {}

        # Compute overall pass@k using unbiased estimator
        for k in self.k_values:
            # Collect all execution results
            all_n = []
            all_c = []

            for result in results:
                if "execution_results" in result.metadata:
                    exec_results = result.metadata["execution_results"]
                    all_n.append(len(exec_results))
                    all_c.append(sum(exec_results))

            if all_n:
                # Compute pass@k for each sample and average
                pass_at_k_scores = []
                for n, c in zip(all_n, all_c):
                    if n >= k:
                        score = PassAtK.compute_pass_at_k(n, c, k)
                        pass_at_k_scores.append(score)

                if pass_at_k_scores:
                    aggregated_metrics[f"pass@{k}"] = sum(pass_at_k_scores) / len(pass_at_k_scores)

        # Average pass rate
        pass_rates = [r.metrics.get("pass_rate", 0.0) for r in results if "pass_rate" in r.metrics]
        if pass_rates:
            aggregated_metrics["avg_pass_rate"] = sum(pass_rates) / len(pass_rates)

        # Execution accuracy (at least one solution passes)
        aggregated_metrics["execution_accuracy"] = successful / len(results)

        return AggregatedResults(
            total_samples=len(results),
            successful=successful,
            failed=failed,
            metrics=aggregated_metrics,
            per_sample_results=results,
            metadata={
                "k_values": self.k_values,
                "timeout": self.executor.timeout,
            },
        )

