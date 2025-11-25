"""Qiskit HumanEval runner for code generation evaluation."""

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from evaluate.evaluators.code import CodeEvaluator
from evaluate.evaluators.base import AggregatedResults, EvaluationResult
from evaluate.execution.sandbox import CodeExecutor
from evaluate.utils.results import ResultsManager
from models.client import LLMClient, VLMClient, Message


class DatasetType(str, Enum):
    """Type of Qiskit HumanEval dataset."""

    NORMAL = "normal"  # Prompt has imports + function signature, model completes
    HARD = "hard"  # Prompt is natural language, model generates full code


@dataclass
class SampleResult:
    """Detailed result for a single sample."""

    task_id: str
    prompt: str
    generated_solutions: list[str]
    combined_codes: list[str]
    execution_results: list[dict[str, Any]]
    canonical_solution: str | None
    canonical_combined: str | None
    canonical_passed: bool | None
    metrics: dict[str, float]
    success: bool


class QiskitHumanEvalRunner:
    """
    Runner for Qiskit HumanEval benchmark evaluation.

    Follows the evaluation methodology from the Qiskit HumanEval paper:
    "Qiskit HumanEval: An Evaluation Benchmark For Quantum Code Generative Models"
    (arXiv:2406.14712)

    Key aspects:
    - Supports both normal (completion) and hard (full generation) dataset types
    - Uses the standard prompt format from the dataset
    - Generates multiple solutions per task for Pass@k evaluation
    - Uses unbiased Pass@k estimator from HumanEval paper
    - Evaluates functional correctness via unit test execution
    - Saves detailed raw results including input/output for debugging
    """

    def __init__(
        self,
        dataset_path: Path | str,
        model_client: LLMClient | VLMClient,
        k_values: list[int] | None = None,
        num_samples_per_task: int = 1,
        timeout: int = 30,
        max_concurrent: int = 10,
        dataset_type: DatasetType | str | None = None,
    ):
        """
        Initialize Qiskit HumanEval runner.

        Args:
            dataset_path: Path to Qiskit HumanEval JSON dataset
            model_client: Model client for inference
            k_values: List of k values for Pass@k metrics
            num_samples_per_task: Number of solutions to generate per task
            timeout: Execution timeout in seconds
            max_concurrent: Maximum concurrent API requests
            dataset_type: Dataset type (normal/hard), auto-detected if None
        """
        self.dataset_path = Path(dataset_path)
        self.model_client = model_client
        self.num_samples = num_samples_per_task
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.evaluator = CodeEvaluator(k_values=k_values or [1, 5, 10], timeout=timeout)
        self.executor = CodeExecutor(timeout=timeout)
        self.console = Console()

        # Determine dataset type
        if dataset_type:
            self.dataset_type = (
                DatasetType(dataset_type) if isinstance(dataset_type, str) else dataset_type
            )
        else:
            self.dataset_type = self._detect_dataset_type()

    def _detect_dataset_type(self) -> DatasetType:
        """Auto-detect dataset type based on filename and content."""
        filename = self.dataset_path.name.lower()
        if "hard" in filename:
            return DatasetType.HARD
        return DatasetType.NORMAL

    def load_dataset(self) -> list[dict[str, Any]]:
        """Load Qiskit HumanEval dataset."""
        with open(self.dataset_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("samples", data.get("data", []))

        return data

    def _is_completion_prompt(self, prompt: str) -> bool:
        """Check if prompt is a completion-style (has function signature)."""
        # Completion prompts have "def function_name(" pattern
        return bool(re.search(r"def\s+\w+\s*\([^)]*\)\s*:", prompt))

    def extract_code_from_response(self, response: str, entry_point: str | None = None) -> str:
        """
        Extract code from model response, handling markdown blocks.

        For responses with multiple code blocks, prioritizes:
        1. Code blocks containing the entry_point function definition
        2. The largest code block (most likely to be the main implementation)
        """
        # Find all markdown code blocks
        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)

        if not code_blocks:
            # No code blocks found, return stripped response
            return response.strip("\n")

        if len(code_blocks) == 1:
            return code_blocks[0].rstrip()

        # Multiple code blocks - find the best one
        if entry_point:
            # Prioritize blocks containing the function definition
            for block in code_blocks:
                if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", block):
                    return block.rstrip()

        # Fallback: return the largest code block (most likely the main implementation)
        largest_block = max(code_blocks, key=len)
        return largest_block.rstrip()

    def _extract_function_body(self, code: str, entry_point: str) -> str | None:
        """
        Extract only the function body from generated code.

        Args:
            code: Full generated code that may contain a function definition
            entry_point: Name of the function to extract

        Returns:
            The function body (indented content after signature), or None if not found
        """
        # Pattern to match function definition and capture the body
        pattern = (
            rf"def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*(?:->[^:]+)?:\s*\n((?:[ \t]+.*\n?)+)"
        )
        match = re.search(pattern, code)
        if match:
            return match.group(1).rstrip()
        return None

    def _generated_has_function_def(self, code: str, entry_point: str) -> bool:
        """Check if the generated code contains a function definition for entry_point."""
        pattern = rf"def\s+{re.escape(entry_point)}\s*\("
        return bool(re.search(pattern, code))

    def combine_code(self, sample: dict[str, Any], generated: str) -> str:
        """
        Combine prompt and generated code based on dataset type.

        For normal dataset: prompt (imports + signature) + generated (body)
        For hard dataset: generated should be full code

        Handles the case where the model generates full code (with function
        definition) instead of just the function body for normal dataset.
        """
        prompt = sample.get("prompt", "")
        entry_point = sample.get("entry_point", "")
        generated = self.extract_code_from_response(generated, entry_point)

        if self.dataset_type == DatasetType.NORMAL:
            # Check if model generated full code with function definition
            if entry_point and self._generated_has_function_def(generated, entry_point):
                # Model generated full code - use it directly instead of concatenating
                # This handles cases where the model doesn't understand it should
                # only complete the function body
                return generated
            else:
                # Model generated just the function body - concatenate with prompt
                # Ensure there's a newline separator to avoid syntax errors
                if not generated.startswith("\n"):
                    generated = "\n" + generated
                return prompt + generated
        else:
            # Hard: model generates full code
            return generated

    def combine_canonical(self, sample: dict[str, Any]) -> str:
        """Combine prompt with canonical solution for verification."""
        prompt = sample.get("prompt", "")
        canonical = sample.get("canonical_solution", "")

        if self.dataset_type == DatasetType.NORMAL:
            return prompt + canonical
        else:
            return canonical

    def create_messages(
        self,
        sample: dict[str, Any],
        system_prompt: str | None = None,
    ) -> list[Message]:
        """
        Create messages list for code generation with proper system/user roles.

        Args:
            sample: Sample from dataset
            system_prompt: Optional system prompt (None means no system message)

        Returns:
            List of Message objects with proper roles
        """
        messages = []

        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        prompt = sample.get("prompt", "")
        messages.append(Message(role="user", content=prompt))

        return messages

    def execute_code(self, code: str, test_code: str, entry_point: str | None) -> dict[str, Any]:
        """
        Execute code with test and return detailed result.

        Returns:
            Dict with success, error, output, and timeout info
        """
        result = self.executor.execute(code, test_code, entry_point)
        return {
            "success": result.success,
            "error": result.error if not result.success else "",
            "output": result.output,
            "timeout": result.timeout,
        }

    async def generate_solutions_async(
        self,
        samples: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> dict[str, list[str]]:
        """
        Generate solutions for all samples asynchronously with efficient batching.

        Args:
            samples: List of samples from dataset
            system_prompt: Optional system prompt (None means no system message)

        Returns:
            Dictionary mapping task_id to list of generated solutions
        """
        all_messages = []
        task_ids = []

        for sample in samples:
            task_id = sample["task_id"]
            messages = self.create_messages(sample, system_prompt)

            for _ in range(self.num_samples):
                all_messages.append(messages)
                task_ids.append(task_id)

        total_requests = len(all_messages)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Generating solutions...", total=total_requests)

            def update_progress(count):
                progress.update(task, completed=count)

            all_generated = await self.model_client.generate_batch_async(
                all_messages,
                max_concurrent=self.max_concurrent,
                progress_callback=update_progress,
            )

        solutions = {}
        for task_id, generated_code in zip(task_ids, all_generated):
            if task_id not in solutions:
                solutions[task_id] = []
            solutions[task_id].append(generated_code)

        return solutions

    def generate_solutions(
        self,
        samples: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> dict[str, list[str]]:
        """Generate solutions for all samples (sync wrapper)."""
        return asyncio.run(self.generate_solutions_async(samples, system_prompt))

    def evaluate_sample_detailed(
        self,
        sample: dict[str, Any],
        solutions: list[str],
        verify_canonical: bool = False,
    ) -> SampleResult:
        """
        Evaluate a single sample with detailed results.

        Args:
            sample: Sample from dataset
            solutions: List of generated solutions
            verify_canonical: Whether to also verify the canonical solution

        Returns:
            SampleResult with detailed execution info
        """
        task_id = sample.get("task_id", "unknown")
        test_code = sample.get("test", "")
        entry_point = sample.get("entry_point")
        canonical_solution = sample.get("canonical_solution", "")

        # Combine and execute each generated solution
        combined_codes = []
        execution_results = []

        for solution in solutions:
            combined = self.combine_code(sample, solution)
            combined_codes.append(combined)

            exec_result = self.execute_code(combined, test_code, entry_point)
            execution_results.append(exec_result)

        # Verify canonical solution if requested
        canonical_combined = None
        canonical_passed = None

        if verify_canonical:
            canonical_combined = self.combine_canonical(sample)
            canonical_result = self.execute_code(canonical_combined, test_code, entry_point)
            canonical_passed = canonical_result["success"]

        # Compute metrics
        passed_count = sum(1 for r in execution_results if r["success"])
        success = passed_count > 0

        metrics = {"pass_rate": passed_count / len(solutions) if solutions else 0.0}

        for k in self.evaluator.k_values:
            if len(solutions) >= k:
                from evaluate.metrics.code_metrics import PassAtK

                n = len(solutions)
                c = passed_count
                metrics[f"pass@{k}"] = PassAtK.compute_pass_at_k(n, c, k)

        return SampleResult(
            task_id=task_id,
            prompt=sample.get("prompt", ""),
            generated_solutions=solutions,
            combined_codes=combined_codes,
            execution_results=execution_results,
            canonical_solution=canonical_solution,
            canonical_combined=canonical_combined,
            canonical_passed=canonical_passed,
            metrics=metrics,
            success=success,
        )

    def evaluate(
        self,
        samples: list[dict[str, Any]],
        system_prompt: str | None = None,
        save_results: Path | None = None,
        verify_canonical: bool = False,
        model_name: str | None = None,
        run_timestamp: datetime | None = None,
    ) -> AggregatedResults:
        """
        Run full evaluation pipeline.

        Args:
            samples: List of samples to evaluate
            system_prompt: Optional system prompt (None means no system message)
            save_results: Optional path to save detailed results
            verify_canonical: Whether to verify canonical solutions work
            model_name: Model name for metadata (auto-detected if None)
            run_timestamp: Timestamp for this run (defaults to now)

        Returns:
            AggregatedResults with evaluation metrics
        """
        if run_timestamp is None:
            run_timestamp = datetime.now()
        self.console.print("\n[bold cyan]Qiskit HumanEval Evaluation[/bold cyan]")
        self.console.print(f"Dataset type: {self.dataset_type.value}")
        self.console.print(f"Samples: {len(samples)}")
        self.console.print(f"Solutions per task: {self.num_samples}")
        self.console.print(f"Pass@k values: {self.evaluator.k_values}")
        if system_prompt:
            prompt_preview = (
                system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt
            )
            self.console.print(f"System prompt: {prompt_preview}")
        if verify_canonical:
            self.console.print("[yellow]Canonical solution verification: enabled[/yellow]")
        self.console.print()

        # Generate solutions
        solutions_dict = self.generate_solutions(samples, system_prompt)

        # Evaluate each sample with detailed results
        self.console.print("\n[bold]Evaluating solutions...[/bold]")
        detailed_results: list[SampleResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            eval_task = progress.add_task("Evaluating...", total=len(samples))

            for sample in samples:
                task_id = sample["task_id"]
                solutions = solutions_dict.get(task_id, [])

                result = self.evaluate_sample_detailed(
                    sample, solutions, verify_canonical=verify_canonical
                )
                detailed_results.append(result)

                progress.update(eval_task, advance=1)

        # Convert to EvaluationResult for aggregation
        eval_results = []
        for dr in detailed_results:
            eval_results.append(
                EvaluationResult(
                    task_id=dr.task_id,
                    success=dr.success,
                    predictions=dr.generated_solutions,
                    ground_truth=dr.canonical_solution,
                    metrics=dr.metrics,
                    metadata={
                        "execution_results": [r["success"] for r in dr.execution_results],
                        "num_predictions": len(dr.generated_solutions),
                        "num_passed": sum(1 for r in dr.execution_results if r["success"]),
                    },
                )
            )

        # Aggregate results
        aggregated = self.evaluator.aggregate_results(eval_results)

        # Save detailed results
        if save_results:
            self._save_results(
                save_results,
                samples,
                detailed_results,
                aggregated,
                system_prompt,
                verify_canonical,
                model_name,
                run_timestamp,
            )

        # Print summary
        self._print_summary(aggregated, detailed_results, verify_canonical)

        return aggregated

    def _save_results(
        self,
        save_path: Path,
        samples: list[dict[str, Any]],
        detailed_results: list[SampleResult],
        aggregated: AggregatedResults,
        system_prompt: str | None,
        verify_canonical: bool,
        model_name: str | None,
        run_timestamp: datetime,
    ) -> None:
        """Save detailed results to JSON."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Build comprehensive metadata using ResultsManager
        metadata = ResultsManager.build_metadata(
            dataset_path=self.dataset_path,
            dataset_type="qiskit_humaneval",
            dataset_variant=self.dataset_type.value,
            model_name=model_name or "unknown",
            num_samples=len(samples),
            num_samples_per_task=self.num_samples,
            k_values=self.evaluator.k_values,
            timeout=self.timeout,
            system_prompt=system_prompt,
            verify_canonical=verify_canonical,
            timestamp=run_timestamp,
        )

        output_data = {
            "metadata": metadata,
            "metrics": aggregated.metrics,
            "summary": {
                "total_samples": aggregated.total_samples,
                "successful": aggregated.successful,
                "failed": aggregated.failed,
                "success_rate": aggregated.success_rate,
            },
            "results": [],
        }

        # Add canonical verification summary if enabled
        if verify_canonical:
            canonical_passed = sum(1 for r in detailed_results if r.canonical_passed is True)
            canonical_failed = sum(1 for r in detailed_results if r.canonical_passed is False)
            output_data["canonical_verification"] = {
                "passed": canonical_passed,
                "failed": canonical_failed,
                "total": len(detailed_results),
            }

        # Add detailed per-sample results
        for dr in detailed_results:
            sample_result = {
                "task_id": dr.task_id,
                "success": dr.success,
                "metrics": dr.metrics,
                "prompt": dr.prompt,
                "generated_solutions": dr.generated_solutions,
                "combined_codes": dr.combined_codes,
                "execution_results": dr.execution_results,
            }

            if verify_canonical:
                sample_result["canonical"] = {
                    "solution": dr.canonical_solution,
                    "combined_code": dr.canonical_combined,
                    "passed": dr.canonical_passed,
                }

            output_data["results"].append(sample_result)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.console.print(f"\n[green]Results saved to: {save_path}[/green]")

    def _print_summary(
        self,
        results: AggregatedResults,
        detailed_results: list[SampleResult],
        verify_canonical: bool,
    ) -> None:
        """Print evaluation summary."""
        from rich.table import Table

        self.console.print("\n[bold cyan]Evaluation Results[/bold cyan]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Samples", str(results.total_samples))
        table.add_row("Successful", str(results.successful))
        table.add_row("Failed", str(results.failed))
        table.add_row("Success Rate", f"{results.success_rate:.1%}")

        self.console.print(table)

        # Canonical verification results
        if verify_canonical:
            canonical_passed = sum(1 for r in detailed_results if r.canonical_passed is True)
            canonical_failed = sum(1 for r in detailed_results if r.canonical_passed is False)

            self.console.print("\n[bold]Canonical Solution Verification:[/bold]")
            canon_table = Table(show_header=True, header_style="bold magenta")
            canon_table.add_column("Metric", style="cyan")
            canon_table.add_column("Value", style="green", justify="right")

            canon_table.add_row("Passed", str(canonical_passed))
            canon_table.add_row("Failed", str(canonical_failed))
            canon_table.add_row(
                "Pass Rate",
                (
                    f"{canonical_passed / len(detailed_results) * 100:.1f}%"
                    if detailed_results
                    else "N/A"
                ),
            )

            self.console.print(canon_table)

            if canonical_failed > 0:
                self.console.print(
                    f"\n[yellow]⚠ {canonical_failed} canonical solutions failed - "
                    "check test setup or dataset issues[/yellow]"
                )

        # Print Pass@k metrics
        if results.metrics:
            self.console.print("\n[bold]Pass@k Metrics:[/bold]")
            metrics_table = Table(show_header=True, header_style="bold magenta")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Score", style="green", justify="right")

            for metric_name, value in sorted(results.metrics.items()):
                if isinstance(value, float):
                    metrics_table.add_row(metric_name, f"{value:.4f}")
                else:
                    metrics_table.add_row(metric_name, str(value))

            self.console.print(metrics_table)

    def verify_canonical_solutions(
        self, samples: list[dict[str, Any]], save_results: Path | None = None
    ) -> dict[str, Any]:
        """
        Verify that all canonical solutions pass their tests.

        This is useful for debugging and validating the evaluation setup.

        Args:
            samples: List of samples to verify
            save_results: Optional path to save verification results

        Returns:
            Dict with verification summary and failures
        """
        self.console.print("\n[bold cyan]Canonical Solution Verification[/bold cyan]")
        self.console.print(f"Dataset type: {self.dataset_type.value}")
        self.console.print(f"Samples: {len(samples)}\n")

        results = []
        failures = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Verifying...", total=len(samples))

            for sample in samples:
                task_id = sample.get("task_id", "unknown")
                test_code = sample.get("test", "")
                entry_point = sample.get("entry_point")

                combined_code = self.combine_canonical(sample)
                exec_result = self.execute_code(combined_code, test_code, entry_point)

                result = {
                    "task_id": task_id,
                    "passed": exec_result["success"],
                    "combined_code": combined_code,
                    "test_code": test_code,
                    "error": exec_result.get("error", ""),
                }
                results.append(result)

                if not exec_result["success"]:
                    failures.append(result)

                progress.update(task, advance=1)

        # Print summary
        passed = len(results) - len(failures)
        self.console.print("\n[bold]Results:[/bold]")
        self.console.print(f"  Passed: [green]{passed}[/green]")
        self.console.print(f"  Failed: [red]{len(failures)}[/red]")

        if failures:
            self.console.print("\n[bold red]Failed Tasks:[/bold red]")
            for f in failures[:5]:  # Show first 5 failures
                self.console.print(f"  • {f['task_id']}: {f['error'][:100]}...")

            if len(failures) > 5:
                self.console.print(f"  ... and {len(failures) - 5} more")

        # Save results
        if save_results:
            save_path = Path(save_results)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            output = {
                "metadata": {
                    "dataset": str(self.dataset_path),
                    "dataset_type": self.dataset_type.value,
                    "total_samples": len(samples),
                },
                "summary": {
                    "passed": passed,
                    "failed": len(failures),
                    "pass_rate": passed / len(samples) if samples else 0,
                },
                "failures": failures,
                "all_results": results,
            }

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)

            self.console.print(f"\n[green]Results saved to: {save_path}[/green]")

        return {
            "passed": passed,
            "failed": len(failures),
            "failures": failures,
            "results": results,
        }
