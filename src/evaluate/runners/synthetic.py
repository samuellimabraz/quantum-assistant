"""Synthetic dataset runner for multimodal evaluation with HuggingFace support."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
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

from evaluate.evaluators.base import AggregatedResults, EvaluationResult
from evaluate.execution.sandbox import CodeExecutor
from evaluate.metrics.code_metrics import PassAtK
from evaluate.metrics.text_metrics import BLEU, ROUGE
from evaluate.utils.results import ResultsManager
from models.client import LLMClient, VLMClient, Message


class QuestionType:
    """Valid question types in synthetic dataset."""

    FUNCTION_COMPLETION = "function_completion"
    CODE_GENERATION = "code_generation"
    QA = "qa"

    CODE_TYPES = {FUNCTION_COMPLETION, CODE_GENERATION}
    ALL_TYPES = {FUNCTION_COMPLETION, CODE_GENERATION, QA}


@dataclass
class SampleResult:
    """Detailed result for a single sample."""

    task_id: str
    question: str
    question_type: str
    category: str
    is_multimodal: bool
    generated_solutions: list[str]
    combined_codes: list[str]
    execution_results: list[dict[str, Any]]
    reference_answer: str
    metrics: dict[str, float]
    success: bool


@dataclass
class CanonicalVerification:
    """Result of canonical solution verification."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


@dataclass
class ModalityMetrics:
    """Metrics breakdown by modality (text-only vs multimodal)."""

    count: int = 0
    success_rate: float = 0.0
    pass_at_1: float | None = None
    rouge_l: float | None = None
    bleu: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {"count": self.count, "success_rate": self.success_rate}
        if self.pass_at_1 is not None:
            result["pass@1"] = self.pass_at_1
        if self.rouge_l is not None:
            result["rouge_l"] = self.rouge_l
        if self.bleu is not None:
            result["bleu"] = self.bleu
        return result


class SyntheticDatasetRunner:
    """
    Runner for evaluating models on synthetic multimodal dataset.

    Supports evaluation of three question types:
    - function_completion: Code stub completion (like Qiskit HumanEval normal)
    - code_generation: Full code generation from natural language (like Qiskit HumanEval hard)
    - qa: Text-based Q&A with similarity metrics

    Uses HuggingFace datasets format for loading, with proper test execution
    for code types and text similarity metrics for QA.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        model_client: LLMClient | VLMClient,
        images_dir: Path | str | None = None,
        k_values: list[int] | None = None,
        num_samples_per_task: int = 1,
        timeout: int = 30,
        max_concurrent: int = 10,
    ):
        """
        Initialize synthetic dataset runner.

        Args:
            dataset_path: Path to HuggingFace dataset directory or split file
            model_client: Model client for inference
            images_dir: Directory containing images (optional, can be in dataset)
            k_values: K values for Pass@k metrics
            num_samples_per_task: Number of solutions to generate per task
            timeout: Execution timeout for code samples
            max_concurrent: Maximum concurrent API requests
        """
        self.dataset_path = Path(dataset_path)
        self.model_client = model_client
        self.images_dir = Path(images_dir) if images_dir else None
        self.k_values = k_values or [1]
        self.num_samples = num_samples_per_task
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.executor = CodeExecutor(timeout=timeout)
        self.console = Console()

        # Text metrics for QA evaluation
        self.bleu = BLEU(n=4)
        self.rouge = ROUGE()

    def load_dataset(self, split: str = "test") -> list[dict[str, Any]]:
        """
        Load dataset from HuggingFace format.

        Args:
            split: Dataset split to load (train, validation, test)

        Returns:
            List of sample dictionaries
        """
        if self.dataset_path.is_file():
            return self._load_from_file()

        return self._load_from_huggingface(split)

    def _load_from_huggingface(self, split: str) -> list[dict[str, Any]]:
        """Load from HuggingFace dataset directory."""
        from datasets import load_from_disk

        dataset_dict = load_from_disk(str(self.dataset_path))

        if split not in dataset_dict:
            available = list(dataset_dict.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available}")

        dataset = dataset_dict[split]
        samples = []

        for idx, row in enumerate(dataset):
            sample = {
                "task_id": f"synthetic/{split}/{idx}",
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "category": row.get("category", ""),
                "question_type": row.get("type", "qa"),
                "test_code": row.get("test_code", ""),
                "entry_point": row.get("entry_point", ""),
                "image": row.get("image"),
                "source": row.get("source", ""),
            }
            samples.append(sample)

        return samples

    def _load_from_file(self) -> list[dict[str, Any]]:
        """Load from pickle or JSONL file."""
        import pickle

        if self.dataset_path.suffix == ".pkl":
            with open(self.dataset_path, "rb") as f:
                raw_samples = pickle.load(f)

            samples = []
            for idx, sample in enumerate(raw_samples):
                if hasattr(sample, "to_dict"):
                    data = sample.to_dict()
                elif hasattr(sample, "__dict__"):
                    data = sample.__dict__
                else:
                    data = dict(sample)

                # Normalize field names
                data["task_id"] = f"synthetic/{idx}"
                data["question_type"] = data.get("question_type", data.get("type", "qa"))
                data["test_code"] = data.get("test_code", "")
                data["image"] = None
                if data.get("image_path"):
                    data["image"] = data["image_path"]

                samples.append(data)

            return samples

        elif self.dataset_path.suffix == ".jsonl":
            samples = []
            with open(self.dataset_path, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    data = json.loads(line)
                    data["task_id"] = f"synthetic/{idx}"
                    data["question_type"] = data.get("question_type", data.get("type", "qa"))
                    samples.append(data)
            return samples

        raise ValueError(f"Unsupported file format: {self.dataset_path.suffix}")

    def extract_code_from_response(self, response: str, entry_point: str | None = None) -> str:
        """
        Extract code from model response.

        Handles markdown code blocks and prioritizes blocks with the entry point function.
        """
        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)

        if not code_blocks:
            return response.strip("\n")

        if len(code_blocks) == 1:
            return code_blocks[0].rstrip()

        if entry_point:
            for block in code_blocks:
                if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", block):
                    return block.rstrip()

        return max(code_blocks, key=len).rstrip()

    def _has_function_definition(self, code: str, entry_point: str) -> bool:
        """Check if code contains a function definition for entry_point."""
        pattern = rf"def\s+{re.escape(entry_point)}\s*\("
        return bool(re.search(pattern, code))

    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract code from markdown code block if present."""
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        return match.group(1).rstrip() if match else text.rstrip()

    def _assemble_function_completion(self, stub: str, body: str) -> str:
        """
        Assemble function completion by replacing pass with body.

        This matches the logic in synthetic_data/generators/sessions.py
        to ensure canonical verification uses the same assembly method.
        """
        # Extract code from markdown if present
        stub_code = self._extract_code_from_markdown(stub)
        body_code = self._extract_code_from_markdown(body)

        # Find the pass statement and its indentation
        stub_lines = stub_code.split("\n")
        result_lines = []
        replaced = False

        for line in stub_lines:
            stripped = line.strip()
            if stripped == "pass" and not replaced:
                # Get the indentation of the pass statement
                pass_indent = len(line) - len(line.lstrip())

                # Normalize body indentation and insert
                body_lines = body_code.split("\n")
                normalized = self._normalize_body_indentation(body_lines, pass_indent)

                for body_line in normalized:
                    result_lines.append(body_line)
                replaced = True
            else:
                result_lines.append(line)

        if not replaced:
            # No pass found - append body with default indent
            result_lines.append("")
            body_lines = body_code.split("\n")
            for body_line in body_lines:
                if body_line.strip():
                    result_lines.append("    " + body_line.lstrip())
                else:
                    result_lines.append("")

        return "\n".join(result_lines)

    def _normalize_body_indentation(self, body_lines: list[str], target_indent: int) -> list[str]:
        """
        Normalize body indentation to target level.

        Handles the common case where the body has inconsistent indentation:
        - First line has 0 indentation
        - Subsequent lines have 4 spaces (relative to what should be 0)

        This normalizes all lines to use target_indent as the base.
        """
        non_empty = [(i, line) for i, line in enumerate(body_lines) if line.strip()]
        if not non_empty:
            return body_lines

        # Get first non-empty line's indentation
        first_idx, first_line = non_empty[0]
        first_indent = len(first_line) - len(first_line.lstrip())

        # Check for the common pattern: first line at 0, rest at 4
        if first_indent == 0 and len(non_empty) > 1:
            subsequent_indents = [len(line) - len(line.lstrip()) for _, line in non_empty[1:]]
            min_subsequent = min(subsequent_indents) if subsequent_indents else 0

            # If subsequent lines have extra indentation, they should align with first line
            if min_subsequent > 0:
                # All lines should be at target_indent base
                result = []
                for i, line in enumerate(body_lines):
                    if not line.strip():
                        result.append("")
                    elif i == first_idx:
                        # First line gets target indent
                        result.append(" " * target_indent + line.lstrip())
                    else:
                        # Subsequent lines: remove extra base indent, add target
                        current_indent = len(line) - len(line.lstrip())
                        relative = current_indent - min_subsequent
                        new_indent = " " * (target_indent + relative)
                        result.append(new_indent + line.lstrip())
                return result

        # Standard case: subtract min indent and add target
        min_indent = min(len(line) - len(line.lstrip()) for _, line in non_empty)

        result = []
        for line in body_lines:
            if not line.strip():
                result.append("")
            else:
                current_indent = len(line) - len(line.lstrip())
                relative_indent = current_indent - min_indent
                new_indent = " " * (target_indent + relative_indent)
                result.append(new_indent + line.lstrip())

        return result

    def combine_code(self, sample: dict[str, Any], generated: str) -> str:
        """
        Combine prompt and generated code based on question type.

        For function_completion: Assembles stub + body by replacing pass statement
        For code_generation: Uses generated code directly
        """
        question_type = sample.get("question_type", "qa")
        question = sample.get("question", "")
        entry_point = sample.get("entry_point", "")

        # Extract code from markdown if response contains it
        generated = self.extract_code_from_response(generated, entry_point)

        if question_type == QuestionType.FUNCTION_COMPLETION:
            # Check if model generated full code instead of just body
            if entry_point and self._has_function_definition(generated, entry_point):
                return generated
            else:
                # Assemble stub + body by replacing pass statement
                return self._assemble_function_completion(question, generated)

        elif question_type == QuestionType.CODE_GENERATION:
            # Full code generation - use generated code directly
            return generated

        # QA type - return as-is
        return generated

    def create_messages(
        self,
        sample: dict[str, Any],
        system_prompt: str | None = None,
    ) -> list[Message]:
        """
        Create messages for model inference.

        Args:
            sample: Sample dictionary
            system_prompt: Optional system prompt

        Returns:
            List of Message objects
        """
        messages = []

        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        question = sample.get("question", "")
        messages.append(Message(role="user", content=question))

        return messages

    def _get_image_path(self, sample: dict[str, Any]) -> Path | None:
        """Resolve image path for a sample."""
        image_data = sample.get("image")

        if image_data is None:
            return None

        # If image is a PIL Image object, it's embedded in dataset
        if hasattr(image_data, "save"):
            # Save to temporary file for processing
            import tempfile

            temp_path = Path(tempfile.mktemp(suffix=".png"))
            image_data.save(temp_path)
            return temp_path

        # If string path
        if isinstance(image_data, str):
            if self.images_dir:
                return self.images_dir / image_data
            return Path(image_data)

        return None

    async def generate_solutions_async(
        self,
        samples: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> dict[str, list[str]]:
        """
        Generate solutions for all samples asynchronously.

        Args:
            samples: List of samples
            system_prompt: Optional system prompt

        Returns:
            Dictionary mapping task_id to list of generated solutions
        """
        is_vlm = isinstance(self.model_client, VLMClient)

        all_tasks = []
        for sample in samples:
            task_id = sample["task_id"]
            messages = self.create_messages(sample, system_prompt)
            image_path = self._get_image_path(sample)

            for _ in range(self.num_samples):
                all_tasks.append(
                    {
                        "task_id": task_id,
                        "messages": messages,
                        "image_path": image_path,
                        "is_multimodal": image_path is not None and is_vlm,
                    }
                )

        total_requests = len(all_tasks)
        solutions: dict[str, list[str]] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Generating solutions...", total=total_requests)
            completed = [0]
            lock = asyncio.Lock()

            async def generate_one(task_info: dict) -> tuple[str, str]:
                """Generate single solution."""
                task_id = task_info["task_id"]
                messages = task_info["messages"]
                image_path = task_info["image_path"]
                is_multimodal = task_info["is_multimodal"]

                try:
                    if is_multimodal and image_path:
                        prompt = messages[-1].content
                        sys_prompt = messages[0].content if len(messages) > 1 else None
                        result = await self.model_client.generate_with_image_async(
                            text=prompt,
                            image_path=image_path,
                            system_prompt=sys_prompt,
                        )
                    else:
                        result = await self.model_client.generate_async(messages)
                except Exception as e:
                    self.console.print(f"[red]Error generating for {task_id}: {e}[/red]")
                    result = ""

                async with lock:
                    completed[0] += 1
                    progress.update(task, completed=completed[0])

                return task_id, result

            semaphore = asyncio.Semaphore(self.max_concurrent)

            async def bounded_generate(task_info: dict) -> tuple[str, str]:
                async with semaphore:
                    return await generate_one(task_info)

            results = await asyncio.gather(*[bounded_generate(t) for t in all_tasks])

            for task_id, result in results:
                if task_id not in solutions:
                    solutions[task_id] = []
                solutions[task_id].append(result)

        return solutions

    def generate_solutions(
        self,
        samples: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> dict[str, list[str]]:
        """Generate solutions (sync wrapper)."""
        return asyncio.run(self.generate_solutions_async(samples, system_prompt))

    def execute_code(self, code: str, test_code: str, entry_point: str | None) -> dict[str, Any]:
        """Execute code with test and return detailed result.

        Uses the same execution model as the generation phase for consistency.
        """
        result = self.executor.execute(code, test_code, entry_point)
        return {
            "success": result.success,
            "error": result.error if not result.success else "",
            "output": result.output,
            "timeout": result.timeout,
        }

    def verify_canonical_solutions(
        self, samples: list[dict[str, Any]], save_results: Path | None = None
    ) -> CanonicalVerification:
        """
        Verify that canonical (reference) solutions pass their tests.

        This validates the test/execution system by running reference answers
        against their unit tests.

        Args:
            samples: List of samples to verify
            save_results: Optional path to save verification results

        Returns:
            CanonicalVerification with pass/fail statistics
        """
        self.console.print("\n[bold cyan]Canonical Solution Verification[/bold cyan]")
        self.console.print(f"Samples: {len(samples)}\n")

        verification = CanonicalVerification(total=len(samples))
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Verifying canonical solutions...", total=len(samples))

            for sample in samples:
                task_id = sample.get("task_id", "unknown")
                question_type = sample.get("question_type", "qa")
                answer = sample.get("answer", "")
                test_code = sample.get("test_code", "")
                entry_point = sample.get("entry_point", "")

                result = {
                    "task_id": task_id,
                    "question_type": question_type,
                    "passed": True,
                    "error": "",
                    "skipped": False,
                }

                # Only verify code types with tests
                if question_type not in QuestionType.CODE_TYPES or not test_code:
                    result["skipped"] = True
                    result["skip_reason"] = "No test code" if not test_code else "QA type"
                    verification.passed += 1
                else:
                    # Combine answer with test code
                    combined = self.combine_code(sample, answer)
                    exec_result = self.execute_code(combined, test_code, entry_point)

                    result["passed"] = exec_result["success"]
                    result["error"] = exec_result.get("error", "")

                    if exec_result["success"]:
                        verification.passed += 1
                    else:
                        verification.failed += 1
                        verification.failures.append(
                            {
                                "task_id": task_id,
                                "question_type": question_type,
                                "error": exec_result.get("error", "Unknown error"),
                            }
                        )

                results.append(result)
                progress.update(task, advance=1)

        # Print summary
        self._print_canonical_verification_summary(verification)

        # Save results if requested
        if save_results:
            self._save_canonical_verification(save_results, verification, results)

        return verification

    def _print_canonical_verification_summary(self, verification: CanonicalVerification) -> None:
        """Print canonical verification summary."""
        from rich.table import Table

        self.console.print("\n[bold]Canonical Verification Results:[/bold]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total", str(verification.total))
        table.add_row("Passed", str(verification.passed))
        table.add_row("Failed", str(verification.failed))
        table.add_row("Pass Rate", f"{verification.pass_rate:.1%}")

        self.console.print(table)

        if verification.failures:
            self.console.print(
                f"\n[yellow]⚠ {verification.failed} canonical solutions failed[/yellow]"
            )
            for failure in verification.failures[:5]:
                self.console.print(f"  • {failure['task_id']}: {failure['error'][:100]}...")
            if len(verification.failures) > 5:
                self.console.print(f"  ... and {len(verification.failures) - 5} more")

    def _save_canonical_verification(
        self, save_path: Path, verification: CanonicalVerification, results: list[dict]
    ) -> None:
        """Save canonical verification results."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "summary": {
                "total": verification.total,
                "passed": verification.passed,
                "failed": verification.failed,
                "pass_rate": verification.pass_rate,
            },
            "failures": verification.failures,
            "results": results,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self.console.print(f"\n[green]Verification results saved to: {save_path}[/green]")

    def evaluate_sample(
        self,
        sample: dict[str, Any],
        solutions: list[str],
    ) -> SampleResult:
        """
        Evaluate a single sample with detailed results.

        Args:
            sample: Sample dictionary
            solutions: List of generated solutions

        Returns:
            SampleResult with detailed evaluation info
        """
        task_id = sample.get("task_id", "unknown")
        question_type = sample.get("question_type", "qa")
        category = sample.get("category", "")
        reference_answer = sample.get("answer", "")
        test_code = sample.get("test_code", "")
        entry_point = sample.get("entry_point", "")
        is_multimodal = sample.get("image") is not None

        combined_codes = []
        execution_results = []
        metrics = {}

        if question_type in QuestionType.CODE_TYPES:
            # Evaluate code types with test execution
            for solution in solutions:
                combined = self.combine_code(sample, solution)
                combined_codes.append(combined)

                if test_code:
                    exec_result = self.execute_code(combined, test_code, entry_point)
                else:
                    # No test - check syntax only
                    try:
                        compile(combined, "<string>", "exec")
                        exec_result = {"success": True, "error": "", "output": "", "timeout": False}
                    except SyntaxError as e:
                        exec_result = {
                            "success": False,
                            "error": str(e),
                            "output": "",
                            "timeout": False,
                        }

                execution_results.append(exec_result)

            passed_count = sum(1 for r in execution_results if r["success"])
            success = passed_count > 0
            metrics["pass_rate"] = passed_count / len(solutions) if solutions else 0.0

            # Compute Pass@k metrics
            for k in self.k_values:
                if len(solutions) >= k:
                    n = len(solutions)
                    c = passed_count
                    metrics[f"pass@{k}"] = PassAtK.compute_pass_at_k(n, c, k)

        else:
            # QA type - use text similarity metrics
            combined_codes = solutions
            prediction = solutions[0] if solutions else ""

            try:
                metrics["bleu"] = self.bleu.compute([prediction], [reference_answer])
                metrics["rouge_l"] = self.rouge.compute([prediction], [reference_answer])
            except Exception as e:
                metrics["error"] = str(e)

            # Success based on ROUGE-L threshold
            success = metrics.get("rouge_l", 0.0) > 0.3

            execution_results = [{"success": success, "error": "", "output": "", "timeout": False}]

        return SampleResult(
            task_id=task_id,
            question=sample.get("question", ""),
            question_type=question_type,
            category=category,
            is_multimodal=is_multimodal,
            generated_solutions=solutions,
            combined_codes=combined_codes,
            execution_results=execution_results,
            reference_answer=reference_answer,
            metrics=metrics,
            success=success,
        )

    def _compute_modality_metrics(
        self, results: list[SampleResult], is_multimodal: bool
    ) -> ModalityMetrics:
        """Compute metrics for a specific modality (text-only or multimodal)."""
        filtered = [r for r in results if r.is_multimodal == is_multimodal]

        if not filtered:
            return ModalityMetrics()

        success_count = sum(1 for r in filtered if r.success)
        metrics = ModalityMetrics(
            count=len(filtered),
            success_rate=success_count / len(filtered),
        )

        # Compute code metrics (Pass@1)
        code_results = [r for r in filtered if r.question_type in QuestionType.CODE_TYPES]
        if code_results:
            pass_values = [r.metrics.get("pass@1") for r in code_results if "pass@1" in r.metrics]
            if pass_values:
                metrics.pass_at_1 = sum(pass_values) / len(pass_values)

        # Compute text metrics (ROUGE-L, BLEU)
        qa_results = [r for r in filtered if r.question_type == QuestionType.QA]
        if qa_results:
            rouge_values = [r.metrics.get("rouge_l") for r in qa_results if "rouge_l" in r.metrics]
            bleu_values = [r.metrics.get("bleu") for r in qa_results if "bleu" in r.metrics]
            if rouge_values:
                metrics.rouge_l = sum(rouge_values) / len(rouge_values)
            if bleu_values:
                metrics.bleu = sum(bleu_values) / len(bleu_values)

        return metrics

    def aggregate_results(self, results: list[SampleResult]) -> AggregatedResults:
        """
        Aggregate evaluation results with detailed breakdowns.

        Args:
            results: List of sample results

        Returns:
            AggregatedResults with metrics grouped by type, category, and modality
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
        by_type: dict[str, list[SampleResult]] = {}
        for result in results:
            qtype = result.question_type
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(result)

        # Group by category
        by_category: dict[str, list[SampleResult]] = {}
        for result in results:
            cat = result.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result)

        # Compute aggregated metrics
        aggregated_metrics: dict[str, Any] = {}

        # Overall metrics
        overall = {"count": len(results), "success_rate": successful / len(results)}

        # Aggregate code metrics
        code_results = [r for r in results if r.question_type in QuestionType.CODE_TYPES]
        if code_results:
            for k in self.k_values:
                key = f"pass@{k}"
                values = [r.metrics.get(key) for r in code_results if key in r.metrics]
                if values:
                    overall[key] = sum(values) / len(values)

        # Aggregate text metrics
        qa_results = [r for r in results if r.question_type == QuestionType.QA]
        if qa_results:
            for metric_name in ["bleu", "rouge_l"]:
                values = [
                    r.metrics.get(metric_name) for r in qa_results if metric_name in r.metrics
                ]
                if values:
                    overall[metric_name] = sum(values) / len(values)

        aggregated_metrics["overall"] = overall

        # Per-type metrics
        for qtype, type_results in by_type.items():
            type_metrics = {
                "count": len(type_results),
                "success_rate": sum(1 for r in type_results if r.success) / len(type_results),
            }

            # Add type-specific metrics
            if qtype in QuestionType.CODE_TYPES:
                for k in self.k_values:
                    key = f"pass@{k}"
                    values = [r.metrics.get(key) for r in type_results if key in r.metrics]
                    if values:
                        type_metrics[key] = sum(values) / len(values)
            else:
                for metric_name in ["bleu", "rouge_l"]:
                    values = [
                        r.metrics.get(metric_name) for r in type_results if metric_name in r.metrics
                    ]
                    if values:
                        type_metrics[metric_name] = sum(values) / len(values)

            aggregated_metrics[f"by_type.{qtype}"] = type_metrics

        # Per-category metrics
        for cat, cat_results in by_category.items():
            cat_metrics = {
                "count": len(cat_results),
                "success_rate": sum(1 for r in cat_results if r.success) / len(cat_results),
            }
            aggregated_metrics[f"by_category.{cat}"] = cat_metrics

        # Modality breakdown with detailed metrics
        text_only_metrics = self._compute_modality_metrics(results, is_multimodal=False)
        multimodal_metrics = self._compute_modality_metrics(results, is_multimodal=True)

        if text_only_metrics.count > 0:
            aggregated_metrics["text_only"] = text_only_metrics.to_dict()

        if multimodal_metrics.count > 0:
            aggregated_metrics["multimodal"] = multimodal_metrics.to_dict()

        # Convert SampleResult to EvaluationResult for compatibility
        per_sample_results = [
            EvaluationResult(
                task_id=r.task_id,
                success=r.success,
                predictions=r.generated_solutions,
                ground_truth=r.reference_answer,
                metrics=r.metrics,
                metadata={
                    "question_type": r.question_type,
                    "category": r.category,
                    "is_multimodal": r.is_multimodal,
                    "execution_results": [e["success"] for e in r.execution_results],
                },
            )
            for r in results
        ]

        return AggregatedResults(
            total_samples=len(results),
            successful=successful,
            failed=failed,
            metrics=aggregated_metrics,
            per_sample_results=per_sample_results,
            metadata={
                "question_types": list(by_type.keys()),
                "categories": list(by_category.keys()),
            },
        )

    def evaluate(
        self,
        samples: list[dict[str, Any]] | None = None,
        split: str = "test",
        system_prompt: str | None = None,
        save_results: Path | None = None,
        verify_canonical: bool = False,
        model_name: str | None = None,
        run_timestamp: datetime | None = None,
    ) -> AggregatedResults:
        """
        Run full evaluation pipeline.

        Args:
            samples: Optional list of samples (loads from dataset if None)
            split: Dataset split to load if samples is None
            system_prompt: Optional system prompt
            save_results: Optional path to save detailed results
            verify_canonical: Whether to verify canonical solutions first
            model_name: Model name for metadata
            run_timestamp: Timestamp for this run

        Returns:
            AggregatedResults with evaluation metrics
        """
        if run_timestamp is None:
            run_timestamp = datetime.now()

        if samples is None:
            samples = self.load_dataset(split)

        # Count statistics
        type_counts: dict[str, int] = {}
        multimodal_count = 0
        for s in samples:
            qtype = s.get("question_type", "qa")
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
            if s.get("image") is not None:
                multimodal_count += 1

        self.console.print("\n[bold cyan]Synthetic Dataset Evaluation[/bold cyan]")
        self.console.print(f"Dataset: {self.dataset_path}")
        self.console.print(f"Samples: {len(samples)}")
        self.console.print(f"Solutions per task: {self.num_samples}")
        self.console.print(f"Pass@k values: {self.k_values}")
        self.console.print(f"\nBy type: {type_counts}")
        self.console.print(
            f"Multimodal: {multimodal_count}, Text-only: {len(samples) - multimodal_count}"
        )
        if system_prompt:
            preview = system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt
            self.console.print(f"System prompt: {preview}")
        self.console.print()

        # Verify canonical solutions if requested
        canonical_verification = None
        if verify_canonical:
            canonical_verification = self.verify_canonical_solutions(samples)

        # Generate solutions
        solutions_dict = self.generate_solutions(samples, system_prompt)

        # Evaluate each sample
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
                result = self.evaluate_sample(sample, solutions)
                detailed_results.append(result)
                progress.update(eval_task, advance=1)

        # Aggregate results
        aggregated = self.aggregate_results(detailed_results)

        # Save detailed results
        if save_results:
            self._save_results(
                save_results,
                samples,
                detailed_results,
                aggregated,
                canonical_verification,
                system_prompt,
                model_name,
                run_timestamp,
            )

        # Print summary
        self._print_summary(aggregated, canonical_verification)

        return aggregated

    def _save_results(
        self,
        save_path: Path,
        samples: list[dict[str, Any]],
        detailed_results: list[SampleResult],
        aggregated: AggregatedResults,
        canonical_verification: CanonicalVerification | None,
        system_prompt: str | None,
        model_name: str | None,
        run_timestamp: datetime,
    ) -> None:
        """Save detailed results to JSON."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = ResultsManager.build_metadata(
            dataset_path=self.dataset_path,
            dataset_type="synthetic",
            dataset_variant=None,
            model_name=model_name or "unknown",
            num_samples=len(samples),
            num_samples_per_task=self.num_samples,
            k_values=self.k_values,
            timeout=self.timeout,
            system_prompt=system_prompt,
            verify_canonical=canonical_verification is not None,
            timestamp=run_timestamp,
        )

        # Add synthetic-specific metadata
        type_counts: dict[str, int] = {}
        multimodal_count = 0
        for s in samples:
            qtype = s.get("question_type", "qa")
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
            if s.get("image") is not None:
                multimodal_count += 1

        metadata["dataset"]["type_distribution"] = type_counts
        metadata["dataset"]["multimodal_count"] = multimodal_count
        metadata["dataset"]["text_only_count"] = len(samples) - multimodal_count

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

        # Add canonical verification results if available
        if canonical_verification is not None:
            output_data["canonical_verification"] = {
                "total": canonical_verification.total,
                "passed": canonical_verification.passed,
                "failed": canonical_verification.failed,
                "pass_rate": canonical_verification.pass_rate,
                "failures": canonical_verification.failures,
            }

        for dr in detailed_results:
            sample_result = {
                "task_id": dr.task_id,
                "success": dr.success,
                "question_type": dr.question_type,
                "category": dr.category,
                "is_multimodal": dr.is_multimodal,
                "metrics": dr.metrics,
                "question": dr.question,
                "reference_answer": dr.reference_answer,
                "generated_solutions": dr.generated_solutions,
                "combined_codes": (
                    dr.combined_codes if dr.question_type in QuestionType.CODE_TYPES else None
                ),
                "execution_results": dr.execution_results,
            }
            output_data["results"].append(sample_result)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.console.print(f"\n[green]Results saved to: {save_path}[/green]")

    def _print_summary(
        self, results: AggregatedResults, canonical_verification: CanonicalVerification | None
    ) -> None:
        """Print evaluation summary."""
        from rich.table import Table

        self.console.print("\n[bold cyan]Evaluation Results[/bold cyan]\n")

        # Overall stats
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Samples", str(results.total_samples))
        table.add_row("Successful", str(results.successful))
        table.add_row("Failed", str(results.failed))
        table.add_row("Success Rate", f"{results.success_rate:.1%}")

        self.console.print(table)

        # Overall metrics
        if "overall" in results.metrics:
            self.console.print("\n[bold]Overall Metrics:[/bold]")
            metrics_table = Table(show_header=True, header_style="bold magenta")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Score", style="green", justify="right")

            for metric_name, value in sorted(results.metrics["overall"].items()):
                if metric_name not in ["count", "success_rate"] and isinstance(value, float):
                    metrics_table.add_row(metric_name, f"{value:.4f}")

            self.console.print(metrics_table)

        # By-type metrics
        type_metrics = {k: v for k, v in results.metrics.items() if k.startswith("by_type.")}
        if type_metrics:
            self.console.print("\n[bold]Metrics by Question Type:[/bold]")
            type_table = Table(show_header=True, header_style="bold magenta")
            type_table.add_column("Type", style="cyan")
            type_table.add_column("Count", justify="right")
            type_table.add_column("Success Rate", justify="right")
            type_table.add_column("Key Metrics", justify="right")

            for key, metrics in sorted(type_metrics.items()):
                qtype = key.replace("by_type.", "")
                count = metrics.get("count", 0)
                success_rate = metrics.get("success_rate", 0.0)

                key_metric = ""
                if "pass@1" in metrics:
                    key_metric = f"pass@1: {metrics['pass@1']:.3f}"
                elif "rouge_l" in metrics:
                    key_metric = f"ROUGE-L: {metrics['rouge_l']:.3f}"

                type_table.add_row(qtype, str(count), f"{success_rate:.1%}", key_metric)

            self.console.print(type_table)

        # Modality breakdown with detailed metrics
        if "text_only" in results.metrics or "multimodal" in results.metrics:
            self.console.print("\n[bold]Metrics by Modality:[/bold]")
            mod_table = Table(show_header=True, header_style="bold magenta")
            mod_table.add_column("Modality", style="cyan")
            mod_table.add_column("Count", justify="right")
            mod_table.add_column("Success Rate", justify="right")
            mod_table.add_column("Pass@1", justify="right")
            mod_table.add_column("ROUGE-L", justify="right")

            for modality in ["text_only", "multimodal"]:
                if modality in results.metrics:
                    m = results.metrics[modality]
                    pass_at_1 = f"{m['pass@1']:.3f}" if m.get("pass@1") is not None else "-"
                    rouge_l = f"{m['rouge_l']:.3f}" if m.get("rouge_l") is not None else "-"
                    mod_table.add_row(
                        modality.replace("_", " ").title(),
                        str(m["count"]),
                        f"{m['success_rate']:.1%}",
                        pass_at_1,
                        rouge_l,
                    )

            self.console.print(mod_table)

        # Canonical verification summary
        if canonical_verification is not None:
            self.console.print("\n[bold]Canonical Verification:[/bold]")
            if canonical_verification.failed == 0:
                self.console.print(
                    f"[green]✓ All {canonical_verification.passed} canonical solutions passed[/green]"
                )
            else:
                self.console.print(
                    f"[yellow]⚠ {canonical_verification.failed}/{canonical_verification.total} "
                    f"canonical solutions failed[/yellow]"
                )
