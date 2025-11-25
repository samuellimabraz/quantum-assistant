"""Synthetic dataset runner for multimodal evaluation."""

import asyncio
import json
import pickle
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

from evaluate.evaluators.base import AggregatedResults
from evaluate.evaluators.multimodal import MultimodalEvaluator
from models.client import LLMClient, VLMClient


class SyntheticDatasetRunner:
    """Runner for evaluating on synthetic multimodal dataset."""

    def __init__(
        self,
        test_split_path: Path | str,
        model_client: LLMClient | VLMClient,
        images_dir: Path | str | None = None,
        timeout: int = 30,
        max_concurrent: int = 10,
    ):
        """
        Initialize synthetic dataset runner.

        Args:
            test_split_path: Path to test split (pkl or jsonl)
            model_client: Model client for inference
            images_dir: Directory containing images for multimodal samples
            timeout: Execution timeout for code samples
            max_concurrent: Maximum concurrent API requests
        """
        self.test_split_path = Path(test_split_path)
        self.model_client = model_client
        self.images_dir = Path(images_dir) if images_dir else None
        self.max_concurrent = max_concurrent
        self.evaluator = MultimodalEvaluator(timeout=timeout)
        self.console = Console()

    def load_test_split(self) -> list[dict[str, Any]]:
        """Load test split from file."""
        if self.test_split_path.suffix == ".pkl":
            return self._load_pickle()
        elif self.test_split_path.suffix == ".jsonl":
            return self._load_jsonl()
        else:
            raise ValueError(f"Unsupported file format: {self.test_split_path.suffix}")

    def _load_pickle(self) -> list[dict[str, Any]]:
        """Load samples from pickle file."""
        with open(self.test_split_path, "rb") as f:
            samples = pickle.load(f)

        # Convert sample objects to dicts if needed
        if samples and not isinstance(samples[0], dict):
            samples = [self._sample_to_dict(s) for s in samples]

        return samples

    def _load_jsonl(self) -> list[dict[str, Any]]:
        """Load samples from JSONL file."""
        samples = []
        with open(self.test_split_path) as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    @staticmethod
    def _sample_to_dict(sample: Any) -> dict[str, Any]:
        """Convert sample object to dictionary."""
        if hasattr(sample, "__dict__"):
            return {k: v for k, v in sample.__dict__.items()}
        return dict(sample)

    def create_prompt(self, sample: dict[str, Any]) -> str:
        """
        Create prompt from sample.

        Args:
            sample: Sample dictionary

        Returns:
            Formatted prompt string
        """
        question = sample.get("question", "")
        question_type = sample.get("question_type", "qa")

        # Add context based on question type
        if question_type == "code":
            prefix = "Generate Qiskit code to solve the following:\n\n"
        elif question_type == "function_completion":
            prefix = "Complete the following Qiskit function:\n\n"
        elif question_type == "caption":
            prefix = "Provide a caption for the quantum circuit:\n\n"
        elif question_type == "summary":
            prefix = "Summarize the following:\n\n"
        else:  # qa
            prefix = ""

        return prefix + question

    async def generate_predictions_async(
        self, samples: list[dict[str, Any]], num_predictions: int = 1
    ) -> dict[str, list[str]]:
        """
        Generate predictions for all samples asynchronously.

        Args:
            samples: List of test samples
            num_predictions: Number of predictions per sample

        Returns:
            Dictionary mapping sample ID to list of predictions
        """
        from models.client import Message

        predictions = {}
        is_vlm = isinstance(self.model_client, VLMClient)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Generating predictions...", total=len(samples) * num_predictions
            )

            completed = 0

            def update_progress(count):
                nonlocal completed
                completed += count
                progress.update(task, completed=completed)

            for sample in samples:
                sample_id = sample.get("id", sample.get("task_id", str(hash(frozenset(sample.items())))))
                prompt = self.create_prompt(sample)
                image_path = sample.get("image_path")

                # Check if sample is multimodal
                if is_vlm and image_path and self.images_dir:
                    full_image_path = self.images_dir / image_path
                    if full_image_path.exists():
                        # Generate with image
                        prompts_batch = [(prompt, full_image_path)] * num_predictions
                        generated = await self.model_client.generate_batch_with_images_async(
                            prompts_batch,
                            max_concurrent=self.max_concurrent,
                            progress_callback=update_progress,
                        )
                    else:
                        # Fallback to text-only
                        batch = [[Message(role="user", content=prompt)]] * num_predictions
                        generated = await self.model_client.generate_batch_async(
                            batch,
                            max_concurrent=self.max_concurrent,
                            progress_callback=update_progress,
                        )
                else:
                    # Text-only generation
                    batch = [[Message(role="user", content=prompt)]] * num_predictions
                    generated = await self.model_client.generate_batch_async(
                        batch,
                        max_concurrent=self.max_concurrent,
                        progress_callback=update_progress,
                    )

                predictions[sample_id] = generated

        return predictions

    def generate_predictions(
        self, samples: list[dict[str, Any]], num_predictions: int = 1
    ) -> dict[str, list[str]]:
        """Generate predictions (sync wrapper)."""
        return asyncio.run(self.generate_predictions_async(samples, num_predictions))

    def evaluate(
        self,
        samples: list[dict[str, Any]] | None = None,
        num_predictions: int = 1,
        save_results: Path | None = None,
    ) -> AggregatedResults:
        """
        Run full evaluation pipeline.

        Args:
            samples: Optional list of samples (loads from test_split_path if None)
            num_predictions: Number of predictions per sample
            save_results: Optional path to save detailed results

        Returns:
            AggregatedResults with evaluation metrics
        """
        if samples is None:
            samples = self.load_test_split()

        self.console.print(f"\n[bold cyan]Synthetic Dataset Evaluation[/bold cyan]")
        self.console.print(f"Samples: {len(samples)}")
        self.console.print(f"Predictions per sample: {num_predictions}")

        # Count multimodal samples
        multimodal_count = sum(1 for s in samples if s.get("image_path"))
        self.console.print(f"Multimodal samples: {multimodal_count}")
        self.console.print(f"Text-only samples: {len(samples) - multimodal_count}\n")

        # Generate predictions
        predictions_dict = self.generate_predictions(samples, num_predictions)

        # Evaluate each sample
        self.console.print("\n[bold]Evaluating predictions...[/bold]")
        results = []

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
                sample_id = sample.get("id", sample.get("task_id", str(hash(frozenset(sample.items())))))
                preds = predictions_dict.get(sample_id, [])

                # Add task_id for evaluation
                sample["task_id"] = sample_id

                result = self.evaluator.evaluate_sample(sample, preds)
                results.append(result)

                progress.update(eval_task, advance=1)

        # Aggregate results
        aggregated = self.evaluator.aggregate_results(results)

        # Save detailed results if requested
        if save_results:
            save_results = Path(save_results)
            save_results.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "metadata": {
                    "test_split": str(self.test_split_path),
                    "num_samples": len(samples),
                    "predictions_per_sample": num_predictions,
                    "multimodal_count": multimodal_count,
                },
                "metrics": aggregated.metrics,
                "results": [
                    {
                        "task_id": r.task_id,
                        "success": r.success,
                        "metrics": r.metrics,
                        "metadata": r.metadata,
                    }
                    for r in results
                ],
            }

            with open(save_results, "w") as f:
                json.dump(output_data, f, indent=2)

            self.console.print(f"\n[green]Results saved to: {save_results}[/green]")

        # Print summary
        self._print_summary(aggregated)

        return aggregated

    def _print_summary(self, results: AggregatedResults) -> None:
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
                if isinstance(value, float):
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

                # Get key metric for this type
                key_metric = ""
                if "pass@1" in metrics:
                    key_metric = f"pass@1: {metrics['pass@1']:.3f}"
                elif "bleu" in metrics:
                    key_metric = f"BLEU: {metrics['bleu']:.3f}"
                elif "rouge_l" in metrics:
                    key_metric = f"ROUGE-L: {metrics['rouge_l']:.3f}"

                type_table.add_row(qtype, str(count), f"{success_rate:.1%}", key_metric)

            self.console.print(type_table)

