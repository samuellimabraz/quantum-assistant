"""CLI for evaluation module."""

import os
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

from evaluate.config.schema import EvaluationConfig
from evaluate.config.system_prompts import get_system_prompt
from evaluate.runners.qiskit_humaneval import QiskitHumanEvalRunner
from evaluate.runners.synthetic import SyntheticDatasetRunner
from evaluate.utils.results import ResultsManager
from models.client import LLMClient, VLMClient

app = typer.Typer(
    name="evaluate",
    help="Evaluation tools for quantum computing models",
    add_completion=False,
)

console = Console()


def _create_model_client(config: EvaluationConfig) -> LLMClient | VLMClient:
    """Create model client from configuration."""
    if config.model.is_vlm:
        return VLMClient(
            base_url=config.model.base_url,
            api_key=config.model.api_key or os.getenv("API_KEY", ""),
            model_name=config.model.model_name,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
            timeout=config.model.timeout,
        )
    else:
        return LLMClient(
            base_url=config.model.base_url,
            api_key=config.model.api_key or os.getenv("API_KEY", ""),
            model_name=config.model.model_name,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
            timeout=config.model.timeout,
        )


def _get_system_prompt(config: EvaluationConfig) -> str | None:
    """Get system prompt based on configuration."""
    if config.metrics.system_prompt_type == "custom":
        return config.metrics.custom_system_prompt
    elif config.metrics.system_prompt_type:
        return get_system_prompt(config.metrics.system_prompt_type)
    return None


def _get_results_path(config: EvaluationConfig, run_timestamp: datetime) -> Path:
    """Determine the results file path based on configuration."""
    if config.output.results_file and not config.output.auto_filename:
        return config.output.results_file

    return ResultsManager.get_result_path(
        dataset_type=config.dataset.type,
        model_name=config.model.model_name,
        num_samples_per_task=config.metrics.num_samples_per_task,
        k_values=config.metrics.k_values,
        dataset_variant=config.dataset.dataset_variant,
        timestamp=run_timestamp,
    )


def _run_qiskit_humaneval(config: EvaluationConfig, run_timestamp: datetime) -> None:
    """Run Qiskit HumanEval evaluation."""
    results_path = _get_results_path(config, run_timestamp)
    ResultsManager.ensure_output_dir(results_path)
    console.print(f"[dim]Results will be saved to: {results_path}[/dim]\n")

    client = _create_model_client(config)
    system_prompt = _get_system_prompt(config)

    runner = QiskitHumanEvalRunner(
        dataset_path=config.dataset.path,
        model_client=client,
        k_values=config.metrics.k_values,
        num_samples_per_task=config.metrics.num_samples_per_task,
        timeout=config.metrics.execution_timeout,
        max_concurrent=config.metrics.max_concurrent,
        dataset_type=config.dataset.dataset_variant,
    )

    samples = runner.load_dataset()

    if config.dataset.max_samples:
        samples = samples[: config.dataset.max_samples]
        console.print(f"[yellow]Limiting to {config.dataset.max_samples} samples[/yellow]")

    results = runner.evaluate(
        samples=samples,
        system_prompt=system_prompt,
        save_results=results_path,
        verify_canonical=config.metrics.verify_canonical,
        model_name=config.model.model_name,
        run_timestamp=run_timestamp,
    )

    console.print("\n[green]✓ Evaluation complete![/green]")
    console.print(f"Success rate: {results.success_rate:.1%}")


def _run_synthetic(config: EvaluationConfig, run_timestamp: datetime) -> None:
    """Run synthetic dataset evaluation."""
    results_path = _get_results_path(config, run_timestamp)
    ResultsManager.ensure_output_dir(results_path)
    console.print(f"[dim]Results will be saved to: {results_path}[/dim]\n")

    client = _create_model_client(config)
    system_prompt = _get_system_prompt(config)

    runner = SyntheticDatasetRunner(
        dataset_path=config.dataset.path,
        model_client=client,
        images_dir=config.dataset.images_dir,
        k_values=config.metrics.k_values,
        num_samples_per_task=config.metrics.num_samples_per_task,
        timeout=config.metrics.execution_timeout,
        max_concurrent=config.metrics.max_concurrent,
    )

    samples = runner.load_dataset(split=config.dataset.split)

    # Filter to text-only samples if requested
    if config.dataset.text_only:
        original_count = len(samples)
        samples = [s for s in samples if s.get("image") is None]
        console.print(
            f"[yellow]Filtering to text-only: {len(samples)}/{original_count} samples[/yellow]"
        )

    if config.dataset.max_samples:
        samples = samples[: config.dataset.max_samples]
        console.print(f"[yellow]Limiting to {config.dataset.max_samples} samples[/yellow]")

    results = runner.evaluate(
        samples=samples,
        split=config.dataset.split,
        system_prompt=system_prompt,
        save_results=results_path,
        verify_canonical=config.metrics.verify_canonical,
        model_name=config.model.model_name,
        run_timestamp=run_timestamp,
    )

    console.print("\n[green]✓ Evaluation complete![/green]")
    console.print(f"Success rate: {results.success_rate:.1%}")


@app.command()
def run(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to YAML configuration file"),
):
    """Run evaluation from YAML configuration file."""
    try:
        config = EvaluationConfig.from_yaml(config_path)
        run_timestamp = datetime.now()

        console.print(f"\n[bold cyan]Evaluation: {config.dataset.type}[/bold cyan]\n")

        if config.dataset.type == "qiskit_humaneval":
            _run_qiskit_humaneval(config, run_timestamp)
        elif config.dataset.type == "synthetic":
            _run_synthetic(config, run_timestamp)
        else:
            raise ValueError(f"Unknown dataset type: {config.dataset.type}")

    except Exception as e:
        console.print(f"\n[red]✗ Evaluation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def qiskit_humaneval(
    dataset_path: Path = typer.Option(..., "--dataset", "-d", help="Path to Qiskit HumanEval JSON"),
    model_url: str = typer.Option(..., "--model-url", help="Model API base URL"),
    model_name: str = typer.Option("qwen", "--model-name", help="Model name"),
    api_key: str = typer.Option("", "--api-key", help="API key (optional)"),
    system_prompt: str = typer.Option(
        "",
        "--system-prompt",
        "-s",
        help="System prompt: 'qiskit' for IBM prompt, 'minimal', or custom text",
    ),
    num_samples: int = typer.Option(
        1, "--num-samples", "-n", help="Number of solutions per task", min=1
    ),
    k_values: str = typer.Option("1", "--k-values", help="Comma-separated k values for Pass@k"),
    timeout: int = typer.Option(30, "--timeout", help="Execution timeout in seconds"),
    output: Path = typer.Option(
        None, "--output", "-o", help="Path to save results (auto-generated if not set)"
    ),
    max_samples: int = typer.Option(
        None, "--max-samples", help="Limit number of samples to evaluate"
    ),
    dataset_type: str = typer.Option(
        None, "--dataset-type", help="Dataset type: 'normal' or 'hard' (auto-detected if not set)"
    ),
    do_verify_canonical: bool = typer.Option(
        False, "--verify-canonical", help="Also verify canonical solutions pass tests"
    ),
):
    """Evaluate model on Qiskit HumanEval benchmark."""
    console.print("\n[bold cyan]Qiskit HumanEval Evaluation[/bold cyan]\n")

    run_timestamp = datetime.now()
    k_list = [int(k.strip()) for k in k_values.split(",")]

    # Resolve system prompt
    resolved_prompt = None
    if system_prompt:
        if system_prompt.lower() == "qiskit":
            resolved_prompt = get_system_prompt("qiskit_humaneval")
        elif system_prompt.lower() == "minimal":
            resolved_prompt = get_system_prompt("qiskit_humaneval_minimal")
        else:
            resolved_prompt = system_prompt

    # Determine output path
    if output:
        results_path = output
    else:
        results_path = ResultsManager.get_result_path(
            dataset_type="qiskit_humaneval",
            model_name=model_name,
            num_samples_per_task=num_samples,
            k_values=k_list,
            dataset_variant=dataset_type,
            timestamp=run_timestamp,
        )

    ResultsManager.ensure_output_dir(results_path)
    console.print(f"[dim]Results will be saved to: {results_path}[/dim]\n")

    client = LLMClient(
        base_url=model_url,
        api_key=api_key or os.getenv("API_KEY", ""),
        model_name=model_name,
    )

    runner = QiskitHumanEvalRunner(
        dataset_path=dataset_path,
        model_client=client,
        k_values=k_list,
        num_samples_per_task=num_samples,
        timeout=timeout,
        dataset_type=dataset_type,
    )

    samples = runner.load_dataset()

    if max_samples:
        samples = samples[:max_samples]
        console.print(f"[yellow]Limiting to first {max_samples} samples[/yellow]")

    try:
        results = runner.evaluate(
            samples=samples,
            system_prompt=resolved_prompt,
            save_results=results_path,
            verify_canonical=do_verify_canonical,
            model_name=model_name,
            run_timestamp=run_timestamp,
        )

        console.print("\n[green]✓ Evaluation complete![/green]")
        console.print(f"Success rate: {results.success_rate:.1%}")

    except Exception as e:
        console.print(f"\n[red]✗ Evaluation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def synthetic(
    dataset_path: Path = typer.Option(
        ..., "--dataset", "-d", help="Path to HuggingFace dataset directory or file"
    ),
    model_url: str = typer.Option(..., "--model-url", help="Model API base URL"),
    model_name: str = typer.Option("qwen", "--model-name", help="Model name"),
    api_key: str = typer.Option("", "--api-key", help="API key (optional)"),
    images_dir: Path = typer.Option(None, "--images-dir", help="Directory containing images"),
    is_vlm: bool = typer.Option(False, "--vlm", help="Use Vision-Language Model"),
    split: str = typer.Option("test", "--split", help="Dataset split to evaluate"),
    text_only: bool = typer.Option(
        False, "--text-only", help="Filter to text-only samples (no images)"
    ),
    system_prompt: str = typer.Option(
        "",
        "--system-prompt",
        "-s",
        help="System prompt: 'qiskit', 'minimal', or custom text",
    ),
    num_samples: int = typer.Option(
        1, "--num-samples", "-n", help="Number of solutions per task", min=1
    ),
    k_values: str = typer.Option("1", "--k-values", help="Comma-separated k values for Pass@k"),
    timeout: int = typer.Option(30, "--timeout", help="Execution timeout in seconds"),
    max_concurrent: int = typer.Option(10, "--max-concurrent", help="Maximum concurrent requests"),
    output: Path = typer.Option(None, "--output", "-o", help="Path to save evaluation results"),
    max_samples: int = typer.Option(
        None, "--max-samples", help="Limit number of samples to evaluate"
    ),
    do_verify_canonical: bool = typer.Option(
        False, "--verify-canonical", help="Verify canonical solutions pass tests first"
    ),
):
    """Evaluate model on synthetic multimodal dataset."""
    console.print("\n[bold cyan]Synthetic Dataset Evaluation[/bold cyan]\n")

    run_timestamp = datetime.now()
    k_list = [int(k.strip()) for k in k_values.split(",")]

    # Resolve system prompt
    resolved_prompt = None
    if system_prompt:
        if system_prompt.lower() == "qiskit":
            resolved_prompt = get_system_prompt("qiskit_humaneval")
        elif system_prompt.lower() == "minimal":
            resolved_prompt = get_system_prompt("qiskit_humaneval_minimal")
        else:
            resolved_prompt = system_prompt

    # Determine output path
    if output:
        results_path = output
    else:
        results_path = ResultsManager.get_result_path(
            dataset_type="synthetic",
            model_name=model_name,
            num_samples_per_task=num_samples,
            k_values=k_list,
            timestamp=run_timestamp,
        )

    ResultsManager.ensure_output_dir(results_path)
    console.print(f"[dim]Results will be saved to: {results_path}[/dim]\n")

    # Create client
    if is_vlm:
        client = VLMClient(
            base_url=model_url,
            api_key=api_key or os.getenv("API_KEY", ""),
            model_name=model_name,
        )
    else:
        client = LLMClient(
            base_url=model_url,
            api_key=api_key or os.getenv("API_KEY", ""),
            model_name=model_name,
        )

    runner = SyntheticDatasetRunner(
        dataset_path=dataset_path,
        model_client=client,
        images_dir=images_dir,
        k_values=k_list,
        num_samples_per_task=num_samples,
        timeout=timeout,
        max_concurrent=max_concurrent,
    )

    samples = runner.load_dataset(split=split)

    # Filter to text-only samples if requested
    if text_only:
        original_count = len(samples)
        samples = [s for s in samples if s.get("image") is None]
        console.print(
            f"[yellow]Filtering to text-only: {len(samples)}/{original_count} samples[/yellow]"
        )

    if max_samples:
        samples = samples[:max_samples]
        console.print(f"[yellow]Limiting to first {max_samples} samples[/yellow]")

    try:
        results = runner.evaluate(
            samples=samples,
            split=split,
            system_prompt=resolved_prompt,
            save_results=results_path,
            verify_canonical=do_verify_canonical,
            model_name=model_name,
            run_timestamp=run_timestamp,
        )

        console.print("\n[green]✓ Evaluation complete![/green]")
        console.print(f"Success rate: {results.success_rate:.1%}")

    except Exception as e:
        console.print(f"\n[red]✗ Evaluation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def verify_canonical(
    dataset_path: Path = typer.Option(..., "--dataset", "-d", help="Path to Qiskit HumanEval JSON"),
    timeout: int = typer.Option(30, "--timeout", help="Execution timeout in seconds"),
    output: Path = typer.Option(None, "--output", "-o", help="Path to save verification results"),
    max_samples: int = typer.Option(
        None, "--max-samples", help="Limit number of samples to verify"
    ),
    dataset_type: str = typer.Option(
        None, "--dataset-type", help="Dataset type: 'normal' or 'hard' (auto-detected if not set)"
    ),
):
    """Verify that canonical solutions in Qiskit HumanEval dataset pass their tests."""
    client = LLMClient(base_url="http://localhost", model_name="dummy")

    runner = QiskitHumanEvalRunner(
        dataset_path=dataset_path,
        model_client=client,
        timeout=timeout,
        dataset_type=dataset_type,
    )

    samples = runner.load_dataset()

    if max_samples:
        samples = samples[:max_samples]
        console.print(f"[yellow]Limiting to first {max_samples} samples[/yellow]")

    try:
        result = runner.verify_canonical_solutions(samples, save_results=output)

        if result["failed"] == 0:
            console.print("\n[green]✓ All canonical solutions passed![/green]")
        else:
            console.print(f"\n[yellow]⚠ {result['failed']} canonical solutions failed[/yellow]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[red]✗ Verification failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def compare(
    results_dir: Path = typer.Option(
        Path("outputs/evaluate"),
        "--results-dir",
        "-r",
        help="Directory containing result JSON files",
    ),
    dataset_type: str = typer.Option(
        None,
        "--dataset-type",
        "-t",
        help="Filter by dataset type (qiskit-humaneval, qiskit-humaneval-hard, synthetic)",
    ),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", help="Search subdirectories recursively"
    ),
):
    """Compare evaluation results from multiple models."""
    console.print("\n[bold cyan]Model Comparison[/bold cyan]\n")

    import json
    from rich.table import Table

    search_path = Path(results_dir)
    if dataset_type:
        search_path = search_path / dataset_type

    if recursive:
        results_files = list(search_path.rglob("*.json"))
    else:
        results_files = list(search_path.glob("*.json"))

    if not results_files:
        console.print(f"[red]No result files found in {search_path}![/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Found {len(results_files)} result file(s)[/dim]\n")

    all_results = {}
    for result_file in results_files:
        try:
            with open(result_file, encoding="utf-8") as f:
                data = json.load(f)

            metadata = data.get("metadata", {})
            if isinstance(metadata, dict) and "model" in metadata:
                model_name = metadata["model"].get("name", result_file.stem)
            else:
                model_name = result_file.stem

            dataset_info = ""
            if isinstance(metadata, dict):
                if "dataset" in metadata:
                    variant = metadata["dataset"].get("variant", "")
                    if variant:
                        dataset_info = f" ({variant})"
                elif "dataset_type" in metadata:
                    dataset_info = f" ({metadata['dataset_type']})"

            rel_path = result_file.relative_to(results_dir)
            display_name = f"{model_name}{dataset_info}"

            all_results[str(rel_path)] = {
                "display_name": display_name,
                "data": data,
                "path": result_file,
            }
        except (json.JSONDecodeError, KeyError) as e:
            console.print(f"[yellow]Warning: Could not parse {result_file}: {e}[/yellow]")
            continue

    if not all_results:
        console.print("[red]No valid result files found![/red]")
        raise typer.Exit(1)

    table = Table(title="Model Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Dataset", style="dim")

    metric_keys = set()
    for entry in all_results.values():
        data = entry["data"]
        if "metrics" in data:
            if "overall" in data["metrics"]:
                metric_keys.update(data["metrics"]["overall"].keys())
            else:
                metric_keys.update(data["metrics"].keys())

    priority_metrics = ["pass@1", "pass@5", "pass@10", "rouge_l", "bleu", "execution_accuracy"]
    sorted_metrics = [m for m in priority_metrics if m in metric_keys]
    sorted_metrics.extend(
        sorted(
            m
            for m in metric_keys
            if m not in priority_metrics and m not in ["count", "success_rate"]
        )
    )

    for metric in sorted_metrics:
        table.add_column(metric.replace("_", " ").title(), justify="right")

    for rel_path, entry in sorted(all_results.items(), key=lambda x: x[1]["display_name"]):
        data = entry["data"]
        metrics = data.get("metrics", {})
        if "overall" in metrics:
            metrics = metrics["overall"]

        metadata = data.get("metadata", {})
        if isinstance(metadata, dict) and "dataset" in metadata:
            dataset_name = Path(metadata["dataset"].get("path", "")).stem[:30]
        else:
            dataset_name = rel_path.split("/")[0] if "/" in rel_path else "-"

        row = [entry["display_name"], dataset_name]
        for metric in sorted_metrics:
            value = metrics.get(metric, "-")
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))

        table.add_row(*row)

    console.print(table)
