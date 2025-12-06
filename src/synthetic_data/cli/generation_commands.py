"""Generation stage CLI commands."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TaskID,
)

from synthetic_data.config import PipelineConfig
from synthetic_data.generators.stages import (
    AnswerStage,
    ClassifyStage,
    CurateStage,
    FilterCandidatesStage,
    PlanStage,
)
from synthetic_data.models import ModelRegistry

console = Console()


class ProgressManager:
    """Manages progress bar updates with total tracking."""

    def __init__(self, progress: Progress, task: TaskID):
        self.progress = progress
        self.task = task

    def set_total(self, total: int):
        """Set the total count."""
        self.progress.update(self.task, total=total)

    def update(self, completed: int):
        """Update completed count."""
        self.progress.update(self.task, completed=completed)


def plan(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
):
    """Step 6a: Plan - Generate questions and tests from chunks."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 6a: Input Planning[/bold cyan]\n"
            "Generate questions and unit tests from chunks",
            title="Plan",
        )
    )

    config = PipelineConfig.from_yaml(config_path)
    base_dir = Path(config.dataset.parsed_dir).parent

    with ModelRegistry(config.models) as model_registry:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Planning...", total=None)
            progress_mgr = ProgressManager(progress, task)

            stage = PlanStage(config, model_registry, base_dir, no_cache)
            stage.run(progress_mgr)


def filter_candidates(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
):
    """Step 6b: Filter Candidates - Filter candidates for quality."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 6b: Candidate Filtering[/bold cyan]\n"
            "Filter candidates for quality using LLM",
            title="Filter Candidates",
        )
    )

    config = PipelineConfig.from_yaml(config_path)
    base_dir = Path(config.dataset.parsed_dir).parent

    with ModelRegistry(config.models) as model_registry:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Filtering candidates...", total=None)
            progress_mgr = ProgressManager(progress, task)

            stage = FilterCandidatesStage(config, model_registry, base_dir, no_cache)
            stage.run(progress_mgr)


def answer(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
):
    """Step 6c: Answer - Generate and validate answers."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 6c: Answer Generation[/bold cyan]\n"
            "Generate and validate answers for candidates",
            title="Answer",
        )
    )

    config = PipelineConfig.from_yaml(config_path)
    base_dir = Path(config.dataset.parsed_dir).parent

    with ModelRegistry(config.models) as model_registry:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating answers...", total=None)
            progress_mgr = ProgressManager(progress, task)

            stage = AnswerStage(config, model_registry, base_dir, no_cache)
            stage.run(progress_mgr)


def curate(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
):
    """Step 6d: Curate - Quality curation of samples."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 6d: Quality Curation[/bold cyan]\n"
            "Final quality check on generated samples",
            title="Curate",
        )
    )

    config = PipelineConfig.from_yaml(config_path)
    base_dir = Path(config.dataset.parsed_dir).parent

    with ModelRegistry(config.models) as model_registry:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Curating samples...", total=None)
            progress_mgr = ProgressManager(progress, task)

            stage = CurateStage(config, model_registry, base_dir, no_cache)
            stage.run(progress_mgr)


def classify(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
):
    """Step 6e: Classify - Classify samples into categories."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 6e: Classification[/bold cyan]\n" "Classify samples into categories",
            title="Classify",
        )
    )

    config = PipelineConfig.from_yaml(config_path)
    base_dir = Path(config.dataset.parsed_dir).parent

    with ModelRegistry(config.models) as model_registry:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Classifying samples...", total=None)
            progress_mgr = ProgressManager(progress, task)

            stage = ClassifyStage(config, model_registry, base_dir, no_cache)
            stage.run(progress_mgr)


def generate(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
):
    """Step 6: Generate - Run all generation stages (plan → filter → answer → curate → classify)."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 6: Sample Generation[/bold cyan]\n"
            "Run all generation stages sequentially",
            title="Generate",
        )
    )

    config = PipelineConfig.from_yaml(config_path)
    base_dir = Path(config.dataset.parsed_dir).parent

    stages = [
        ("6a: Plan", PlanStage),
        ("6b: Filter Candidates", FilterCandidatesStage),
        ("6c: Answer", AnswerStage),
        ("6d: Curate", CurateStage),
        ("6e: Classify", ClassifyStage),
    ]

    with ModelRegistry(config.models) as model_registry:
        for stage_name, stage_class in stages:
            console.print(f"\n[bold]━━━ {stage_name} ━━━[/bold]\n")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Running {stage_name}...", total=None)
                progress_mgr = ProgressManager(progress, task)

                stage = stage_class(config, model_registry, base_dir, no_cache)
                stage.run(progress_mgr)

    console.print(
        Panel.fit(
            "[bold green]✓ Generation Complete![/bold green]\n"
            "All generation stages completed successfully",
            title="Success",
            border_style="green",
        )
    )
