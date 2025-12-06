"""Main CLI application."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from synthetic_data.config import PipelineConfig
from synthetic_data.cli.commands import (
    analyze,
    analyze_allocation,
    build,
    chunk,
    export,
    filter_images,
    filter_quality,
    inspect,
    inspect_traces,
    parse,
    transcribe,
)
from synthetic_data.cli.generation_commands import (
    answer,
    classify,
    curate,
    filter_candidates,
    generate,
    plan,
)
from synthetic_data.utils import PipelineCache

app = typer.Typer(
    name="synthetic-data",
    help="Synthetic dataset generation pipeline for quantum computing VLM fine-tuning",
    add_completion=False,
)

console = Console()

# Register pipeline commands - Document processing
app.command(name="parse", help="Step 1: Parse documents and resolve images")(parse)
app.command(name="transcribe", help="Step 2: Transcribe images using VLM")(transcribe)
app.command(name="filter-images", help="Step 3: Filter images for quality")(filter_images)
app.command(name="chunk", help="Step 4: Chunk documents into context-sized pieces")(chunk)
app.command(name="filter-chunks", help="Step 5: Filter chunks for quality")(filter_quality)

# Register pipeline commands - Generation stages
app.command(name="plan", help="Step 6a: Generate questions and tests from chunks")(plan)
app.command(name="filter-candidates", help="Step 6b: Filter candidates for quality")(
    filter_candidates
)
app.command(name="answer", help="Step 6c: Generate and validate answers")(answer)
app.command(name="curate", help="Step 6d: Quality curation of samples")(curate)
app.command(name="classify", help="Step 6e: Classify samples into categories")(classify)
app.command(name="generate", help="Step 6: Run all generation stages")(generate)

# Register pipeline commands - Dataset building
app.command(name="build", help="Step 7: Build train/val/test splits")(build)
app.command(name="export", help="Step 8: Export to HuggingFace format")(export)

# Register utility commands
app.command(name="analyze", help="Analyze dataset and generate visualizations")(analyze)
app.command(name="inspect", help="Inspect intermediate results (PKL files)")(inspect)
app.command(name="inspect-traces", help="Inspect generation prompts and responses")(inspect_traces)
app.command(name="analyze-allocation", help="Analyze allocation strategies and diversity")(
    analyze_allocation
)


@app.command()
def pipeline(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
    hub_id: str = typer.Option(None, "--hub-id", "-h", help="HuggingFace Hub ID"),
    token: str = typer.Option(None, "--token", "-t", help="HuggingFace token"),
    skip_analysis: bool = typer.Option(False, "--skip-analysis", help="Skip analysis step"),
):
    """Run the complete pipeline: parse → transcribe → filter-images → chunk → filter-chunks → generate → build → export → analyze."""
    from rich.panel import Panel
    import time

    console.print(
        Panel.fit(
            "[bold cyan]Complete Pipeline[/bold cyan]\nRunning all steps sequentially",
            title="Full Pipeline",
        )
    )

    start_time = time.time()

    # Build steps list
    steps = [
        ("Parse", lambda: parse(config_path, no_cache, False)),
        ("Transcribe", lambda: transcribe(config_path, no_cache)),
        ("Filter Images", lambda: filter_images(config_path, no_cache)),
        ("Chunk", lambda: chunk(config_path, no_cache)),
        ("Filter Chunks", lambda: filter_quality(config_path, no_cache)),
        ("Generate", lambda: generate(config_path, no_cache)),
        ("Build", lambda: build(config_path)),
        ("Export", lambda: export(config_path, hub_id, token, False)),
    ]

    if not skip_analysis:
        _hub_id = hub_id
        _token = token
        steps.append(
            (
                "Analyze",
                lambda: analyze(
                    config_path=config_path,
                    source="splits",
                    output_dir=None,
                    no_plots=False,
                    include_allocation=True,
                    include_pipeline=True,
                    hub_id=_hub_id,
                    token=_token,
                ),
            )
        )

    for i, (step_name, step_func) in enumerate(steps, 1):
        console.print(f"\n[bold]━━━ Step {i}/{len(steps)}: {step_name} ━━━[/bold]\n")
        try:
            step_func()
        except Exception as e:
            console.print(f"\n[red]✗ Pipeline failed at step {step_name}: {e}[/red]")
            raise

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    console.print(
        Panel.fit(
            f"[bold green]✓ Pipeline Complete![/bold green]\n\n"
            f"All {len(steps)} steps completed successfully\n"
            f"Total time: {minutes}m {seconds}s",
            title="Success",
            border_style="green",
        )
    )


@app.command()
def info(config_path: Path = typer.Option(..., "--config", "-c", help="Config file")):
    """Display configuration information."""
    config = PipelineConfig.from_yaml(config_path)

    console.print("\n[bold cyan]Pipeline Configuration[/bold cyan]\n")

    console.print("[bold]Sources:[/bold]")
    for i, source in enumerate(config.sources, 1):
        console.print(f"  {i}. {source.path} ({source.type.value})")

    console.print("\n[bold]Categories:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="dim")

    for cat in config.categories:
        desc = cat.description[:50] + "..." if len(cat.description) > 50 else cat.description
        table.add_row(cat.name, desc)

    console.print(table)

    console.print("\n[bold]Generation:[/bold]")
    console.print(f"  Target samples: {config.generation.target_samples:,}")
    console.print(f"  Over-allocation: {config.generation.over_allocation_factor}x")
    console.print(f"  Diversity weight: {config.generation.diversity_weight}")
    console.print(f"  Question model: {config.generation.question_model}")
    console.print(f"  Vision model: {config.generation.vision_model or 'None'}")
    console.print(f"  Answer model: {config.generation.answer_model}")
    console.print(f"  Curate model: {config.generation.curate_model}")
    console.print(f"  Candidate filtering: {config.generation.enable_candidate_filtering}")
    console.print(f"  Quality curation: {config.generation.enable_curate_filtering}")

    console.print("\n[bold]Dataset:[/bold]")
    console.print(f"  Name: {config.dataset.name}")
    console.print(f"  Parsed: {config.dataset.parsed_dir}")
    console.print(f"  Generated: {config.dataset.generated_dir}")
    console.print(f"  Final: {config.dataset.final_dir}")
    console.print(
        f"  Splits: {config.dataset.train_split:.0%} / "
        f"{config.dataset.val_split:.0%} / "
        f"{config.dataset.test_split:.0%}"
    )
    console.print()


@app.command()
def validate_config(config_path: Path = typer.Option(..., "--config", "-c", help="Config file")):
    """Validate configuration file."""
    try:
        config = PipelineConfig.from_yaml(config_path)
        console.print("[green]✓ Configuration is valid[/green]")

        for source in config.sources:
            if not Path(source.path).exists():
                console.print(
                    f"[yellow]⚠ Warning: Source path does not exist: {source.path}[/yellow]"
                )

        total_split = (
            config.dataset.train_split + config.dataset.val_split + config.dataset.test_split
        )
        if abs(total_split - 1.0) > 0.001:
            console.print(
                f"[yellow]⚠ Warning: Dataset splits sum to {total_split:.3f}, not 1.0[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]✗ Configuration error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def cache_info(config_path: Path = typer.Option(..., "--config", "-c", help="Config file")):
    """Show cache information."""
    config = PipelineConfig.from_yaml(config_path)
    cache_dir = Path(config.dataset.parsed_dir).parent / ".cache"
    cache = PipelineCache(cache_dir)

    cache_data = cache.get_cache_info()
    if not cache_data:
        console.print("[yellow]No cache found[/yellow]")
        return

    console.print("\n[bold cyan]Cache Information[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Stage", style="cyan")
    table.add_column("Items", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Timestamp")

    total_size = 0
    for stage, info in cache_data.items():
        size_mb = info["size"] / 1024 / 1024
        total_size += info["size"]
        table.add_row(
            stage,
            str(info["count"]),
            f"{size_mb:.1f} MB",
            info["timestamp"].split("T")[0],
        )

    console.print(table)
    console.print(f"\n[bold]Total cache size:[/bold] {total_size / 1024 / 1024:.1f} MB")


@app.command()
def cache_clear(
    config_path: Path = typer.Option(..., "--config", "-c", help="Config file"),
    stage: str = typer.Option(None, "--stage", "-s", help="Clear specific stage"),
):
    """Clear cache."""
    config = PipelineConfig.from_yaml(config_path)
    cache_dir = Path(config.dataset.parsed_dir).parent / ".cache"
    cache = PipelineCache(cache_dir)

    if stage:
        cache.clear_stage(stage)
        console.print(f"[green]✓ Cleared cache for stage: {stage}[/green]")
    else:
        cache.clear_all()
        console.print("[green]✓ Cleared all cache[/green]")


if __name__ == "__main__":
    app()
