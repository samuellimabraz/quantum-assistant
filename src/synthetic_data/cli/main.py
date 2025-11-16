"""Main CLI application."""

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from synthetic_data.config import PipelineConfig
from synthetic_data.cli.commands import (
    parse,
    transcribe,
    chunk,
    filter_quality,
    classify,
    generate,
    build,
    export,
    pipeline,
)
from synthetic_data.utils import PipelineCache

app = typer.Typer(
    name="synthetic-data",
    help="Clean synthetic dataset generation pipeline",
    add_completion=False,
)

console = Console()

# Register pipeline commands
app.command(name="parse", help="Step 1: Parse documents and resolve images")(parse)
app.command(name="transcribe", help="Step 2: Transcribe images using VLM")(transcribe)
app.command(name="chunk", help="Step 3: Chunk documents into context-sized pieces")(chunk)
app.command(name="filter", help="Step 4: Filter chunks for quality")(filter_quality)
app.command(name="classify", help="Step 5: Classify chunks into categories")(classify)
app.command(name="generate", help="Step 6: Generate synthetic Q&A samples")(generate)
app.command(name="build", help="Step 7: Build train/val/test splits")(build)
app.command(name="export", help="Step 8: Export to HuggingFace format")(export)
app.command(name="pipeline", help="Run complete pipeline (all steps)")(pipeline)


@app.command()
def info(config_path: Path = typer.Option(..., "--config", "-c", help="Config file")):
    """Display configuration information."""
    config = PipelineConfig.from_yaml(config_path)

    console.print("\n[bold cyan]Pipeline Configuration[/bold cyan]\n")

    # Sources
    console.print("[bold]Sources:[/bold]")
    for i, source in enumerate(config.sources, 1):
        console.print(f"  {i}. {source.path} ({source.type.value})")

    # Categories
    console.print("\n[bold]Categories:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="dim")

    for cat in config.categories:
        desc = cat.description[:50] + "..." if len(cat.description) > 50 else cat.description
        table.add_row(cat.name, desc)

    console.print(table)

    # Generation
    console.print("\n[bold]Generation:[/bold]")
    console.print(f"  Target samples: {config.generation.target_samples:,}")
    console.print(f"  Question model: {config.generation.question_model}")
    console.print(f"  Vision model: {config.generation.vision_model or 'None'}")
    console.print(f"  Answer model: {config.generation.answer_model}")
    console.print(f"  Curate model: {config.generation.curate_model}")
    console.print(f"  Multimodal ratio: {config.generation.multimodal_ratio:.1%}")
    console.print(f"  Content filtering: {config.generation.enable_content_filtering}")
    console.print(f"  Deduplication: {config.generation.enable_deduplication}")

    # Dataset
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

        # Check if sources exist
        for source in config.sources:
            if not Path(source.path).exists():
                console.print(
                    f"[yellow]⚠ Warning: Source path does not exist: {source.path}[/yellow]"
                )

        # Validate splits
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

    cache_info = cache.get_cache_info()
    if not cache_info:
        console.print("[yellow]No cache found[/yellow]")
        return

    console.print("\n[bold cyan]Cache Information[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Stage", style="cyan")
    table.add_column("Items", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Timestamp")

    total_size = 0
    for stage, info in cache_info.items():
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
