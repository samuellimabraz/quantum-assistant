"""Command-line interface for fine-tuning data preparation."""

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import FinetuneConfig, QuestionType
from .preparer import DatasetPreparer

app = typer.Typer(
    name="finetune",
    help="Fine-tuning data preparation tools for ms-swift framework.",
    no_args_is_help=True,
)
console = Console()


class ImageFormat(str, Enum):
    """Supported image formats."""

    JPEG = "JPEG"
    PNG = "PNG"


@app.command()
def prepare(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to YAML configuration file"),
    ] = None,
    dataset_path: Annotated[
        Path,
        typer.Option("--dataset-path", "-d", help="Path to HuggingFace dataset directory"),
    ] = Path("outputs/final"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Output directory for prepared data"),
    ] = Path("outputs/finetune"),
    max_size: Annotated[
        int,
        typer.Option("--max-size", help="Maximum image dimension (preserves aspect ratio)"),
    ] = 640,
    image_format: Annotated[
        ImageFormat,
        typer.Option("--image-format", help="Output image format"),
    ] = ImageFormat.JPEG,
    max_samples: Annotated[
        Optional[int],
        typer.Option("--max-samples", help="Maximum samples per split (for testing)"),
    ] = None,
    question_types: Annotated[
        Optional[str],
        typer.Option("--question-types", help="Comma-separated list of question types to include"),
    ] = None,
    no_system_prompt: Annotated[
        bool,
        typer.Option("--no-system-prompt", help="Exclude system prompt from messages"),
    ] = False,
):
    """Prepare dataset for ms-swift fine-tuning.

    Converts HuggingFace dataset to ms-swift JSONL format with:

    - Image resizing

    - Proper message structure with system/user/assistant roles

    - Image placeholders for multimodal samples
    """
    if config and config.exists():
        finetune_config = FinetuneConfig.from_yaml(config)
    else:
        qt_list = None
        if question_types:
            qt_list = [QuestionType(qt.strip()) for qt in question_types.split(",")]

        finetune_config = FinetuneConfig(
            dataset_path=dataset_path,
            output_dir=output_dir,
            max_samples=max_samples,
            question_types=qt_list,
        )
        finetune_config.image.max_size = max_size
        finetune_config.image.format = image_format.value
        finetune_config.swift.include_system_prompt = not no_system_prompt

    console.print(
        Panel.fit(
            "[bold cyan]ms-swift Dataset Preparation[/bold cyan]",
            border_style="cyan",
        )
    )

    info_table = Table(show_header=False, box=None)
    info_table.add_column("Property", style="dim")
    info_table.add_column("Value", style="green")
    info_table.add_row("Dataset path", str(finetune_config.dataset_path))
    info_table.add_row("Output directory", str(finetune_config.output_dir))
    info_table.add_row("Image max size", str(finetune_config.image.max_size))
    info_table.add_row("Image format", finetune_config.image.format)
    info_table.add_row(
        "System prompt", "Yes" if finetune_config.swift.include_system_prompt else "No"
    )
    console.print(info_table)
    console.print()

    preparer = DatasetPreparer(finetune_config)
    result = preparer.prepare()

    console.print(
        Panel.fit(
            "[bold green]Preparation Complete[/bold green]",
            border_style="green",
        )
    )

    results_table = Table(title="Results by Split")
    results_table.add_column("Split", style="cyan")
    results_table.add_column("Output", style="dim")
    results_table.add_column("Total", justify="right")
    results_table.add_column("Multimodal", justify="right", style="yellow")
    results_table.add_column("Text-only", justify="right")

    for split_name, split_path in result.splits.items():
        stats = result.statistics[split_name]
        results_table.add_row(
            split_name,
            str(split_path.name),
            str(stats["total"]),
            str(stats["multimodal"]),
            str(stats["text_only"]),
        )

    console.print(results_table)


@app.command()
def init_config(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path for configuration file"),
    ] = Path("finetune_config.yaml"),
):
    """Generate a default configuration file.

    Creates a YAML configuration file with default values that can be
    customized for your fine-tuning needs.
    """
    config = FinetuneConfig()
    config.to_yaml(output)
    console.print(f"[green]Configuration file created:[/green] {output}")
    console.print("\nEdit this file to customize your preparation settings.")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
