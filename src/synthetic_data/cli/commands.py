"""Pipeline CLI commands."""

import asyncio
import json
import pickle
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.syntax import Syntax
from rich.table import Table

from synthetic_data.config import PipelineConfig
from synthetic_data.dataset import DatasetBuilder, HuggingFaceExporter
from synthetic_data.extractors import ContentChunker, DocumentIngestion, ImageTranscriber
from synthetic_data.generators import GenerationPipeline
from synthetic_data.models import ModelRegistry
from synthetic_data.parsers.base import Document, ImageType
from synthetic_data.utils import CheckpointManager, PipelineCache, QualityFilter

console = Console()


def parse(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
    clear_cache: bool = typer.Option(False, "--clear-cache", help="Clear cache before running"),
):
    """Step 1: Parse documents and resolve images."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 1: Document Parsing[/bold cyan]\n"
            "Parse documents and resolve image paths",
            title="Parse",
        )
    )

    config = PipelineConfig.from_yaml(config_path)
    images_dir = Path(config.dataset.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(config.dataset.parsed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir.parent / ".cache"
    cache = PipelineCache(cache_dir)

    if clear_cache:
        console.print("[yellow]Clearing cache...[/yellow]")
        cache.clear_stage("parse")

    output_file = output_dir / "documents.pkl"
    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Loading from: {output_file}[/cyan]")
        with open(output_file, "rb") as f:
            all_documents = pickle.load(f)
        console.print(f"[green]✓ Loaded {len(all_documents)} documents[/green]")
        return

    ingestion = DocumentIngestion(images_output_dir=images_dir)
    all_documents = []

    def parse_file_safe(file_path: Path, source) -> Optional[Document]:
        try:
            if ingestion.should_include(file_path, source):
                return ingestion.parse_file(file_path)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to parse {file_path}: {e}[/yellow]")
        return None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        all_files_to_process = []
        for source in config.sources:
            source_path = Path(source.path)
            files = []

            if source_path.is_file():
                files = [source_path]
            elif source_path.is_dir():
                for pattern in source.include_patterns:
                    files.extend(list(source_path.rglob(pattern)))

            if source.max_files:
                files = files[: source.max_files]

            all_files_to_process.extend([(f, source) for f in files])

        total_files = len(all_files_to_process)
        task = progress.add_task(f"Parsing {total_files} files...", total=total_files)

        max_workers = min(8, total_files)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(parse_file_safe, file_path, source): file_path
                for file_path, source in all_files_to_process
            }

            for future in as_completed(future_to_file):
                doc = future.result()
                if doc:
                    all_documents.append(doc)
                progress.update(task, advance=1)

    with open(output_file, "wb") as f:
        pickle.dump(all_documents, f)

    summary = {
        "total_documents": len(all_documents),
        "total_images": sum(len(doc.images) for doc in all_documents),
        "images_resolved": sum(
            1 for doc in all_documents for img in doc.images if img.resolved_path
        ),
        "total_code_blocks": sum(len(doc.code_blocks) for doc in all_documents),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table = Table(title="Parsing Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_row("Documents", str(len(all_documents)))
    table.add_row("Code Blocks", str(summary["total_code_blocks"]))
    table.add_row("Images Found", str(summary["total_images"]))
    table.add_row("Images Resolved", str(summary["images_resolved"]))

    console.print("\n")
    console.print(table)
    console.print(f"\n[green]✓ Saved to: {output_dir}[/green]")


def transcribe(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
):
    """Step 2: Transcribe images using VLM."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 2: Image Transcription[/bold cyan]\n"
            "Transcribe images using Vision Language Model",
            title="Transcribe",
        )
    )

    config = PipelineConfig.from_yaml(config_path)

    if not config.generation.enable_image_transcription or not config.generation.vision_model:
        console.print("[yellow]Image transcription disabled in config, skipping[/yellow]")
        return

    parsed_dir = Path(config.dataset.parsed_dir)
    input_file = parsed_dir / "documents.pkl"

    if not input_file.exists():
        console.print("[red]✗ No parsed documents found. Run 'parse' first.[/red]")
        raise typer.Exit(1)

    with open(input_file, "rb") as f:
        all_documents = pickle.load(f)

    console.print(f"[cyan]Loaded {len(all_documents)} documents[/cyan]")

    output_dir = Path(config.dataset.parsed_dir).parent / "transcribed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "documents.pkl"

    checkpoint_dir = output_dir.parent / ".checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir, "transcribe")

    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Transcriptions already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-transcribe[/yellow]")
        return

    if not no_cache and checkpoint_manager.exists():
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            all_documents, _ = checkpoint_data
            already_transcribed = sum(
                1 for doc in all_documents for img in doc.images if img.transcription
            )
            if already_transcribed > 0:
                console.print(
                    f"[cyan]Resuming from checkpoint: "
                    f"{already_transcribed} images already transcribed[/cyan]"
                )

    images_to_transcribe = sum(
        1
        for doc in all_documents
        for img in doc.images
        if img.resolved_path and not img.transcription
    )

    if images_to_transcribe == 0:
        console.print("[yellow]No images need transcription[/yellow]")
        with open(output_file, "wb") as f:
            pickle.dump(all_documents, f)
        checkpoint_manager.clear_checkpoint()
        return

    console.print(f"[cyan]Found {images_to_transcribe} images to transcribe[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        already_transcribed = sum(
            1 for doc in all_documents for img in doc.images if img.transcription
        )

        task = progress.add_task(
            "Transcribing images...",
            total=images_to_transcribe + already_transcribed,
            completed=already_transcribed,
        )

        def update_progress(completed):
            progress.update(task, completed=already_transcribed + completed)

        def save_checkpoint(documents):
            transcribed_count = sum(
                1 for doc in documents for img in doc.images if img.transcription
            )
            metadata = {
                "transcribed_count": transcribed_count,
                "total_images": images_to_transcribe + already_transcribed,
            }
            checkpoint_manager.save_checkpoint(documents, metadata)

        with ModelRegistry(config.models) as model_registry:
            try:
                vision_client = model_registry.get_vlm_client(config.generation.vision_model)
                transcriber = ImageTranscriber(
                    vision_client,
                    config.prompts.image_transcription,
                    system_prompt=getattr(config.prompts, "image_transcription_system", ""),
                    batch_size=config.generation.vlm_batch_size,
                    max_concurrent=config.generation.vlm_concurrency,
                )

                asyncio.run(
                    transcriber.transcribe_batch_documents_async(
                        all_documents,
                        progress_callback=update_progress,
                        checkpoint_callback=save_checkpoint,
                        checkpoint_interval=config.generation.vlm_batch_size,
                    )
                )

                transcribed_count = sum(
                    1 for doc in all_documents for img in doc.images if img.transcription
                )

                console.print(f"[green]✓ Transcribed {transcribed_count} images[/green]")

            except Exception as e:
                console.print(f"[red]✗ Transcription failed: {e}[/red]")
                console.print("[yellow]Progress has been saved to checkpoint[/yellow]")
                raise

    with open(output_file, "wb") as f:
        pickle.dump(all_documents, f)

    checkpoint_manager.clear_checkpoint()

    # Calculate image classification statistics
    all_images = [img for doc in all_documents for img in doc.images]
    transcribed_images = [img for img in all_images if img.transcription]
    classified_images = [img for img in transcribed_images if img.image_type != ImageType.UNKNOWN]

    # Count by type
    type_counts = {}
    for img in transcribed_images:
        type_name = img.image_type.value if img.image_type else "unknown"
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    # Summary table
    summary_table = Table(title="Transcription Complete")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="green", justify="right")
    summary_table.add_row("Total Images", str(len(all_images)))
    summary_table.add_row("Transcribed", str(len(transcribed_images)))
    summary_table.add_row("Classified", str(len(classified_images)))

    console.print("\n")
    console.print(summary_table)

    # Image type distribution table
    if type_counts:
        type_table = Table(title="Image Type Distribution")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", style="green", justify="right")
        type_table.add_column("Percentage", style="magenta", justify="right")

        total = len(transcribed_images)
        for img_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total * 100) if total > 0 else 0
            type_table.add_row(img_type, str(count), f"{pct:.1f}%")

        console.print("\n")
        console.print(type_table)

    # Save summary to JSON
    summary = {
        "total_images": len(all_images),
        "transcribed": len(transcribed_images),
        "classified": len(classified_images),
        "type_distribution": type_counts,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[green]✓ Saved to: {output_dir}[/green]")


def chunk(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
):
    """Step 3: Chunk documents into manageable pieces."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 3: Content Chunking[/bold cyan]\n"
            "Split documents into context-sized chunks",
            title="Chunk",
        )
    )

    config = PipelineConfig.from_yaml(config_path)

    transcribed_dir = Path(config.dataset.parsed_dir).parent / "transcribed"
    parsed_dir = Path(config.dataset.parsed_dir)

    if (transcribed_dir / "documents.pkl").exists():
        input_file = transcribed_dir / "documents.pkl"
        console.print("[cyan]Loading transcribed documents[/cyan]")
    elif (parsed_dir / "documents.pkl").exists():
        input_file = parsed_dir / "documents.pkl"
        console.print("[cyan]Loading parsed documents (no transcriptions)[/cyan]")
    else:
        console.print("[red]✗ No documents found. Run 'parse' first.[/red]")
        raise typer.Exit(1)

    with open(input_file, "rb") as f:
        all_documents = pickle.load(f)

    output_dir = Path(config.dataset.parsed_dir).parent / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks.pkl"

    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Chunks already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-chunk[/yellow]")
        return

    chunker = ContentChunker(
        max_length=config.generation.max_context_length,
        overlap=config.generation.chunk_overlap,
    )

    all_chunks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Chunking documents...", total=len(all_documents))

        max_workers = min(8, len(all_documents))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc = {
                executor.submit(chunker.chunk_document, doc): doc for doc in all_documents
            }

            for future in as_completed(future_to_doc):
                chunks = future.result()
                all_chunks.extend(chunks)
                progress.update(task, advance=1)

    with open(output_file, "wb") as f:
        pickle.dump(all_chunks, f)

    # Calculate image statistics
    all_images_in_chunks = []
    for chunk in all_chunks:
        all_images_in_chunks.extend(chunk.images)

    # Count unique images by image_id
    unique_images_in_chunks = {img.image_id: img for img in all_images_in_chunks if img.image_id}
    images_with_transcription = sum(
        1 for img in unique_images_in_chunks.values() if img.transcription
    )

    # Diagnostic: Compare with transcribed images
    transcribed_dir = Path(config.dataset.parsed_dir).parent / "transcribed"
    transcribed_count = 0
    missing_images = []
    if (transcribed_dir / "documents.pkl").exists():
        with open(transcribed_dir / "documents.pkl", "rb") as f:
            transcribed_docs = pickle.load(f)

        all_transcribed_images = [
            img for doc in transcribed_docs for img in doc.images if img.transcription
        ]
        transcribed_count = len(all_transcribed_images)
        transcribed_image_ids_set = {img.image_id for img in all_transcribed_images if img.image_id}
        missing_images = [
            {
                "image_id": img.image_id,
                "path": img.path,
                "alt_text": img.alt_text,
                "image_type": img.image_type.value if img.image_type else "unknown",
                "source": str(img.path).split(":")[0] if ":" in str(img.path) else "unknown",
            }
            for img in all_transcribed_images
            if img.image_id and img.image_id not in unique_images_in_chunks
        ]

    summary = {
        "total_chunks": len(all_chunks),
        "chunks_with_code": sum(1 for c in all_chunks if c.code_blocks),
        "chunks_with_images": sum(1 for c in all_chunks if c.images),
        "multimodal_chunks": sum(1 for c in all_chunks if c.is_multimodal),
        "avg_chunk_length": (
            sum(len(c.text) for c in all_chunks) / len(all_chunks) if all_chunks else 0
        ),
        "total_images_in_chunks": len(unique_images_in_chunks),
        "images_with_transcription": images_with_transcription,
        "transcribed_images_total": transcribed_count,
        "images_missing_from_chunks": len(missing_images),
    }

    # Save diagnostic file for missing images
    if missing_images:
        diagnostic_file = output_dir / "missing_images.jsonl"
        with open(diagnostic_file, "w", encoding="utf-8") as f:
            for img_info in missing_images:
                json.dump(img_info, f, ensure_ascii=False)
                f.write("\n")
        console.print(
            f"[yellow]⚠ {len(missing_images)} transcribed images not found in chunks (saved to {diagnostic_file.name})[/yellow]"
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table = Table(title="Chunking Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Total Chunks", str(len(all_chunks)))
    table.add_row("Chunks with Code", str(summary["chunks_with_code"]))
    table.add_row("Chunks with Images", str(summary["chunks_with_images"]))
    table.add_row("Multimodal Chunks", str(summary["multimodal_chunks"]))
    table.add_row("Images in Chunks", str(summary["total_images_in_chunks"]))
    if transcribed_count > 0:
        table.add_row(
            "Transcribed Images",
            f"{transcribed_count} (total) / {summary['images_with_transcription']} (in chunks)",
        )
        if missing_images:
            table.add_row(
                "Missing from Chunks",
                f"[red]{len(missing_images)}[/red]",
            )
    table.add_row("Avg Length", f"{summary['avg_chunk_length']:.0f} chars")

    console.print("\n")
    console.print(table)
    console.print(f"[green]✓ Saved to: {output_dir}[/green]")


def filter_quality(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
):
    """Step 4: Filter chunks for quality."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 4: Quality Filtering[/bold cyan]\n"
            "Filter low-quality content and images",
            title="Filter",
        )
    )

    config = PipelineConfig.from_yaml(config_path)

    if not config.generation.enable_content_filtering:
        console.print("[yellow]Content filtering disabled in config, skipping[/yellow]")

        chunks_dir = Path(config.dataset.parsed_dir).parent / "chunks"
        output_dir = Path(config.dataset.parsed_dir).parent / "filtered"
        output_dir.mkdir(parents=True, exist_ok=True)

        if (chunks_dir / "chunks.pkl").exists():
            import shutil

            shutil.copy2(chunks_dir / "chunks.pkl", output_dir / "chunks.pkl")
            console.print(f"[green]✓ Copied unfiltered chunks to: {output_dir}[/green]")
        return

    chunks_dir = Path(config.dataset.parsed_dir).parent / "chunks"
    input_file = chunks_dir / "chunks.pkl"

    if not input_file.exists():
        console.print("[red]✗ No chunks found. Run 'chunk' first.[/red]")
        raise typer.Exit(1)

    with open(input_file, "rb") as f:
        all_chunks = pickle.load(f)

    console.print(f"[cyan]Loaded {len(all_chunks)} chunks[/cyan]")

    # Diagnostic: Check how many transcribed images are in chunks
    transcribed_dir = Path(config.dataset.parsed_dir).parent / "transcribed"
    if (transcribed_dir / "documents.pkl").exists():
        with open(transcribed_dir / "documents.pkl", "rb") as f:
            transcribed_docs = pickle.load(f)

        all_transcribed_images = [
            img for doc in transcribed_docs for img in doc.images if img.transcription
        ]
        images_in_chunks = set()
        for chunk in all_chunks:
            for img in chunk.images:
                if img.image_id:
                    images_in_chunks.add(img.image_id)

        transcribed_image_ids = {img.image_id for img in all_transcribed_images if img.image_id}
        missing_from_chunks = transcribed_image_ids - images_in_chunks

        if missing_from_chunks:
            console.print(
                f"[yellow]⚠ Warning: {len(missing_from_chunks)} transcribed images not found in any chunk[/yellow]"
            )
            console.print(
                "[dim]  This may indicate images were in content that was filtered out or markers were not properly inserted[/dim]"
            )

    output_dir = Path(config.dataset.parsed_dir).parent / "filtered"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks.pkl"

    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Filtered chunks already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-filter[/yellow]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Filtering chunks...", total=len(all_chunks))

        def update_progress(completed):
            progress.update(task, completed=completed)

        with ModelRegistry(config.models) as model_registry:
            filter_model_name = config.generation.filter_model or config.generation.curate_model
            filter_client = model_registry.get_llm_client(filter_model_name)
            quality_filter = QualityFilter(filter_client)

            try:
                filter_results, debug_info = asyncio.run(
                    quality_filter.filter_chunks_batch_async(
                        all_chunks,
                        config.prompts.content_quality_check,
                        config.prompts.image_quality_check,
                        config.prompts.content_filter_system,
                        config.prompts.image_filter_system,
                        batch_size=config.generation.llm_batch_size,
                        max_concurrent=config.generation.llm_concurrency,
                        progress_callback=update_progress,
                        save_debug=True,
                    )
                )
            except Exception as e:
                console.print(f"[red]✗ Filtering failed: {e}[/red]")
                raise

    filtered_chunks = [chunk for chunk, passed in filter_results if passed]

    # Save detailed filter decisions
    debug_file = output_dir / "filter_decisions.jsonl"
    with open(debug_file, "w", encoding="utf-8") as f:
        for entry in debug_info:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    if debug_info:
        console.print(f"[dim]Saved {len(debug_info)} filter decisions to {debug_file.name}[/dim]")

    # Calculate image statistics
    images_before = sum(len(c.images) for c in all_chunks)
    images_after = sum(len(c.images) for c, passed in filter_results if passed)
    images_removed = images_before - images_after

    # Count unique images before and after
    unique_images_before = set()
    for chunk in all_chunks:
        for img in chunk.images:
            if img.image_id:
                unique_images_before.add(img.image_id)

    unique_images_after = set()
    for chunk, passed in filter_results:
        if passed:
            for img in chunk.images:
                if img.image_id:
                    unique_images_after.add(img.image_id)

    # Count multimodal chunks before and after
    multimodal_before = sum(1 for c in all_chunks if c.is_multimodal)
    multimodal_after = sum(1 for c in filtered_chunks if c.is_multimodal)

    # Count images by filter decision
    content_rejects = sum(
        1 for d in debug_info if d.get("type") == "content" and d.get("decision") == "REJECT"
    )
    image_rejects = sum(
        1 for d in debug_info if d.get("type") == "image" and d.get("decision") == "REJECT"
    )
    content_passes = sum(
        1 for d in debug_info if d.get("type") == "content" and d.get("decision") == "PASS"
    )
    image_passes = sum(
        1 for d in debug_info if d.get("type") == "image" and d.get("decision") == "PASS"
    )

    # Extract image evaluation statistics
    image_summary = next((d for d in debug_info if d.get("type") == "summary"), {})
    images_evaluated = image_summary.get("images_evaluated", 0)
    images_skipped_no_content = image_summary.get("images_skipped_no_content_pass", 0)
    images_skipped_no_transcription = image_summary.get("images_skipped_no_transcription", 0)
    images_skipped_no_resolved = image_summary.get("images_skipped_no_resolved_path", 0)

    with open(output_file, "wb") as f:
        pickle.dump(filtered_chunks, f)

    summary = {
        "chunks_before": len(all_chunks),
        "chunks_after": len(filtered_chunks),
        "chunks_removed": len(all_chunks) - len(filtered_chunks),
        "images_before": images_before,
        "images_after": images_after,
        "images_removed": images_removed,
        "unique_images_before": len(unique_images_before),
        "unique_images_after": len(unique_images_after),
        "multimodal_before": multimodal_before,
        "multimodal_after": multimodal_after,
        "content_decisions": {
            "passed": content_passes,
            "rejected": content_rejects,
        },
        "image_decisions": {
            "passed": image_passes,
            "rejected": image_rejects,
        },
        "image_evaluation_stats": {
            "evaluated": images_evaluated,
            "skipped_no_content_pass": images_skipped_no_content,
            "skipped_no_transcription": images_skipped_no_transcription,
            "skipped_no_resolved_path": images_skipped_no_resolved,
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table = Table(title="Filtering Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Before", style="yellow", justify="right")
    table.add_column("After", style="green", justify="right")
    table.add_column("Removed", style="red", justify="right")
    table.add_row(
        "Chunks",
        str(len(all_chunks)),
        str(len(filtered_chunks)),
        str(len(all_chunks) - len(filtered_chunks)),
    )
    table.add_row(
        "Images (total)",
        str(images_before),
        str(images_after),
        str(images_removed),
    )
    table.add_row(
        "Images (unique)",
        str(len(unique_images_before)),
        str(len(unique_images_after)),
        str(len(unique_images_before) - len(unique_images_after)),
    )
    table.add_row(
        "Multimodal Chunks",
        str(multimodal_before),
        str(multimodal_after),
        str(multimodal_before - multimodal_after),
    )

    console.print("\n")
    console.print(table)

    # Filter decision breakdown
    if debug_info:
        decision_table = Table(title="Filter Decisions")
        decision_table.add_column("Type", style="cyan")
        decision_table.add_column("Passed", style="green", justify="right")
        decision_table.add_column("Rejected", style="red", justify="right")
        decision_table.add_row(
            "Content",
            str(content_passes),
            str(content_rejects),
        )
        decision_table.add_row(
            "Images",
            str(image_passes),
            str(image_rejects),
        )

        console.print("\n")
        console.print(decision_table)

        # Image evaluation statistics
        if images_evaluated > 0 or images_skipped_no_content > 0:
            eval_table = Table(title="Image Evaluation Statistics")
            eval_table.add_column("Category", style="cyan")
            eval_table.add_column("Count", style="yellow", justify="right")
            eval_table.add_row("Images Evaluated", str(images_evaluated))
            if images_skipped_no_content > 0:
                eval_table.add_row(
                    "Skipped (content failed)",
                    f"[red]{images_skipped_no_content}[/red]",
                )
            if images_skipped_no_transcription > 0:
                eval_table.add_row(
                    "Skipped (no transcription)",
                    f"[yellow]{images_skipped_no_transcription}[/yellow]",
                )
            if images_skipped_no_resolved > 0:
                eval_table.add_row(
                    "Skipped (no resolved path)",
                    f"[yellow]{images_skipped_no_resolved}[/yellow]",
                )

            console.print("\n")
            console.print(eval_table)

    console.print(f"\n[green]✓ Saved to: {output_dir}[/green]")
    if debug_info:
        console.print(f"[dim]  Filter decisions: {debug_file.name}[/dim]")


def generate(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
    enable_tracing: bool = typer.Option(
        True, "--trace/--no-trace", help="Enable detailed prompt/response tracing"
    ),
):
    """Step 5: Generate synthetic Q&A samples with post-generation classification."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 5: Sample Generation[/bold cyan]\n"
            "Generate synthetic Q&A samples from chunks\n"
            "Includes input planning, answer generation, and classification",
            title="Generate",
        )
    )

    config = PipelineConfig.from_yaml(config_path)

    # Load chunks (prefer filtered, fallback to unfiltered)
    filtered_dir = Path(config.dataset.parsed_dir).parent / "filtered"
    chunks_dir = Path(config.dataset.parsed_dir).parent / "chunks"

    if (filtered_dir / "chunks.pkl").exists():
        input_file = filtered_dir / "chunks.pkl"
        console.print("[cyan]Loading filtered chunks[/cyan]")
    elif (chunks_dir / "chunks.pkl").exists():
        input_file = chunks_dir / "chunks.pkl"
        console.print("[cyan]Loading unfiltered chunks[/cyan]")
    else:
        console.print("[red]✗ No chunks found. Run 'chunk' first.[/red]")
        raise typer.Exit(1)

    with open(input_file, "rb") as f:
        all_chunks = pickle.load(f)

    console.print(f"[cyan]Loaded {len(all_chunks)} chunks[/cyan]")
    console.print(f"[cyan]Target: {config.generation.target_samples} samples[/cyan]")

    output_dir = Path(config.dataset.generated_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "samples.pkl"

    checkpoint_dir = output_dir.parent / ".checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir, "generate")

    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Samples already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-generate[/yellow]")
        return

    samples = []
    rejected_samples_list = []
    code_failures_list = []

    if not no_cache and checkpoint_manager.exists():
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            checkpoint_state, metadata = checkpoint_data
            samples = checkpoint_state.get("samples", [])
            rejected_samples_list = checkpoint_state.get("rejected_samples", [])
            code_failures_list = checkpoint_state.get("code_failures", [])
            current_stage = metadata.get("current_stage", "start")

            if samples:
                console.print(f"[cyan]Resuming from checkpoint: {len(samples)} samples[/cyan]")
            if current_stage != "start" and current_stage != "complete":
                console.print(f"[cyan]Resuming from stage: {current_stage}[/cyan]")

    console.print("\n[cyan]Pipeline stages:[/cyan]")
    console.print("  [dim]Stage 1: Input Planning (candidates per chunk)[/dim]")
    console.print("  [dim]Stage 2: Candidate Filtering[/dim]")
    console.print("  [dim]Stage 3: Answer Generation + Validation[/dim]")
    if config.generation.enable_curate_filtering:
        console.print("  [dim]Stage 4: Quality Curation[/dim]")
    console.print("  [dim]Stage 5: Post-Generation Classification[/dim]")

    if enable_tracing:
        trace_dir = output_dir / "traces"
        console.print(f"\n[dim]Tracing enabled: {trace_dir}[/dim]")
        console.print("[dim]  - generation_traces.jsonl: Individual prompt/response entries[/dim]")
        console.print("[dim]  - conversations.jsonl: Complete conversation threads[/dim]")
    console.print()

    # Track current stage for proper progress display
    current_stage_name = {"value": "input_generation"}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,  # Keep completed tasks visible
    ) as progress:
        # Create all tasks upfront but hide some initially
        questions_task = progress.add_task(
            "[cyan]Stage 1: Input Planning[/cyan]", total=None, visible=True
        )
        filter_task = progress.add_task(
            "[cyan]Stage 2: Candidate Filtering[/cyan]", total=None, visible=False
        )
        answers_task = progress.add_task(
            "[cyan]Stage 3: Answer Generation[/cyan]", total=None, visible=False
        )
        curation_task = progress.add_task(
            "[cyan]Stage 4: Quality Curation[/cyan]", total=None, visible=False
        )
        classification_task = progress.add_task(
            "[cyan]Stage 5: Classification[/cyan]", total=None, visible=False
        )

        def set_questions_total(total):
            progress.update(questions_task, total=total)

        def update_question_progress(completed):
            progress.update(questions_task, completed=completed)

        def set_filter_total(total):
            progress.update(filter_task, total=total, visible=True)

        def update_filter_progress(completed):
            progress.update(filter_task, completed=completed)

        def set_answers_total(total):
            progress.update(answers_task, total=total, visible=True)

        def update_answer_progress(completed):
            progress.update(answers_task, completed=completed)

        def set_curation_total(total):
            progress.update(curation_task, total=total, visible=True)

        def update_curation_progress(completed):
            progress.update(curation_task, completed=completed)

        def set_classification_total(total):
            progress.update(classification_task, total=total, visible=True)

        def update_classification_progress(completed):
            progress.update(classification_task, completed=completed)

        def handle_stage_change(stage_name):
            """Handle stage changes to update progress visibility."""
            current_stage_name["value"] = stage_name

            # Mark completed stages as done
            if stage_name == "candidate_filtering":
                # Stage 1 complete, show stage 2
                progress.update(
                    questions_task, description="[green]✓ Stage 1: Input Planning[/green]"
                )
                progress.update(filter_task, visible=True)
            elif stage_name == "answer_generation":
                # Stage 2 complete, show stage 3
                progress.update(
                    filter_task, description="[green]✓ Stage 2: Candidate Filtering[/green]"
                )
                progress.update(answers_task, visible=True)
            elif stage_name == "quality_curation":
                # Stage 3 complete, show stage 4
                progress.update(
                    answers_task, description="[green]✓ Stage 3: Answer Generation[/green]"
                )
                progress.update(curation_task, visible=True)
            elif stage_name == "classification":
                # Stage 4 complete, show stage 5
                progress.update(
                    curation_task, description="[green]✓ Stage 4: Quality Curation[/green]"
                )
                progress.update(classification_task, visible=True)

        def save_rejected(rejected):
            rejected_samples_list.extend(rejected)

        def save_code_failures(failures_batch):
            code_failures_list.extend(failures_batch)

        progress_callbacks = {
            "set_questions_total": set_questions_total,
            "questions": update_question_progress,
            "set_filter_total": set_filter_total,
            "filter": update_filter_progress,
            "set_answers_total": set_answers_total,
            "answers": update_answer_progress,
            "set_curation_total": set_curation_total,
            "curation": update_curation_progress,
            "set_classification_total": set_classification_total,
            "classification": update_classification_progress,
            "stage_change": handle_stage_change,
            "save_rejected": save_rejected,
            "save_code_failures": save_code_failures,
            "no_cache": no_cache,
        }

        with ModelRegistry(config.models) as model_registry:
            gen_pipeline = GenerationPipeline(
                config,
                model_registry,
                checkpoint_manager,
                output_dir=output_dir,
                enable_tracing=enable_tracing,
            )

            try:
                samples = gen_pipeline.generate_samples(all_chunks, progress_callbacks)

                # Mark final stage as complete
                progress.update(
                    classification_task, description="[green]✓ Stage 5: Classification[/green]"
                )

            except Exception as e:
                console.print(f"\n[red]✗ Generation failed: {e}[/red]")
                console.print("[yellow]Progress has been saved to checkpoint[/yellow]")
                if enable_tracing:
                    console.print(f"[yellow]Check traces at: {output_dir / 'traces'}[/yellow]")
                raise

    console.print()

    # Save samples
    with open(output_file, "wb") as f:
        pickle.dump(samples, f)

    checkpoint_manager.clear_checkpoint()

    # Save JSONL
    with open(output_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for sample in samples:
            json.dump(
                {
                    "question": sample.question,
                    "answer": sample.answer,
                    "category": sample.category,
                    "question_type": sample.question_type,
                    "test_code": sample.test_code,
                    "entry_point": sample.entry_point,
                    "image_path": sample.image_path,
                    "source_path": sample.source_path,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

    # Save rejected samples
    if rejected_samples_list:
        with open(output_dir / "rejected_samples.jsonl", "w", encoding="utf-8") as f:
            for rejected in rejected_samples_list:
                json.dump(rejected, f, ensure_ascii=False)
                f.write("\n")

        console.print(f"\n[yellow]ℹ {len(rejected_samples_list)} rejected samples saved[/yellow]")

    # Save code verification failures
    if code_failures_list:
        with open(output_dir / "code_verification_failures.jsonl", "w", encoding="utf-8") as f:
            for failure in code_failures_list:
                json.dump(failure, f, ensure_ascii=False)
                f.write("\n")

        console.print(
            f"[yellow]ℹ {len(code_failures_list)} code verification failures saved[/yellow]"
        )

    # Summary
    samples_with_tests = sum(1 for s in samples if s.test_code)
    summary = {
        "total_samples": len(samples),
        "rejected_samples": len(rejected_samples_list),
        "code_verification_failures": len(code_failures_list),
        "multimodal_samples": sum(1 for s in samples if s.image_path),
        "text_only_samples": sum(1 for s in samples if not s.image_path),
        "samples_with_tests": samples_with_tests,
        "by_type": {},
        "by_category": {},
    }

    for sample in samples:
        summary["by_type"][sample.question_type] = (
            summary["by_type"].get(sample.question_type, 0) + 1
        )
        summary["by_category"][sample.category] = summary["by_category"].get(sample.category, 0) + 1

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table = Table(title="Generation Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Total Samples", str(len(samples)))
    table.add_row("With Unit Tests", str(samples_with_tests))
    table.add_row("Multimodal", str(summary["multimodal_samples"]))
    table.add_row("Text-only", str(summary["text_only_samples"]))

    console.print("\n")
    console.print(table)

    # Type distribution
    type_table = Table(title="By Question Type")
    type_table.add_column("Type", style="cyan")
    type_table.add_column("Count", style="green", justify="right")
    for qtype, count in sorted(summary["by_type"].items()):
        type_table.add_row(qtype, str(count))

    console.print("\n")
    console.print(type_table)

    # Category distribution
    if summary["by_category"]:
        cat_table = Table(title="By Category")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", style="green", justify="right")
        for cat, count in sorted(summary["by_category"].items(), key=lambda x: x[1], reverse=True):
            cat_table.add_row(cat, str(count))

        console.print("\n")
        console.print(cat_table)

    console.print(f"\n[green]✓ Saved to: {output_dir}[/green]")

    # Show trace file locations if tracing was enabled
    trace_dir = output_dir / "traces"
    if trace_dir.exists():
        console.print("\n[dim]Generation traces saved to:[/dim]")
        console.print(
            f"  [dim]• {trace_dir / 'generation_traces.jsonl'} - Individual entries[/dim]"
        )
        console.print(
            f"  [dim]• {trace_dir / 'conversations.jsonl'} - Complete conversations[/dim]"
        )
        console.print(f"\n[dim]Inspect with: jq . {trace_dir / 'conversations.jsonl'} | less[/dim]")


def build(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
):
    """Step 6: Build train/val/test splits."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 6: Dataset Splits[/bold cyan]\n"
            "Build stratified train/validation/test splits",
            title="Build",
        )
    )

    config = PipelineConfig.from_yaml(config_path)

    generated_dir = Path(config.dataset.generated_dir)
    input_file = generated_dir / "samples.pkl"

    if not input_file.exists():
        console.print("[red]✗ No samples found. Run 'generate' first.[/red]")
        raise typer.Exit(1)

    with open(input_file, "rb") as f:
        samples = pickle.load(f)

    console.print(f"[cyan]Loaded {len(samples)} samples[/cyan]")

    output_dir = Path(config.dataset.parsed_dir).parent / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = DatasetBuilder(config.dataset, config.seed)
    train_samples, val_samples, test_samples = builder.stratified_build(samples)

    builder.print_split_comparison(train_samples, val_samples, test_samples)

    with open(output_dir / "train.pkl", "wb") as f:
        pickle.dump(train_samples, f)

    with open(output_dir / "val.pkl", "wb") as f:
        pickle.dump(val_samples, f)

    with open(output_dir / "test.pkl", "wb") as f:
        pickle.dump(test_samples, f)

    summary = {
        "total_samples": len(samples),
        "train": builder.get_distribution_stats(train_samples),
        "val": builder.get_distribution_stats(val_samples),
        "test": builder.get_distribution_stats(test_samples),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table = Table(title="Dataset Splits")
    table.add_column("Split", style="cyan")
    table.add_column("Samples", style="green", justify="right")
    table.add_column("Percentage", style="magenta", justify="right")

    total = len(samples)
    table.add_row("Train", str(len(train_samples)), f"{len(train_samples)/total*100:.1f}%")
    table.add_row("Validation", str(len(val_samples)), f"{len(val_samples)/total*100:.1f}%")
    table.add_row("Test", str(len(test_samples)), f"{len(test_samples)/total*100:.1f}%")

    console.print("\n")
    console.print(table)
    console.print(f"[green]✓ Saved to: {output_dir}[/green]")


def export(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    hub_id: Optional[str] = typer.Option(None, "--hub-id", "-h", help="HuggingFace Hub ID"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="HuggingFace token"),
):
    """Step 7: Export dataset to HuggingFace format."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 7: Export Dataset[/bold cyan]\n"
            "Export to HuggingFace Dataset format",
            title="Export",
        )
    )

    config = PipelineConfig.from_yaml(config_path)

    splits_dir = Path(config.dataset.parsed_dir).parent / "splits"

    if not all((splits_dir / f"{split}.pkl").exists() for split in ["train", "val", "test"]):
        console.print("[red]✗ No splits found. Run 'build' first.[/red]")
        raise typer.Exit(1)

    with open(splits_dir / "train.pkl", "rb") as f:
        train_samples = pickle.load(f)

    with open(splits_dir / "val.pkl", "rb") as f:
        val_samples = pickle.load(f)

    with open(splits_dir / "test.pkl", "rb") as f:
        test_samples = pickle.load(f)

    console.print(
        f"[cyan]Loaded splits: train={len(train_samples)}, "
        f"val={len(val_samples)}, test={len(test_samples)}[/cyan]"
    )

    exporter = HuggingFaceExporter(config.dataset)
    dataset_dict = exporter.export(train_samples, val_samples, test_samples)

    output_path = exporter.save_to_disk(dataset_dict)
    console.print(f"[green]✓ Saved dataset to: {output_path}[/green]")

    if hub_id:
        console.print(f"\n[cyan]Pushing to HuggingFace Hub: {hub_id}[/cyan]")

        from datasets import DatasetDict

        non_empty_dict = DatasetDict(
            {split: dataset for split, dataset in dataset_dict.items() if len(dataset) > 0}
        )

        if non_empty_dict:
            non_empty_dict.push_to_hub(
                hub_id,
                token=token,
                private=False,
            )
            console.print(
                f"[green]✓ Dataset available at: https://huggingface.co/datasets/{hub_id}[/green]"
            )

    table = Table(title="Export Complete")
    table.add_column("Split", style="cyan")
    table.add_column("Samples", style="green", justify="right")

    for split in dataset_dict.keys():
        table.add_row(split, str(len(dataset_dict[split])))

    console.print("\n")
    console.print(table)


def inspect(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    stage: str = typer.Option(
        "chunks",
        "--stage",
        "-s",
        help="Stage to inspect: documents, transcribed, chunks, filtered, samples",
    ),
    count: int = typer.Option(3, "--count", "-n", help="Number of items to display"),
    index: Optional[int] = typer.Option(None, "--index", "-i", help="Specific item index"),
    show_images: bool = typer.Option(True, "--images/--no-images", help="Show image info"),
    show_code: bool = typer.Option(True, "--code/--no-code", help="Show code blocks"),
    random_sample: bool = typer.Option(False, "--random", "-r", help="Random sample"),
    export_jsonl: Optional[Path] = typer.Option(
        None, "--export", "-e", help="Export to JSONL file"
    ),
):
    """Inspect intermediate pipeline results (PKL files)."""
    console.print(
        Panel.fit(
            f"[bold cyan]Inspecting: {stage}[/bold cyan]",
            title="Inspect",
        )
    )

    config = PipelineConfig.from_yaml(config_path)
    base_dir = Path(config.dataset.parsed_dir).parent

    # Map stage to file
    stage_files = {
        "documents": base_dir / "parsed" / "documents.pkl",
        "transcribed": base_dir / "transcribed" / "documents.pkl",
        "chunks": base_dir / "chunks" / "chunks.pkl",
        "filtered": base_dir / "filtered" / "chunks.pkl",
        "samples": base_dir / "generated" / "samples.pkl",
    }

    if stage not in stage_files:
        console.print(f"[red]Unknown stage: {stage}[/red]")
        console.print(f"[yellow]Available: {', '.join(stage_files.keys())}[/yellow]")
        raise typer.Exit(1)

    pkl_file = stage_files[stage]
    if not pkl_file.exists():
        console.print(f"[red]File not found: {pkl_file}[/red]")
        console.print("[yellow]Run the corresponding pipeline step first.[/yellow]")
        raise typer.Exit(1)

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    console.print(f"[cyan]Loaded {len(data)} items from {pkl_file.name}[/cyan]\n")

    # Select items to display
    if index is not None:
        if 0 <= index < len(data):
            items = [data[index]]
            indices = [index]
        else:
            console.print(f"[red]Index {index} out of range (0-{len(data)-1})[/red]")
            raise typer.Exit(1)
    elif random_sample:
        indices = random.sample(range(len(data)), min(count, len(data)))
        items = [data[i] for i in indices]
    else:
        indices = list(range(min(count, len(data))))
        items = data[:count]

    # Export to JSONL if requested
    if export_jsonl:
        _export_items_to_jsonl(data, export_jsonl, stage)
        console.print(f"[green]✓ Exported {len(data)} items to {export_jsonl}[/green]")
        return

    # Display based on stage type
    if stage in ("documents", "transcribed"):
        _display_documents(items, indices, show_images)
    elif stage in ("chunks", "filtered"):
        _display_chunks(items, indices, show_images, show_code)
    elif stage == "samples":
        _display_samples(items, indices, show_code)


def _display_documents(items: list, indices: list[int], show_images: bool):
    """Display document items."""
    for idx, doc in zip(indices, items):
        console.print(Panel(f"[bold]Document {idx}[/bold]: {doc.source_path.name}"))
        console.print(f"[cyan]Title:[/cyan] {doc.title[:100] if doc.title else 'N/A'}...")
        console.print(f"[cyan]Content length:[/cyan] {len(doc.content):,} chars")
        console.print(f"[cyan]Code blocks:[/cyan] {len(doc.code_blocks)}")
        console.print(f"[cyan]Images:[/cyan] {len(doc.images)}")

        if show_images and doc.images:
            console.print("\n[bold]Images:[/bold]")
            for i, img in enumerate(doc.images[:5]):
                has_trans = "✓" if img.transcription else "✗"
                console.print(f"  {i+1}. [{has_trans}] {img.alt_text or img.path[:50]}...")
                if img.transcription:
                    trans_preview = img.transcription[:200].replace("\n", " ")
                    console.print(f"      [dim]{trans_preview}...[/dim]")

        # Show content preview
        console.print("\n[bold]Content preview:[/bold]")
        content_preview = doc.content[:500].replace("\n", "\n")
        console.print(Panel(content_preview, border_style="dim"))
        console.print("\n" + "─" * 80 + "\n")


def _display_chunks(items: list, indices: list[int], show_images: bool, show_code: bool):
    """Display chunk items."""
    for idx, chunk_item in zip(indices, items):
        # Header
        multimodal = "🖼️ Multimodal" if chunk_item.is_multimodal else "📝 Text-only"
        console.print(Panel(f"[bold]Chunk {idx}[/bold] ({chunk_item.chunk_id}) - {multimodal}"))
        console.print(f"[cyan]Source:[/cyan] {chunk_item.source_path.name}")
        console.print(
            f"[cyan]Length:[/cyan] {len(chunk_item.text):,} chars (~{chunk_item.token_estimate} tokens)"
        )
        console.print(f"[cyan]Code blocks:[/cyan] {len(chunk_item.code_blocks)}")
        console.print(
            f"[cyan]Images:[/cyan] {len(chunk_item.images)} ({len(chunk_item.transcribed_images)} transcribed)"
        )

        # Images with transcriptions
        if show_images and chunk_item.images:
            console.print("\n[bold]Images in chunk:[/bold]")
            for i, img in enumerate(chunk_item.images):
                status = "[green]✓[/green]" if img.transcription else "[red]✗[/red]"
                console.print(f"  {i+1}. {status} {img.alt_text or 'Image'}")
                if img.transcription:
                    trans_preview = img.transcription[:300].replace("\n", " ")
                    console.print(f"      [dim]{trans_preview}...[/dim]")
                if img.resolved_path:
                    console.print(f"      [dim]Path: {img.resolved_path}[/dim]")

        # Check for embedded [Visual:] blocks
        visual_count = chunk_item.text.count("[Visual:")
        if visual_count > 0:
            console.print(f"\n[green]✓ {visual_count} embedded [Visual:] block(s) found[/green]")

        # Code blocks
        if show_code and chunk_item.code_blocks:
            console.print("\n[bold]Code blocks:[/bold]")
            for i, code in enumerate(chunk_item.code_blocks[:2]):
                code_preview = code[:300] + "..." if len(code) > 300 else code
                console.print(Syntax(code_preview, "python", theme="monokai", line_numbers=True))

        # Main text content
        console.print("\n[bold]Chunk text:[/bold]")
        text_preview = chunk_item.text[:1500]
        if len(chunk_item.text) > 1500:
            text_preview += "\n\n[dim]... (truncated)[/dim]"
        console.print(Panel(text_preview, border_style="dim"))

        # Context info
        if chunk_item.previous_chunk_text:
            console.print(
                f"\n[dim]Has previous context: {len(chunk_item.previous_chunk_text)} chars[/dim]"
            )
        if chunk_item.next_chunk_text:
            console.print(f"[dim]Has next context: {len(chunk_item.next_chunk_text)} chars[/dim]")

        console.print("\n" + "─" * 80 + "\n")


def _display_samples(items: list, indices: list[int], show_code: bool):
    """Display sample items."""
    for idx, sample in zip(indices, items):
        # Header
        multimodal = "🖼️" if sample.image_path else "📝"
        has_test = "✓ Test" if sample.test_code else "✗ No test"
        console.print(
            Panel(f"[bold]Sample {idx}[/bold] - {multimodal} {sample.question_type} [{has_test}]")
        )
        console.print(f"[cyan]Category:[/cyan] {sample.category or 'uncategorized'}")
        console.print(f"[cyan]Source:[/cyan] {sample.source_path}")
        if sample.entry_point:
            console.print(f"[cyan]Entry point:[/cyan] {sample.entry_point}")
        if sample.image_path:
            console.print(f"[cyan]Image:[/cyan] {sample.image_path}")

        # Question
        console.print("\n[bold]Question:[/bold]")
        console.print(Panel(sample.question[:1500], border_style="blue"))

        # Answer
        console.print("\n[bold]Answer:[/bold]")
        answer_preview = sample.answer[:1500]
        if len(sample.answer) > 1500:
            answer_preview += "\n\n[dim]... (truncated)[/dim]"
        if "```" in sample.answer:
            console.print(Panel(answer_preview, border_style="green"))
        else:
            console.print(Panel(answer_preview, border_style="green"))

        # Test code
        if show_code and sample.test_code:
            console.print("\n[bold]Test code:[/bold]")
            test_preview = sample.test_code[:800]
            console.print(Syntax(test_preview, "python", theme="monokai", line_numbers=True))

        console.print("\n" + "─" * 80 + "\n")


def _export_items_to_jsonl(data: list, output_path: Path, stage: str):
    """Export items to JSONL format for external analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(data):
            if stage in ("documents", "transcribed"):
                record = {
                    "index": i,
                    "source_path": str(item.source_path),
                    "title": item.title,
                    "content_length": len(item.content),
                    "code_blocks_count": len(item.code_blocks),
                    "images_count": len(item.images),
                    "images": [
                        {
                            "alt_text": img.alt_text,
                            "has_transcription": bool(img.transcription),
                            "transcription_preview": (
                                img.transcription[:500] if img.transcription else None
                            ),
                        }
                        for img in item.images
                    ],
                    "content_preview": item.content[:2000],
                }
            elif stage in ("chunks", "filtered"):
                record = {
                    "index": i,
                    "chunk_id": item.chunk_id,
                    "source_path": str(item.source_path),
                    "text_length": len(item.text),
                    "is_multimodal": item.is_multimodal,
                    "code_blocks_count": len(item.code_blocks),
                    "images_count": len(item.images),
                    "transcribed_images_count": len(item.transcribed_images),
                    "has_visual_blocks": "[Visual:" in item.text,
                    "text": item.text,
                    "images": [
                        {
                            "alt_text": img.alt_text,
                            "transcription": img.transcription,
                            "resolved_path": img.resolved_path,
                        }
                        for img in item.images
                    ],
                }
            elif stage == "samples":
                record = {
                    "index": i,
                    "question_type": item.question_type,
                    "category": item.category,
                    "question": item.question,
                    "answer": item.answer,
                    "test_code": item.test_code,
                    "entry_point": item.entry_point,
                    "image_path": item.image_path,
                    "source_path": item.source_path,
                }
            else:
                record = {"index": i, "data": str(item)}

            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def pipeline(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
    hub_id: Optional[str] = typer.Option(None, "--hub-id", "-h", help="HuggingFace Hub ID"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="HuggingFace token"),
):
    """Run the complete pipeline: parse -> transcribe -> chunk -> filter -> generate -> build -> export."""
    console.print(
        Panel.fit(
            "[bold cyan]Complete Pipeline[/bold cyan]\nRunning all steps sequentially",
            title="Full Pipeline",
        )
    )

    import time

    start_time = time.time()

    steps = [
        ("Parse", lambda: parse(config_path, no_cache, False)),
        ("Transcribe", lambda: transcribe(config_path, no_cache)),
        ("Chunk", lambda: chunk(config_path, no_cache)),
        ("Filter", lambda: filter_quality(config_path, no_cache)),
        ("Generate", lambda: generate(config_path, no_cache, enable_tracing=True)),
        ("Build", lambda: build(config_path)),
        ("Export", lambda: export(config_path, hub_id, token)),
    ]

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


def inspect_traces(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    stage: str = typer.Option(
        None,
        "--stage",
        "-s",
        help="Filter by stage: input_generation, test_generation, answer_generation",
    ),
    count: int = typer.Option(5, "--count", "-n", help="Number of conversations to display"),
    index: Optional[int] = typer.Option(None, "--index", "-i", help="Specific conversation index"),
    show_full: bool = typer.Option(
        False, "--full", "-f", help="Show full content without truncation"
    ),
    success_only: bool = typer.Option(
        False, "--success", help="Only show successful conversations"
    ),
    failed_only: bool = typer.Option(False, "--failed", help="Only show failed conversations"),
):
    """Inspect generation traces for debugging prompts and responses."""
    console.print(
        Panel.fit(
            "[bold cyan]Trace Inspector[/bold cyan]\n"
            "View detailed prompts and responses from generation",
            title="Inspect Traces",
        )
    )

    config = PipelineConfig.from_yaml(config_path)
    output_dir = Path(config.dataset.generated_dir)
    trace_dir = output_dir / "traces"

    if not trace_dir.exists():
        console.print("[red]✗ No traces found. Run 'generate' with --trace first.[/red]")
        raise typer.Exit(1)

    conversations_file = trace_dir / "conversations.jsonl"
    if not conversations_file.exists():
        console.print("[red]✗ No conversations file found.[/red]")
        raise typer.Exit(1)

    # Load all conversations
    conversations = []
    with open(conversations_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                conv = json.loads(line)
                conversations.append(conv)

    console.print(f"[cyan]Loaded {len(conversations)} conversations[/cyan]")

    # Filter by stage
    if stage:
        conversations = [c for c in conversations if c.get("stage") == stage]
        console.print(
            f"[cyan]Filtered to {len(conversations)} conversations for stage: {stage}[/cyan]"
        )

    # Filter by success/failure
    if success_only:
        conversations = [c for c in conversations if c.get("success")]
        console.print(f"[cyan]Filtered to {len(conversations)} successful conversations[/cyan]")
    elif failed_only:
        conversations = [c for c in conversations if not c.get("success")]
        console.print(f"[cyan]Filtered to {len(conversations)} failed conversations[/cyan]")

    if not conversations:
        console.print("[yellow]No conversations match the criteria.[/yellow]")
        return

    # Show summary first
    summary_table = Table(title="Conversations by Stage")
    summary_table.add_column("Stage", style="cyan")
    summary_table.add_column("Total", style="yellow", justify="right")
    summary_table.add_column("Success", style="green", justify="right")
    summary_table.add_column("Failed", style="red", justify="right")
    summary_table.add_column("Avg Iterations", style="magenta", justify="right")

    stage_stats = {}
    for conv in conversations:
        s = conv.get("stage", "unknown")
        if s not in stage_stats:
            stage_stats[s] = {"total": 0, "success": 0, "iterations": 0}
        stage_stats[s]["total"] += 1
        stage_stats[s]["iterations"] += conv.get("total_iterations", 0)
        if conv.get("success"):
            stage_stats[s]["success"] += 1

    for s, stats in sorted(stage_stats.items()):
        avg_iter = stats["iterations"] / stats["total"] if stats["total"] > 0 else 0
        summary_table.add_row(
            s,
            str(stats["total"]),
            str(stats["success"]),
            str(stats["total"] - stats["success"]),
            f"{avg_iter:.1f}",
        )

    console.print("\n")
    console.print(summary_table)
    console.print("\n")

    # Select conversations to display
    if index is not None:
        if 0 <= index < len(conversations):
            selected = [conversations[index]]
        else:
            console.print(f"[red]Index {index} out of range (0-{len(conversations)-1})[/red]")
            return
    else:
        selected = conversations[:count]

    # Display selected conversations
    for i, conv in enumerate(selected):
        conv_id = conv.get("id", f"conv_{i}")
        stage_name = conv.get("stage", "unknown")
        qtype = conv.get("question_type", "")
        success = "✓" if conv.get("success") else "✗"
        iterations = conv.get("total_iterations", 0)

        header = f"[bold]{conv_id}[/bold] | {stage_name} | {qtype} | {success} | {iterations} iter"
        console.print(Panel(header, border_style="blue" if conv.get("success") else "red"))

        messages = conv.get("messages", [])
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            iteration = msg.get("iteration", 0)
            step = msg.get("step", "")

            role_color = {"system": "yellow", "user": "cyan", "assistant": "green"}.get(
                role, "white"
            )

            # Truncate content for display unless --full is specified
            if not show_full and len(content) > 2000:
                display_content = (
                    content[:2000] + f"\n\n... (truncated, {len(content)} chars total)"
                )
            else:
                display_content = content

            step_info = f" [{step}]" if step else ""
            iter_info = f" (iter={iteration})" if iteration > 0 else ""
            console.print(
                f"\n[{role_color}]━━━ {role.upper()}{step_info}{iter_info} ━━━[/{role_color}]"
            )
            console.print(display_content)

        # Show metadata if present
        metadata = conv.get("metadata", {})
        if metadata:
            console.print("\n[dim]Metadata:[/dim]")
            for key, value in metadata.items():
                if isinstance(value, str) and len(value) > 200 and not show_full:
                    value = value[:200] + "..."
                console.print(f"  [dim]{key}: {value}[/dim]")

        console.print("\n" + "═" * 80 + "\n")

    console.print(f"[dim]Showing {len(selected)} of {len(conversations)} conversations[/dim]")
    console.print("[dim]Use --index N to view a specific conversation[/dim]")
    console.print("[dim]Use --full to see complete content without truncation[/dim]")
