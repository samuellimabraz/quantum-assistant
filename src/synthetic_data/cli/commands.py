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
from synthetic_data.dataset import (
    DatasetAnalyzer,
    DatasetBuilder,
    DatasetPlotter,
    HuggingFaceExporter,
)
from synthetic_data.extractors import ContentChunker, DocumentIngestion, ImageTranscriber
from synthetic_data.models import ModelRegistry
from synthetic_data.parsers.base import Document
from synthetic_data.utils import (
    CheckpointManager,
    ImageQualityFilter,
    PipelineCache,
    QualityFilter,
)

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

    # Calculate detailed statistics by source type
    by_type = {}
    for doc in all_documents:
        doc_type = doc.source_path.suffix.lower()
        if doc_type not in by_type:
            by_type[doc_type] = {
                "count": 0,
                "with_code": 0,
                "with_images": 0,
                "total_code_blocks": 0,
                "total_images": 0,
            }

        by_type[doc_type]["count"] += 1
        if doc.code_blocks:
            by_type[doc_type]["with_code"] += 1
        if doc.images:
            by_type[doc_type]["with_images"] += 1
        by_type[doc_type]["total_code_blocks"] += len(doc.code_blocks)
        by_type[doc_type]["total_images"] += len(doc.images)

    total_images = sum(len(doc.images) for doc in all_documents)

    summary = {
        "total_documents": len(all_documents),
        "total_images": total_images,
        "total_code_blocks": sum(len(doc.code_blocks) for doc in all_documents),
        "by_source_type": by_type,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table = Table(title="Parsing Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_row("Documents", str(len(all_documents)))
    table.add_row("Code Blocks", str(summary["total_code_blocks"]))
    table.add_row("Images", str(total_images))

    console.print("\n")
    console.print(table)

    # Source type breakdown
    if by_type:
        type_table = Table(title="By Source Type")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Docs", style="green", justify="right")
        type_table.add_column("Code", style="yellow", justify="right")
        type_table.add_column("Images", style="magenta", justify="right")

        for doc_type, stats in sorted(by_type.items()):
            type_table.add_row(
                doc_type,
                str(stats["count"]),
                str(stats["total_code_blocks"]),
                str(stats["total_images"]),
            )

        console.print("\n")
        console.print(type_table)

    console.print(f"\n[green]✓ Saved to: {output_dir}[/green]")


def _associate_code_context_with_images(documents: list) -> int:
    """Associate code blocks that PRECEDE images to provide context for transcription.

    Only code blocks appearing BEFORE an image in the document are associated.
    This ensures the VLM understands what code generated the image output.

    Returns the number of images that received code context.
    """
    import re

    code_context_count = 0
    code_pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

    for doc in documents:
        if not doc.images:
            continue

        content = doc.content

        for img in doc.images:
            if not img.image_id:
                continue

            # Find the image marker position in content
            marker = f"[IMAGE:{img.image_id}]"
            marker_pos = content.find(marker)

            if marker_pos == -1:
                continue

            # Only look at content BEFORE this image
            content_before = content[:marker_pos]

            # Find all code blocks in content before the image
            code_matches = list(code_pattern.finditer(content_before))

            if not code_matches:
                continue

            # Use the closest (last) code block before the image
            last_code = code_matches[-1].group(1).strip()
            if last_code and len(last_code) > 20:
                img.code_context = last_code[:2000]
                code_context_count += 1

    return code_context_count


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

    # Associate code context with images before transcription
    code_context_count = _associate_code_context_with_images(all_documents)
    if code_context_count > 0:
        console.print(f"[cyan]Associated code context with {code_context_count} images[/cyan]")

    output_dir = Path(config.dataset.parsed_dir).parent / "transcribed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "documents.pkl"

    checkpoint_dir = output_dir.parent / ".checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir, "transcribe")

    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Transcriptions already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-transcribe[/yellow]")
        return

    transcribed_image_ids = set()
    if not no_cache and checkpoint_manager.exists():
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            all_documents, metadata = checkpoint_data
            transcribed_image_ids = set(metadata.get("transcribed_image_ids", []))
            if transcribed_image_ids:
                console.print(
                    f"[cyan]Resuming from checkpoint: "
                    f"{len(transcribed_image_ids)} images already transcribed[/cyan]"
                )

    images_to_transcribe = sum(
        1
        for doc in all_documents
        for img in doc.images
        if img.resolved_path and img.image_id not in transcribed_image_ids
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
            current_transcribed = {
                img.image_id
                for doc in documents
                for img in doc.images
                if img.image_id and img.transcription
            }
            metadata = {
                "transcribed_image_ids": list(current_transcribed),
                "total_images": len(current_transcribed),
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
                        skip_image_ids=transcribed_image_ids,
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

    # Count by type
    type_counts = {}
    for img in transcribed_images:
        type_name = img.image_type.value if img.image_type else "unknown"
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    classified_count = sum(1 for img in transcribed_images if img.image_type)

    # Summary table
    summary_table = Table(title="Transcription Complete")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="green", justify="right")
    summary_table.add_row("Total Images", str(len(all_images)))
    summary_table.add_row("Transcribed", str(len(transcribed_images)))
    summary_table.add_row("Classified", str(classified_count))

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
        "classified": classified_count,
        "type_distribution": type_counts,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[green]✓ Saved to: {output_dir}[/green]")


def filter_images(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
):
    """Step 3: Filter images for quality after transcription."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 3: Image Quality Filtering[/bold cyan]\n"
            "Filter low-quality images and remove from content",
            title="Filter Images",
        )
    )

    config = PipelineConfig.from_yaml(config_path)

    if not config.generation.enable_content_filtering:
        console.print("[yellow]Content filtering disabled in config, skipping[/yellow]")

        transcribed_dir = Path(config.dataset.parsed_dir).parent / "transcribed"
        output_dir = Path(config.dataset.parsed_dir).parent / "filtered_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        if (transcribed_dir / "documents.pkl").exists():
            import shutil

            shutil.copy2(transcribed_dir / "documents.pkl", output_dir / "documents.pkl")
            console.print(f"[green]✓ Copied unfiltered documents to: {output_dir}[/green]")
        return

    transcribed_dir = Path(config.dataset.parsed_dir).parent / "transcribed"
    input_file = transcribed_dir / "documents.pkl"

    if not input_file.exists():
        console.print("[red]✗ No transcribed documents found. Run 'transcribe' first.[/red]")
        raise typer.Exit(1)

    with open(input_file, "rb") as f:
        all_documents = pickle.load(f)

    console.print(f"[cyan]Loaded {len(all_documents)} documents[/cyan]")

    output_dir = Path(config.dataset.parsed_dir).parent / "filtered_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "documents.pkl"

    # Setup checkpoint manager
    checkpoint_dir = output_dir.parent / ".checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir, "filter_images")

    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Filtered images already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-filter[/yellow]")
        return

    # Try to resume from checkpoint
    processed_image_ids = set()
    if not no_cache and checkpoint_manager.exists():
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            checkpoint_docs, metadata = checkpoint_data
            processed_image_ids = set(metadata.get("processed_image_ids", []))
            if processed_image_ids:
                console.print(
                    f"[cyan]Resuming from checkpoint: "
                    f"{len(processed_image_ids)} images already filtered[/cyan]"
                )
                all_documents = checkpoint_docs

    # Count images to filter (excluding already processed)
    images_to_filter = sum(
        1
        for doc in all_documents
        for img in doc.images
        if img.transcription and img.resolved_path and img.image_id not in processed_image_ids
    )

    if images_to_filter == 0:
        console.print("[yellow]No images to filter[/yellow]")
        with open(output_file, "wb") as f:
            pickle.dump(all_documents, f)
        checkpoint_manager.clear_checkpoint()
        return

    console.print(f"[cyan]Found {images_to_filter} images to filter[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Filtering images...", total=images_to_filter)

        def update_progress(completed_images):
            progress.update(task, completed=completed_images)

        def save_checkpoint(documents, debug_info):
            # Track which images have been processed
            current_processed = {
                img.image_id
                for doc in documents
                for img in doc.images
                if img.image_id and img.transcription and img.resolved_path
            }
            metadata = {
                "processed_image_ids": list(current_processed),
                "total_images": len(current_processed),
                "debug_entries": len(debug_info),
            }
            checkpoint_manager.save_checkpoint(documents, metadata)

        with ModelRegistry(config.models) as model_registry:
            filter_model_name = config.generation.filter_model or config.generation.curate_model
            filter_client = model_registry.get_llm_client(filter_model_name)
            image_filter = ImageQualityFilter(
                filter_client, max_concurrent=config.generation.llm_concurrency
            )

            try:
                filtered_documents, debug_info = asyncio.run(
                    image_filter.filter_documents_batch_async(
                        all_documents,
                        config.prompts.image_quality_check,
                        config.prompts.image_filter_system,
                        progress_callback=update_progress,
                        checkpoint_callback=save_checkpoint,
                        checkpoint_interval=config.generation.llm_batch_size,
                        skip_image_ids=processed_image_ids,
                    )
                )
            except Exception as e:
                console.print(f"[red]✗ Image filtering failed: {e}[/red]")
                console.print("[yellow]Progress has been saved to checkpoint[/yellow]")
                raise

    # Save filtered documents
    with open(output_file, "wb") as f:
        pickle.dump(filtered_documents, f)

    # Clear checkpoint after successful completion
    checkpoint_manager.clear_checkpoint()

    # Save debug info
    debug_file = output_dir / "image_filter_decisions.jsonl"
    with open(debug_file, "w", encoding="utf-8") as f:
        for entry in debug_info:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    # Calculate statistics
    images_before = sum(len(doc.images) for doc in all_documents)
    images_after = sum(len(doc.images) for doc in filtered_documents)
    images_removed = images_before - images_after

    # Get summary from debug info
    summary_entry = next((e for e in debug_info if e.get("type") == "summary"), {})

    summary = {
        "documents": len(all_documents),
        "images_before": images_before,
        "images_after": images_after,
        "images_removed": images_removed,
        **summary_entry,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table = Table(title="Image Filtering Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Before", style="yellow", justify="right")
    table.add_column("After", style="green", justify="right")
    table.add_column("Removed", style="red", justify="right")
    table.add_row(
        "Images",
        str(images_before),
        str(images_after),
        str(images_removed),
    )

    console.print("\n")
    console.print(table)
    console.print(f"\n[green]✓ Saved to: {output_dir}[/green]")
    console.print(f"[dim]  Filter decisions: {debug_file.name}[/dim]")


def chunk(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
):
    """Step 4: Chunk documents into manageable pieces."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 4: Content Chunking[/bold cyan]\n"
            "Split documents into context-sized chunks",
            title="Chunk",
        )
    )

    config = PipelineConfig.from_yaml(config_path)

    # Try to use filtered_images first, then transcribed, then parsed
    filtered_images_dir = Path(config.dataset.parsed_dir).parent / "filtered_images"
    transcribed_dir = Path(config.dataset.parsed_dir).parent / "transcribed"
    parsed_dir = Path(config.dataset.parsed_dir)

    if (filtered_images_dir / "documents.pkl").exists():
        input_file = filtered_images_dir / "documents.pkl"
        console.print("[cyan]Loading filtered image documents[/cyan]")
    elif (transcribed_dir / "documents.pkl").exists():
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
        min_length=config.generation.min_chunk_length,
        overlap=config.generation.chunk_overlap,
        max_code_blocks_per_chunk=config.generation.max_code_blocks_per_chunk,
        max_images_per_chunk=config.generation.max_images_per_chunk,
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

    # Count unique images by image_id (deduplicated)
    unique_images_in_chunks = {img.image_id: img for img in all_images_in_chunks if img.image_id}
    images_with_transcription = sum(
        1 for img in unique_images_in_chunks.values() if img.transcription
    )

    # Breakdown by source type
    by_type = {}
    for chunk in all_chunks:
        doc_type = chunk.source_path.suffix.lower()
        if doc_type not in by_type:
            by_type[doc_type] = {
                "chunks": 0,
                "with_code": 0,
                "with_images": 0,
                "multimodal": 0,
            }

        by_type[doc_type]["chunks"] += 1
        if chunk.code_blocks:
            by_type[doc_type]["with_code"] += 1
        if chunk.images:
            by_type[doc_type]["with_images"] += 1
        if chunk.is_multimodal:
            by_type[doc_type]["multimodal"] += 1

    # Diagnostic: Compare with filtered images (source of chunking)
    # Use the same source that was used for chunking
    filtered_images_dir = Path(config.dataset.parsed_dir).parent / "filtered_images"
    source_images_count = 0
    missing_images = []

    # Determine which source was used for chunking
    if (filtered_images_dir / "documents.pkl").exists():
        source_dir = filtered_images_dir
        source_name = "filtered"
    else:
        transcribed_dir = Path(config.dataset.parsed_dir).parent / "transcribed"
        source_dir = transcribed_dir
        source_name = "transcribed"

    if (source_dir / "documents.pkl").exists():
        with open(source_dir / "documents.pkl", "rb") as f:
            source_docs = pickle.load(f)

        all_source_images = [img for doc in source_docs for img in doc.images if img.transcription]
        source_images_count = len(all_source_images)
        missing_images = [
            {
                "image_id": img.image_id,
                "path": img.path,
                "alt_text": img.alt_text,
                "image_type": img.image_type.value if img.image_type else "unknown",
                "source": str(img.path).split(":")[0] if ":" in str(img.path) else "unknown",
            }
            for img in all_source_images
            if img.image_id and img.image_id not in unique_images_in_chunks
        ]

    # Quality breakdown
    from synthetic_data.extractors.chunker import ChunkQuality

    quality_counts = {"high": 0, "medium": 0, "low": 0}
    for c in all_chunks:
        quality_counts[c.quality.value] = quality_counts.get(c.quality.value, 0) + 1

    summary = {
        "total_chunks": len(all_chunks),
        "chunks_with_code": sum(1 for c in all_chunks if c.code_blocks),
        "chunks_with_images": sum(1 for c in all_chunks if c.images),
        "multimodal_chunks": sum(1 for c in all_chunks if c.is_multimodal),
        "avg_chunk_length": (
            sum(len(c.text) for c in all_chunks) / len(all_chunks) if all_chunks else 0
        ),
        "image_references_in_chunks": len(
            all_images_in_chunks
        ),  # Total references (can have duplicates)
        "unique_images_in_chunks": len(unique_images_in_chunks),  # Deduplicated by image_id
        "images_with_transcription": images_with_transcription,
        "source_images_total": source_images_count,
        "source_type": source_name,
        "images_missing_from_chunks": len(missing_images),
        "by_source_type": by_type,
        "quality_distribution": quality_counts,
    }

    # Save diagnostic file for missing images
    if missing_images:
        diagnostic_file = output_dir / "missing_images.jsonl"
        with open(diagnostic_file, "w", encoding="utf-8") as f:
            for img_info in missing_images:
                json.dump(img_info, f, ensure_ascii=False)
                f.write("\n")
        console.print(
            f"[yellow]⚠ {len(missing_images)} {source_name} images not found in chunks (saved to {diagnostic_file.name})[/yellow]"
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
    table.add_row(
        "Image References",
        str(summary["image_references_in_chunks"]) + " [dim](with duplicates)[/dim]",
    )
    table.add_row(
        "Unique Images", str(summary["unique_images_in_chunks"]) + " [dim](deduplicated)[/dim]"
    )
    if source_images_count > 0:
        table.add_row(
            f"Source ({source_name})",
            f"{source_images_count} [dim](total)[/dim] / {summary['images_with_transcription']} [dim](in chunks)[/dim]",
        )
        if missing_images:
            table.add_row(
                "Missing from Chunks",
                f"[red]{len(missing_images)}[/red]",
            )
    table.add_row("Avg Length", f"{summary['avg_chunk_length']:.0f} chars")

    console.print("\n")
    console.print(table)

    # Source type breakdown
    if by_type:
        type_table = Table(title="By Source Type")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Chunks", style="green", justify="right")
        type_table.add_column("Code", style="yellow", justify="right")
        type_table.add_column("Images", style="magenta", justify="right")
        type_table.add_column("Multimodal", style="blue", justify="right")

        for doc_type, stats in sorted(by_type.items()):
            type_table.add_row(
                doc_type,
                str(stats["chunks"]),
                str(stats["with_code"]),
                str(stats["with_images"]),
                str(stats["multimodal"]),
            )

        console.print("\n")
        console.print(type_table)

    # Quality distribution table
    if quality_counts:
        quality_table = Table(title="Chunk Quality Distribution")
        quality_table.add_column("Quality", style="cyan")
        quality_table.add_column("Count", style="green", justify="right")
        quality_table.add_column("Percentage", style="magenta", justify="right")

        total = len(all_chunks)
        for quality, count in [
            ("high", quality_counts.get("high", 0)),
            ("medium", quality_counts.get("medium", 0)),
            ("low", quality_counts.get("low", 0)),
        ]:
            pct = (count / total * 100) if total > 0 else 0
            style = {"high": "green", "medium": "yellow", "low": "red"}.get(quality, "white")
            quality_table.add_row(f"[{style}]{quality}[/{style}]", str(count), f"{pct:.1f}%")

        console.print("\n")
        console.print(quality_table)

    console.print(f"[green]✓ Saved to: {output_dir}[/green]")


def filter_quality(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
):
    """Step 5: Filter chunks for quality."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 5: Chunk Quality Filtering[/bold cyan]\n"
            "Filter low-quality content chunks",
            title="Filter Chunks",
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

    output_dir = Path(config.dataset.parsed_dir).parent / "filtered"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks.pkl"

    # Setup checkpoint manager
    checkpoint_dir = output_dir.parent / ".checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir, "filter_chunks")

    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Filtered chunks already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-filter[/yellow]")
        return

    # Try to resume from checkpoint
    processed_chunk_ids = set()
    filter_results_cache = {}
    if not no_cache and checkpoint_manager.exists():
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            results_data, metadata = checkpoint_data
            processed_chunk_ids = set(metadata.get("processed_chunk_ids", []))
            # Restore filter results for processed chunks
            filter_results_cache = metadata.get("filter_results", {})
            if processed_chunk_ids:
                console.print(
                    f"[cyan]Resuming from checkpoint: "
                    f"{len(processed_chunk_ids)} chunks already filtered[/cyan]"
                )

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

        def save_checkpoint(results, debug_info):
            # Track which chunks have been processed
            current_processed = {chunk.chunk_id for chunk, _ in results}
            # Store filter decisions for processed chunks
            filter_decisions = {chunk.chunk_id: passed for chunk, passed in results}
            metadata = {
                "processed_chunk_ids": list(current_processed),
                "filter_results": filter_decisions,
                "total_chunks": len(current_processed),
            }
            checkpoint_manager.save_checkpoint(results, metadata)

        with ModelRegistry(config.models) as model_registry:
            filter_model_name = config.generation.filter_model or config.generation.curate_model
            filter_client = model_registry.get_llm_client(filter_model_name)
            quality_filter = QualityFilter(filter_client)

            try:
                filter_results, debug_info = asyncio.run(
                    quality_filter.filter_chunks_batch_async(
                        all_chunks,
                        config.prompts.content_quality_check,
                        config.prompts.content_filter_system,
                        batch_size=config.generation.llm_batch_size,
                        max_concurrent=config.generation.llm_concurrency,
                        progress_callback=update_progress,
                        checkpoint_callback=save_checkpoint,
                        checkpoint_interval=config.generation.llm_batch_size,
                        skip_chunk_ids=processed_chunk_ids,
                        cached_results=filter_results_cache,
                        save_debug=True,
                    )
                )
            except Exception as e:
                console.print(f"[red]✗ Filtering failed: {e}[/red]")
                console.print("[yellow]Progress has been saved to checkpoint[/yellow]")
                raise

    filtered_chunks = [chunk for chunk, passed in filter_results if passed]

    # Clear checkpoint after successful completion
    checkpoint_manager.clear_checkpoint()

    # Save detailed filter decisions
    debug_file = output_dir / "filter_decisions.jsonl"
    with open(debug_file, "w", encoding="utf-8") as f:
        for entry in debug_info:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    if debug_info:
        console.print(f"[dim]Saved {len(debug_info)} filter decisions to {debug_file.name}[/dim]")

    # Calculate statistics
    chunks_before = len(all_chunks)
    chunks_after = len(filtered_chunks)
    chunks_removed = chunks_before - chunks_after

    # Count images in chunks (for informational purposes)
    images_before = sum(len(c.images) for c in all_chunks)
    images_after = sum(len(c.images) for c in filtered_chunks)

    # Count unique images before and after (deduplicated by image_id)
    unique_images_before = set()
    for ch in all_chunks:
        for img in ch.images:
            if img.image_id:
                unique_images_before.add(img.image_id)

    unique_images_after = set()
    for ch in filtered_chunks:
        for img in ch.images:
            if img.image_id:
                unique_images_after.add(img.image_id)

    # Count multimodal chunks before and after
    multimodal_before = sum(1 for c in all_chunks if c.is_multimodal)
    multimodal_after = sum(1 for c in filtered_chunks if c.is_multimodal)

    # Count content filter decisions
    content_passes = sum(
        1 for d in debug_info if d.get("type") == "content" and d.get("decision") == "PASS"
    )
    content_rejects = sum(
        1 for d in debug_info if d.get("type") == "content" and d.get("decision") == "REJECT"
    )

    with open(output_file, "wb") as f:
        pickle.dump(filtered_chunks, f)

    summary = {
        "chunks_before": chunks_before,
        "chunks_after": chunks_after,
        "chunks_removed": chunks_removed,
        "images_in_chunks_before": images_before,
        "images_in_chunks_after": images_after,
        "unique_images_before": len(unique_images_before),
        "unique_images_after": len(unique_images_after),
        "multimodal_before": multimodal_before,
        "multimodal_after": multimodal_after,
        "content_decisions": {
            "passed": content_passes,
            "rejected": content_rejects,
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
        str(chunks_before),
        str(chunks_after),
        str(chunks_removed),
    )
    table.add_row(
        "Images (in chunks)",
        str(images_before),
        str(images_after),
        str(images_before - images_after),
    )
    table.add_row(
        "Unique Images",
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

    # Content filter decision breakdown
    if content_passes > 0 or content_rejects > 0:
        decision_table = Table(title="Content Filter Decisions")
        decision_table.add_column("Type", style="cyan")
        decision_table.add_column("Passed", style="green", justify="right")
        decision_table.add_column("Rejected", style="red", justify="right")
        decision_table.add_row(
            "Chunks",
            str(content_passes),
            str(content_rejects),
        )

        console.print("\n")
        console.print(decision_table)

    console.print(f"\n[green]✓ Saved to: {output_dir}[/green]")
    if debug_info:
        console.print(f"[dim]  Filter decisions: {debug_file.name}[/dim]")


# NOTE: generate function moved to generation_commands.py
# (see synthetic_data/cli/generation_commands.py for plan, filter-candidates, answer, curate, classify commands)


def build(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
):
    """Step 7: Build train/val/test splits."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 7: Dataset Splits[/bold cyan]\n"
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
    with_analysis: bool = typer.Option(
        False, "--analyze", "-a", help="Run analysis and upload with dataset"
    ),
):
    """Step 8: Export dataset to HuggingFace format."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 8: Export Dataset[/bold cyan]\n"
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

    # Run analysis if requested
    analysis_dir = None
    if with_analysis:
        console.print("\n[cyan]Running dataset analysis...[/cyan]")
        base_dir = Path(config.dataset.parsed_dir).parent
        analysis_dir = base_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        analyzer = DatasetAnalyzer()
        analyzer.load_from_splits(train_samples, val_samples, test_samples)
        stats = analyzer.analyze()
        analyzer.save_statistics(analysis_dir / "statistics.json")

        plotter = DatasetPlotter(stats, output_dir=analysis_dir)
        plotter.set_samples(analyzer.samples)
        plot_paths = plotter.plot_all()
        console.print(f"[green]✓ Generated analysis with {len(plot_paths)} plots[/green]")

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

        # Push analysis if generated
        if analysis_dir and analysis_dir.exists():
            _push_analysis_to_hub(analysis_dir, hub_id, token, console)

    table = Table(title="Export Complete")
    table.add_column("Split", style="cyan")
    table.add_column("Samples", style="green", justify="right")

    for split in dataset_dict.keys():
        table.add_row(split, str(len(dataset_dict[split])))

    console.print("\n")
    console.print(table)


def analyze(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    source: str = typer.Option(
        "splits", "--source", "-s", help="Source: splits, generated, or final (HuggingFace)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for analysis"
    ),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip plot generation"),
    include_allocation: bool = typer.Option(
        True, "--allocation/--no-allocation", help="Include allocation analysis plots"
    ),
    include_pipeline: bool = typer.Option(
        True, "--pipeline/--no-pipeline", help="Include pipeline stage analysis plots"
    ),
    hub_id: Optional[str] = typer.Option(
        None, "--hub-id", "-h", help="Push analysis to HuggingFace Hub"
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="HuggingFace token"),
):
    """Analyze dataset distributions and generate visualizations."""
    console.print(
        Panel.fit(
            "[bold cyan]Dataset Analysis[/bold cyan]\n"
            "Analyze distributions and generate Qiskit-styled plots",
            title="Analyze",
        )
    )

    config = PipelineConfig.from_yaml(config_path)
    base_dir = Path(config.dataset.parsed_dir).parent

    analyzer = DatasetAnalyzer()

    if source == "splits":
        splits_dir = base_dir / "splits"
        if not (splits_dir / "train.pkl").exists():
            console.print("[red]✗ No splits found. Run 'build' first.[/red]")
            raise typer.Exit(1)
        console.print(f"[cyan]Loading from: {splits_dir}[/cyan]")
        analyzer.load_from_pickle(splits_dir)

    elif source == "generated":
        generated_dir = Path(config.dataset.generated_dir)
        samples_file = generated_dir / "samples.pkl"
        if not samples_file.exists():
            console.print("[red]✗ No samples found. Run 'generate' first.[/red]")
            raise typer.Exit(1)
        console.print(f"[cyan]Loading from: {samples_file}[/cyan]")
        with open(samples_file, "rb") as f:
            samples = pickle.load(f)
        analyzer.load_from_splits(samples, [], [])

    elif source == "final":
        final_dir = Path(config.dataset.final_dir)
        if not final_dir.exists():
            console.print("[red]✗ No final dataset found. Run 'export' first.[/red]")
            raise typer.Exit(1)
        console.print(f"[cyan]Loading from: {final_dir}[/cyan]")
        analyzer.load_from_huggingface(final_dir)

    else:
        console.print(f"[red]✗ Unknown source: {source}[/red]")
        console.print("[yellow]Available: splits, generated, final[/yellow]")
        raise typer.Exit(1)

    console.print("[cyan]Computing statistics...[/cyan]")
    stats = analyzer.analyze()

    if output_dir is None:
        output_dir = base_dir / "analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = analyzer.save_statistics(output_dir / "statistics.json")
    console.print(f"[green]✓ Statistics saved to: {stats_path}[/green]")

    plot_paths = []
    if not no_plots:
        console.print("[cyan]Generating dataset plots...[/cyan]")
        plotter = DatasetPlotter(stats, output_dir=output_dir)
        plotter.set_samples(analyzer.samples)
        plot_paths = plotter.plot_all()
        console.print(f"[green]✓ Generated {len(plot_paths)} dataset plots[/green]")

    # Pipeline stage analysis (source, images, chunks)
    pipeline_paths = []
    if include_pipeline and not no_plots:
        console.print("[cyan]Generating pipeline stage analysis plots...[/cyan]")

        from synthetic_data.tools.pipeline_analyzer import PipelineAnalyzer
        from synthetic_data.tools.pipeline_plotter import PipelinePlotter

        pipeline_analyzer = PipelineAnalyzer(base_dir)
        pipeline_stats = pipeline_analyzer.analyze()

        if pipeline_stats.source.total_files > 0 or pipeline_stats.chunks.total_chunks > 0:
            pipeline_plotter = PipelinePlotter(pipeline_stats, output_dir)
            pipeline_paths = pipeline_plotter.plot_all()
            console.print(f"[green]✓ Generated {len(pipeline_paths)} pipeline plots[/green]")
        else:
            console.print("[yellow]⚠ No pipeline data found for analysis[/yellow]")

    # Allocation analysis plots (diversity and target sweeps)
    allocation_paths = []
    if include_allocation and not no_plots:
        filtered_dir = base_dir / "filtered"
        chunks_file = filtered_dir / "chunks.pkl"

        if chunks_file.exists():
            console.print("[cyan]Generating allocation sweep plots...[/cyan]")

            from synthetic_data.tools.allocation_analyzer import (
                ChunkAnalyzer as AllocChunkAnalyzer,
                AllocationSimulator,
                AllocationPlotter,
                load_chunks,
                get_default_type_configs,
            )
            from synthetic_data.generators.allocation import TypeAllocationConfig as AllocTypeConfig
            from synthetic_data.config import QuestionType

            chunks = load_chunks(chunks_file)

            type_configs = {}
            if config.generation.type_allocations:
                for type_name, type_cfg in config.generation.type_allocations.items():
                    try:
                        qt = QuestionType(type_name)
                        type_configs[qt] = AllocTypeConfig(
                            ratio=type_cfg.ratio,
                            multimodal_ratio=type_cfg.multimodal_ratio,
                        )
                    except ValueError:
                        pass

            if not type_configs:
                type_configs = get_default_type_configs()

            chunk_analyzer = AllocChunkAnalyzer(chunks)
            chunk_analysis = chunk_analyzer.analyze()

            simulator = AllocationSimulator(chunks)
            target = config.generation.target_samples

            diversity_sweep = simulator.sweep_diversity_weight(target, type_configs)
            target_sweep = simulator.sweep_target_samples(type_configs)

            alloc_plotter = AllocationPlotter(output_dir)
            allocation_paths = alloc_plotter.plot_all(chunk_analysis, diversity_sweep, target_sweep)
            console.print(f"[green]✓ Generated {len(allocation_paths)} allocation plots[/green]")
        else:
            console.print("[yellow]⚠ No filtered chunks found, skipping allocation plots[/yellow]")

    _display_analysis_summary(stats, console)

    if hub_id:
        _push_analysis_to_hub(output_dir, hub_id, token, console)

    all_paths = plot_paths + pipeline_paths + allocation_paths
    console.print(f"\n[green]✓ Analysis complete: {output_dir}[/green]")

    if all_paths:
        console.print("\n[dim]Generated files:[/dim]")
        console.print(f"  [dim]• {stats_path.name}[/dim]")
        for path in all_paths:
            console.print(f"  [dim]• {path.name}[/dim]")


def _display_analysis_summary(stats, console):
    """Display analysis summary tables."""
    # Overview table
    overview_table = Table(title="Dataset Overview")
    overview_table.add_column("Metric", style="cyan")
    overview_table.add_column("Value", style="green", justify="right")

    overview_table.add_row("Total Samples", f"{stats.total_samples:,}")

    aggregated = stats.to_dict()["aggregated"]
    overview_table.add_row("Multimodal", f"{aggregated['multimodal']:,}")
    overview_table.add_row("Text-only", f"{aggregated['text_only']:,}")
    overview_table.add_row("With Tests", f"{aggregated['with_tests']:,}")

    multimodal_ratio = (
        aggregated["multimodal"] / stats.total_samples * 100 if stats.total_samples > 0 else 0
    )
    overview_table.add_row("Multimodal %", f"{multimodal_ratio:.1f}%")

    console.print("\n")
    console.print(overview_table)

    # Split distribution table
    split_table = Table(title="Split Distribution")
    split_table.add_column("Split", style="cyan")
    split_table.add_column("Samples", style="green", justify="right")
    split_table.add_column("%", style="magenta", justify="right")
    split_table.add_column("Multimodal", style="yellow", justify="right")
    split_table.add_column("MM %", style="yellow", justify="right")

    for name, split_stats in stats.splits.items():
        pct = split_stats.total / stats.total_samples * 100 if stats.total_samples > 0 else 0
        mm_pct = split_stats.multimodal_ratio * 100
        split_table.add_row(
            name,
            f"{split_stats.total:,}",
            f"{pct:.1f}%",
            f"{split_stats.multimodal:,}",
            f"{mm_pct:.1f}%",
        )

    console.print("\n")
    console.print(split_table)

    # Type distribution table
    type_table = Table(title="Question Type Distribution")
    type_table.add_column("Type", style="cyan")
    type_table.add_column("Count", style="green", justify="right")
    type_table.add_column("%", style="magenta", justify="right")

    for qtype, count in sorted(aggregated["by_type"].items(), key=lambda x: x[1], reverse=True):
        pct = count / stats.total_samples * 100 if stats.total_samples > 0 else 0
        type_table.add_row(qtype, f"{count:,}", f"{pct:.1f}%")

    console.print("\n")
    console.print(type_table)

    # Category distribution table
    if aggregated["by_category"]:
        cat_table = Table(title="Category Distribution")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", style="green", justify="right")
        cat_table.add_column("%", style="magenta", justify="right")

        for cat, count in sorted(
            aggregated["by_category"].items(), key=lambda x: x[1], reverse=True
        ):
            pct = count / stats.total_samples * 100 if stats.total_samples > 0 else 0
            cat_table.add_row(cat, f"{count:,}", f"{pct:.1f}%")

        console.print("\n")
        console.print(cat_table)


def _push_analysis_to_hub(output_dir: Path, hub_id: str, token: Optional[str], console):
    """Push analysis folder to HuggingFace Hub."""
    from huggingface_hub import HfApi

    console.print(f"\n[cyan]Pushing analysis to HuggingFace Hub: {hub_id}[/cyan]")

    api = HfApi(token=token)

    # Upload all files in the analysis directory
    files_uploaded = 0
    for file_path in output_dir.iterdir():
        if file_path.is_file():
            path_in_repo = f"analysis/{file_path.name}"
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=path_in_repo,
                    repo_id=hub_id,
                    repo_type="dataset",
                    token=token,
                )
                files_uploaded += 1
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to upload {file_path.name}: {e}[/yellow]")

    if files_uploaded > 0:
        console.print(f"[green]✓ Uploaded {files_uploaded} files to {hub_id}/analysis/[/green]")
    else:
        console.print("[yellow]No files were uploaded[/yellow]")


def inspect(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    stage: str = typer.Option(
        "chunks",
        "--stage",
        "-s",
        help="Stage to inspect: documents, transcribed, filtered_images, chunks, filtered, samples",
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
        "filtered_images": base_dir / "filtered_images" / "documents.pkl",
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
    if stage in ("documents", "transcribed", "filtered_images"):
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
            if stage in ("documents", "transcribed", "filtered_images"):
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


# NOTE: pipeline function moved to main.py


def analyze_allocation(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    target: int = typer.Option(8000, "--target", "-t", help="Target number of samples"),
):
    """Analyze allocation strategies and generate visualizations."""
    console.print(
        Panel.fit(
            "[bold cyan]Allocation Analysis[/bold cyan]\n"
            "Analyze chunk/image utilization with Qiskit-styled plots",
            title="Analyze Allocation",
        )
    )

    config = PipelineConfig.from_yaml(config_path)

    filtered_dir = Path(config.dataset.parsed_dir).parent / "filtered"
    chunks_file = filtered_dir / "chunks.pkl"

    if not chunks_file.exists():
        console.print("[red]✗ No filtered chunks found. Run 'filter-quality' first.[/red]")
        raise typer.Exit(1)

    from synthetic_data.tools.allocation_analyzer import (
        ChunkAnalyzer,
        AllocationSimulator,
        AllocationPlotter,
        load_chunks,
        get_default_type_configs,
    )
    from synthetic_data.generators.allocation import TypeAllocationConfig as AllocTypeConfig
    from synthetic_data.config import QuestionType

    chunks = load_chunks(chunks_file)
    console.print(f"[cyan]Loaded {len(chunks)} filtered chunks[/cyan]")

    # Build type configs from pipeline config
    type_configs = {}
    if config.generation.type_allocations:
        for type_name, type_cfg in config.generation.type_allocations.items():
            try:
                qt = QuestionType(type_name)
                type_configs[qt] = AllocTypeConfig(
                    ratio=type_cfg.ratio,
                    multimodal_ratio=type_cfg.multimodal_ratio,
                )
            except ValueError:
                pass

    if not type_configs:
        type_configs = get_default_type_configs()

    # Analyze chunks
    analyzer = ChunkAnalyzer(chunks)
    analysis = analyzer.analyze()

    console.print(f"\n[bold]Chunk Analysis:[/bold]")
    console.print(f"  Total: {analysis.total_chunks:,}")
    console.print(
        f"  Multimodal: {analysis.multimodal_chunks:,} ({analysis.multimodal_chunks/analysis.total_chunks*100:.1f}%)"
    )
    console.print(
        f"  With code: {analysis.chunks_with_code:,} ({analysis.chunks_with_code/analysis.total_chunks*100:.1f}%)"
    )
    console.print(f"  Unique images: {analysis.unique_images:,}")

    # Run simulations
    simulator = AllocationSimulator(chunks)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running allocation simulations...", total=None)

        diversity_sweep = simulator.sweep_diversity_weight(target, type_configs)
        target_sweep = simulator.sweep_target_samples(type_configs)

        over_alloc = config.generation.over_allocation_factor
        div_weight = config.generation.diversity_weight
        current_sim = simulator.simulate(
            target,
            type_configs,
            over_allocation_factor=over_alloc,
            diversity_weight=div_weight,
            config_name="current",
        )

        progress.update(task, description="Generating plots...")

    # Generate plots
    output_dir = Path(config.dataset.parsed_dir).parent / "analysis"
    plotter = AllocationPlotter(output_dir)
    plot_paths = plotter.plot_all(analysis, diversity_sweep, target_sweep)

    console.print(f"\n[bold]Current Allocation (target={target}):[/bold]")
    console.print(f"  Allocated: {current_sim.total_allocated:,}")
    console.print(f"  Multimodal: {current_sim.multimodal_allocated:,}")
    console.print(f"  Chunk coverage: {current_sim.chunk_coverage*100:.1f}%")
    console.print(f"  Image coverage: {current_sim.image_coverage*100:.1f}%")

    console.print(f"\n[green]✓ Generated {len(plot_paths)} plots[/green]")
    for path in plot_paths:
        console.print(f"  [dim]• {path.name}[/dim]")
    console.print(f"\n[dim]Plots saved to: {output_dir}[/dim]")


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
