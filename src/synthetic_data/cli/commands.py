import asyncio
import json
import pickle
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
from rich.table import Table

from synthetic_data.config import PipelineConfig
from synthetic_data.dataset import DatasetBuilder, HuggingFaceExporter
from synthetic_data.extractors import ContentChunker, DocumentIngestion, ImageTranscriber
from synthetic_data.generators import CategoryManager, GenerationPipeline
from synthetic_data.models import ModelRegistry
from synthetic_data.parsers.base import Document
from synthetic_data.utils import CheckpointManager, PipelineCache, QualityFilter

console = Console()


def parse(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
    clear_cache: bool = typer.Option(False, "--clear-cache", help="Clear cache before running"),
):
    """Step 1: Parse documents and resolve images (no transcription)."""
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

    # Initialize cache
    cache_dir = output_dir.parent / ".cache"
    cache = PipelineCache(cache_dir)

    if clear_cache:
        console.print("[yellow]Clearing cache...[/yellow]")
        cache.clear_stage("parse")

    # Check for existing output
    output_file = output_dir / "documents.pkl"
    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Loading from: {output_file}[/cyan]")
        with open(output_file, "rb") as f:
            all_documents = pickle.load(f)
        console.print(f"[green]✓ Loaded {len(all_documents)} documents[/green]")
        return

    # Parse documents with parallel processing
    ingestion = DocumentIngestion(images_output_dir=images_dir)
    all_documents = []

    # Helper function for parallel parsing
    def parse_file_safe(file_path: Path, source) -> Optional[Document]:
        """Parse a file safely, returning None on error."""
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
        # Collect all files to process
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

            # Add source context to each file
            all_files_to_process.extend([(f, source) for f in files])

        total_files = len(all_files_to_process)
        task = progress.add_task(f"Parsing {total_files} files...", total=total_files)

        # Parse files in parallel
        max_workers = min(8, total_files)  # Limit parallelism
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all parsing tasks
            future_to_file = {
                executor.submit(parse_file_safe, file_path, source): file_path
                for file_path, source in all_files_to_process
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                doc = future.result()
                if doc:
                    all_documents.append(doc)
                progress.update(task, advance=1)

    # Save results
    with open(output_file, "wb") as f:
        pickle.dump(all_documents, f)

    # Save summary
    summary = {
        "total_documents": len(all_documents),
        "total_images": sum(len(doc.images) for doc in all_documents),
        "images_resolved": sum(
            1 for doc in all_documents for img in doc.images if img.resolved_path
        ),
        "total_code_blocks": sum(len(doc.code_blocks) for doc in all_documents),
        "documents": [
            {
                "path": str(doc.source_path),
                "title": doc.title,
                "has_code": doc.has_code,
                "has_images": doc.has_images,
                "image_count": len(doc.images),
                "code_block_count": len(doc.code_blocks),
            }
            for doc in all_documents
        ],
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Display results
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
    """Step 2: Transcribe images using VLM (batch async with checkpoints)."""
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

    # Load documents
    parsed_dir = Path(config.dataset.parsed_dir)
    input_file = parsed_dir / "documents.pkl"

    if not input_file.exists():
        console.print("[red]✗ No parsed documents found. Run 'parse' first.[/red]")
        raise typer.Exit(1)

    with open(input_file, "rb") as f:
        all_documents = pickle.load(f)

    console.print(f"[cyan]Loaded {len(all_documents)} documents[/cyan]")

    # Setup output
    output_dir = Path(config.dataset.parsed_dir).parent / "transcribed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "documents.pkl"

    # Setup checkpoint manager
    checkpoint_dir = output_dir.parent / ".checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir, "transcribe")

    # Check for existing final output
    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Transcriptions already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-transcribe[/yellow]")
        return

    # Try to resume from checkpoint
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

    # Count images needing transcription
    images_to_transcribe = sum(
        1
        for doc in all_documents
        for img in doc.images
        if img.resolved_path and not img.transcription
    )

    if images_to_transcribe == 0:
        console.print("[yellow]No images need transcription[/yellow]")
        # Save as-is
        with open(output_file, "wb") as f:
            pickle.dump(all_documents, f)
        checkpoint_manager.clear_checkpoint()
        return

    console.print(f"[cyan]Found {images_to_transcribe} images to transcribe[/cyan]")
    console.print(
        f"[cyan]Using batch_size={config.generation.vlm_batch_size}, "
        f"concurrency={config.generation.vlm_concurrency}[/cyan]"
    )

    # Transcribe images with checkpoints
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Count already transcribed
        already_transcribed = sum(
            1 for doc in all_documents for img in doc.images if img.transcription
        )

        task = progress.add_task(
            "Transcribing images...",
            total=images_to_transcribe + already_transcribed,
            completed=already_transcribed,
        )

        # Progress callback
        def update_transcription_progress(completed):
            progress.update(task, completed=already_transcribed + completed)

        # Checkpoint callback
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

                # Batch transcribe with checkpoint callback
                asyncio.run(
                    transcriber.transcribe_batch_documents_async(
                        all_documents,
                        progress_callback=update_transcription_progress,
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

    # Save final transcribed documents
    with open(output_file, "wb") as f:
        pickle.dump(all_documents, f)

    # Clear checkpoint after successful completion
    checkpoint_manager.clear_checkpoint()

    # Save summary
    summary = {
        "total_documents": len(all_documents),
        "total_images": sum(len(doc.images) for doc in all_documents),
        "images_transcribed": sum(
            1 for doc in all_documents for img in doc.images if img.transcription
        ),
        "sample_transcriptions": [
            {
                "image_path": img.resolved_path,
                "transcription": (
                    img.transcription[:200] + "..."
                    if len(img.transcription) > 200
                    else img.transcription
                ),
            }
            for doc in all_documents[:3]
            for img in doc.images
            if img.transcription
        ][:5],
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"[green]✓ Saved to: {output_dir}[/green]")


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

    # Load documents (prefer transcribed, fallback to parsed)
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

    # Setup output
    output_dir = Path(config.dataset.parsed_dir).parent / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks.pkl"

    # Setup checkpoint manager
    checkpoint_dir = output_dir.parent / ".checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir, "chunk")

    # Check for existing final output
    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Chunks already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-chunk[/yellow]")
        return

    # Try to resume from checkpoint
    processed_indices = set()
    all_chunks = []

    if not no_cache and checkpoint_manager.exists():
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            all_chunks, metadata = checkpoint_data
            processed_indices = set(metadata.get("processed_indices", []))
            console.print(
                f"[cyan]Resuming from checkpoint: "
                f"{len(processed_indices)} documents already chunked[/cyan]"
            )

    # Chunk documents with parallel processing and checkpoints
    chunker = ContentChunker(
        max_length=config.generation.max_context_length,
        overlap=config.generation.chunk_overlap,
    )

    # Filter out already processed documents
    remaining_documents = [
        (i, doc) for i, doc in enumerate(all_documents) if i not in processed_indices
    ]

    if not remaining_documents:
        console.print("[yellow]All documents already chunked![/yellow]")
        # Save and return
        with open(output_file, "wb") as f:
            pickle.dump(all_chunks, f)
        checkpoint_manager.clear_checkpoint()
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        total = len(all_documents)
        task = progress.add_task("Chunking documents...", total=total)
        progress.update(task, completed=len(processed_indices))

        # Checkpoint callback
        def save_checkpoint():
            metadata = {
                "processed_indices": list(processed_indices),
                "total_documents": len(all_documents),
            }
            checkpoint_manager.save_checkpoint(all_chunks, metadata)

        # Process in batches for checkpoint saving
        checkpoint_batch_size = config.generation.llm_batch_size
        max_workers = min(8, len(remaining_documents))

        for batch_start in range(0, len(remaining_documents), checkpoint_batch_size):
            batch = remaining_documents[batch_start : batch_start + checkpoint_batch_size]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_doc = {
                    executor.submit(chunker.chunk_document, doc): (idx, doc) for idx, doc in batch
                }

                for future in as_completed(future_to_doc):
                    idx, _ = future_to_doc[future]
                    chunks = future.result()
                    all_chunks.extend(chunks)
                    processed_indices.add(idx)
                    progress.update(task, advance=1)

            # Save checkpoint after each batch
            save_checkpoint()

    # Save final chunks
    with open(output_file, "wb") as f:
        pickle.dump(all_chunks, f)

    # Clear checkpoint after successful completion
    checkpoint_manager.clear_checkpoint()

    # Save summary
    summary = {
        "total_chunks": len(all_chunks),
        "chunks_with_code": sum(1 for c in all_chunks if c.code_blocks),
        "chunks_with_images": sum(1 for c in all_chunks if c.images),
        "multimodal_chunks": sum(
            1 for c in all_chunks if c.images and any(img.transcription for img in c.images)
        ),
        "avg_chunk_length": (
            sum(len(c.text) for c in all_chunks) / len(all_chunks) if all_chunks else 0
        ),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save chunk samples for debugging (first 10 chunks with details)
    if all_chunks:
        chunk_samples = []
        sample_size = min(10, len(all_chunks))

        for i in range(sample_size):
            chunk = all_chunks[i]
            chunk_samples.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_path": str(chunk.source_path),
                    "text_length": len(chunk.text),
                    "text_preview": (
                        chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text
                    ),
                    "text_full": chunk.text,  # Full text for debugging
                    "has_code": len(chunk.code_blocks) > 0,
                    "num_code_blocks": len(chunk.code_blocks),
                    "has_images": len(chunk.images) > 0,
                    "num_images": len(chunk.images),
                    "images": [
                        {
                            "path": img.path,
                            "alt_text": img.alt_text,
                            "has_transcription": bool(img.transcription),
                            "resolved_path": img.resolved_path,
                        }
                        for img in chunk.images
                    ],
                    "previous_chunk_text_length": len(chunk.previous_chunk_text),
                    "next_chunk_text_length": len(chunk.next_chunk_text),
                    "all_document_code_count": len(chunk.all_document_code),
                }
            )

        with open(output_dir / "chunk_samples.json", "w") as f:
            json.dump({"sample_count": sample_size, "samples": chunk_samples}, f, indent=2)

        console.print(f"[cyan]Saved {sample_size} chunk samples for debugging[/cyan]")

    # Display results
    table = Table(title="Chunking Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Total Chunks", str(len(all_chunks)))
    table.add_row("Chunks with Code", str(summary["chunks_with_code"]))
    table.add_row("Chunks with Images", str(summary["chunks_with_images"]))
    table.add_row("Multimodal Chunks", str(summary["multimodal_chunks"]))
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

        # Just copy chunks as-is
        chunks_dir = Path(config.dataset.parsed_dir).parent / "chunks"
        output_dir = Path(config.dataset.parsed_dir).parent / "filtered"
        output_dir.mkdir(parents=True, exist_ok=True)

        if (chunks_dir / "chunks.pkl").exists():
            import shutil

            shutil.copy2(chunks_dir / "chunks.pkl", output_dir / "chunks.pkl")
            shutil.copy2(chunks_dir / "summary.json", output_dir / "summary.json")
            console.print(f"[green]✓ Copied unfiltered chunks to: {output_dir}[/green]")
        return

    # Load chunks
    chunks_dir = Path(config.dataset.parsed_dir).parent / "chunks"
    input_file = chunks_dir / "chunks.pkl"

    if not input_file.exists():
        console.print("[red]✗ No chunks found. Run 'chunk' first.[/red]")
        raise typer.Exit(1)

    with open(input_file, "rb") as f:
        all_chunks = pickle.load(f)

    console.print(f"[cyan]Loaded {len(all_chunks)} chunks[/cyan]")

    # Setup output
    output_dir = Path(config.dataset.parsed_dir).parent / "filtered"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks.pkl"

    # Setup checkpoint manager
    checkpoint_dir = output_dir.parent / ".checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir, "filter")

    # Check for existing final output
    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Filtered chunks already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-filter[/yellow]")
        return

    # Try to resume from checkpoint
    filter_results = []
    debug_info = []
    processed_count = 0

    if not no_cache and checkpoint_manager.exists():
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            checkpoint_state, _ = checkpoint_data
            filter_results = checkpoint_state.get("results", [])
            debug_info = checkpoint_state.get("debug_info", [])
            processed_count = len(filter_results)
            console.print(
                f"[cyan]Resuming from checkpoint: {processed_count} chunks already filtered[/cyan]"
            )

    # Filter remaining chunks
    remaining_chunks = all_chunks[processed_count:]

    if not remaining_chunks:
        console.print("[yellow]All chunks already filtered![/yellow]")
        filtered_chunks = [chunk for chunk, passed in filter_results if passed]
        with open(output_file, "wb") as f:
            pickle.dump(filtered_chunks, f)
        checkpoint_manager.clear_checkpoint()
        return

    # Filter chunks using batch processing
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Filtering chunks...", total=len(all_chunks))
        progress.update(task, completed=processed_count)

        def update_filter_progress(completed):
            progress.update(task, completed=processed_count + completed)

        def save_checkpoint(partial_results, partial_debug):
            checkpoint_state = {
                "results": filter_results + partial_results,
                "debug_info": debug_info + partial_debug,
            }
            metadata = {
                "processed_count": len(checkpoint_state["results"]),
                "total_chunks": len(all_chunks),
            }
            checkpoint_manager.save_checkpoint(checkpoint_state, metadata)

        with ModelRegistry(config.models) as model_registry:
            filter_model_name = config.generation.filter_model or config.generation.curate_model
            filter_client = model_registry.get_llm_client(filter_model_name)
            quality_filter = QualityFilter(filter_client)

            console.print(
                f"[cyan]Batch size={config.generation.llm_batch_size}, "
                f"concurrency={config.generation.llm_concurrency}[/cyan]"
            )

            try:
                new_results, new_debug = asyncio.run(
                    quality_filter.filter_chunks_batch_async(
                        remaining_chunks,
                        config.prompts.content_quality_check,
                        config.prompts.image_quality_check,
                        config.prompts.content_filter_system,
                        config.prompts.image_filter_system,
                        batch_size=config.generation.llm_batch_size,
                        max_concurrent=config.generation.llm_concurrency,
                        progress_callback=update_filter_progress,
                        checkpoint_callback=save_checkpoint,
                        save_debug=True,
                    )
                )

                filter_results.extend(new_results)
                debug_info.extend(new_debug)

            except Exception as e:
                console.print(f"[red]✗ Filtering failed: {e}[/red]")
                console.print("[yellow]Progress has been saved to checkpoint[/yellow]")
                raise

    filtered_chunks = [chunk for chunk, passed in filter_results if passed]
    total_images_before = sum(len(chunk.images) for chunk, _ in filter_results)
    total_images_after = sum(len(chunk.images) for chunk in filtered_chunks)

    # Save filtered chunks
    with open(output_file, "wb") as f:
        pickle.dump(filtered_chunks, f)

    # Clear checkpoint after successful completion
    checkpoint_manager.clear_checkpoint()

    if debug_info:
        debug_file = output_dir / "filter_debug.jsonl"
        with open(debug_file, "w") as f:
            for entry in debug_info:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")

        rejected_count = sum(1 for entry in debug_info if entry["decision"] == "REJECT")
        console.print(f"\n[cyan]ℹ Filter debug info saved: {debug_file}[/cyan]")
        console.print(
            f"[cyan]  Total filtered: {len(debug_info)} items, {rejected_count} rejected[/cyan]"
        )

    # Save summary
    summary = {
        "chunks_before": len(all_chunks),
        "chunks_after": len(filtered_chunks),
        "chunks_removed": len(all_chunks) - len(filtered_chunks),
        "images_before": total_images_before,
        "images_after": total_images_after,
        "filter_rate": 1 - (len(filtered_chunks) / len(all_chunks)) if all_chunks else 0,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Display results
    table = Table(title="Filtering Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_row("Chunks Before", str(len(all_chunks)))
    table.add_row("Chunks After", str(len(filtered_chunks)))
    table.add_row("Chunks Removed", str(len(all_chunks) - len(filtered_chunks)))
    table.add_row("Images Before", str(total_images_before))
    table.add_row("Images After", str(total_images_after))

    console.print("\n")
    console.print(table)
    console.print(f"[green]✓ Saved to: {output_dir}[/green]")


def classify(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
):
    """Step 5: Classify chunks into categories."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 5: Category Classification[/bold cyan]\n"
            "Classify chunks into knowledge categories",
            title="Classify",
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

    # Setup output
    output_dir = Path(config.dataset.parsed_dir).parent / "classified"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks_by_category.pkl"

    # Setup checkpoint manager
    checkpoint_dir = output_dir.parent / ".checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir, "classify")

    # Check for existing final output
    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Classifications already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-classify[/yellow]")
        return

    # Try to resume from checkpoint
    if not no_cache and checkpoint_manager.exists():
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            chunks_by_category, _ = checkpoint_data
            classified_count = sum(len(chunks) for chunks in chunks_by_category.values())
            console.print(
                f"[cyan]Resuming from checkpoint: {classified_count} chunks already classified[/cyan]"
            )
            # Save final and return if complete
            if classified_count >= len(all_chunks):
                with open(output_file, "wb") as f:
                    pickle.dump(chunks_by_category, f)
                checkpoint_manager.clear_checkpoint()
                console.print("[green]✓ Classification already complete[/green]")
                return

    # Classify chunks using batch processing
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Classifying chunks...", total=len(all_chunks))

        # Progress callback
        def update_classify_progress(completed):
            progress.update(task, completed=completed)

        # Checkpoint callback
        def save_checkpoint(organized_dict):
            classified_count = sum(len(chunks) for chunks in organized_dict.values())
            metadata = {
                "classified_count": classified_count,
                "total_chunks": len(all_chunks),
            }
            checkpoint_manager.save_checkpoint(organized_dict, metadata)

        with ModelRegistry(config.models) as model_registry:
            # Use dedicated classify model if specified, otherwise fall back to curate model
            classify_model_name = config.generation.classify_model or config.generation.curate_model
            classify_client = model_registry.get_llm_client(classify_model_name)
            category_manager = CategoryManager(config.categories, classify_client)

            console.print(
                f"[cyan]Batch size={config.generation.llm_batch_size}, "
                f"concurrency={config.generation.llm_concurrency}[/cyan]"
            )

            try:
                # organize_by_category now uses batch processing with checkpoints
                chunks_by_category = category_manager.organize_by_category(
                    all_chunks,
                    config.prompts.category_classification,
                    config.prompts.category_classification_system,
                    batch_size=config.generation.llm_batch_size,
                    max_concurrent=config.generation.llm_concurrency,
                    progress_callback=update_classify_progress,
                    checkpoint_callback=save_checkpoint,
                )

            except Exception as e:
                console.print(f"[red]✗ Classification failed: {e}[/red]")
                console.print("[yellow]Progress has been saved to checkpoint[/yellow]")
                raise

    # Save classified chunks
    with open(output_file, "wb") as f:
        pickle.dump(chunks_by_category, f)

    # Clear checkpoint after successful completion
    checkpoint_manager.clear_checkpoint()

    # Save summary
    summary = {
        "total_chunks": len(all_chunks),
        "total_categories": len(chunks_by_category),
        "distribution": {
            category: len(chunks)
            for category, chunks in sorted(
                chunks_by_category.items(), key=lambda x: len(x[1]), reverse=True
            )
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Display results
    table = Table(title="Classification Complete")
    table.add_column("Category", style="cyan")
    table.add_column("Chunks", style="green", justify="right")

    for category, chunks in sorted(
        chunks_by_category.items(), key=lambda x: len(x[1]), reverse=True
    ):
        if len(chunks) > 0:
            table.add_row(category, str(len(chunks)))

    console.print("\n")
    console.print(table)
    console.print(f"[green]✓ Saved to: {output_dir}[/green]")


def generate(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
):
    """Step 6: Generate synthetic Q&A samples."""
    console.print(
        Panel.fit(
            "[bold cyan]Step 6: Sample Generation[/bold cyan]\n"
            "Generate synthetic Q&A samples from chunks",
            title="Generate",
        )
    )

    config = PipelineConfig.from_yaml(config_path)

    # Load classified chunks
    classified_dir = Path(config.dataset.parsed_dir).parent / "classified"
    input_file = classified_dir / "chunks_by_category.pkl"

    if not input_file.exists():
        console.print("[red]✗ No classified chunks found. Run 'classify' first.[/red]")
        raise typer.Exit(1)

    with open(input_file, "rb") as f:
        chunks_by_category = pickle.load(f)

    total_chunks = sum(len(chunks) for chunks in chunks_by_category.values())
    console.print(
        f"[cyan]Loaded {total_chunks} chunks in {len(chunks_by_category)} categories[/cyan]"
    )

    # Setup output
    output_dir = Path(config.dataset.generated_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "samples.pkl"

    # Setup checkpoint manager
    checkpoint_dir = output_dir.parent / ".checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir, "generate")

    # Check for existing final output
    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Samples already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-generate[/yellow]")
        return

    # Try to resume from checkpoint
    samples = []
    rejected_samples_list = []
    code_failures_list = []

    if not no_cache and checkpoint_manager.exists():
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            checkpoint_state, _ = checkpoint_data
            samples = checkpoint_state.get("samples", [])
            rejected_samples_list = checkpoint_state.get("rejected_samples", [])
            code_failures_list = checkpoint_state.get("code_failures", [])

            if samples:
                console.print(
                    f"[cyan]Resuming from checkpoint: {len(samples)} samples already generated[/cyan]"
                )

            # Check if we've reached target
            if len(samples) >= config.generation.target_samples:
                console.print("[yellow]Target samples already generated![/yellow]")
                samples = samples[: config.generation.target_samples]
                # Save final output and return
                with open(output_file, "wb") as f:
                    pickle.dump(samples, f)
                checkpoint_manager.clear_checkpoint()
                console.print(f"[green]✓ {len(samples)} samples ready[/green]")
                return

    # Generate samples
    console.print(f"[cyan]Target: {config.generation.target_samples} samples[/cyan]")
    console.print(
        f"[cyan]Using batch_size={config.generation.llm_batch_size}, "
        f"concurrency={config.generation.llm_concurrency}[/cyan]"
    )

    pipeline_steps = "Question → Answer"
    if config.generation.enable_code_verification:
        pipeline_steps += " → Code Verification"
    if config.generation.enable_curate_filtering:
        pipeline_steps += " → Quality Validation"
    console.print(f"[cyan]Generation pipeline: {pipeline_steps}[/cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Create tasks for each substep (will update totals dynamically)
        questions_task = progress.add_task("Generating questions...", total=None)
        answers_task = progress.add_task("Generating answers...", total=None)
        verification_task = progress.add_task("Verifying code...", total=None)
        curation_task = progress.add_task("Validating quality...", total=None)

        # Progress callbacks to update progress bars
        def set_questions_total(total):
            progress.update(questions_task, total=total)

        def update_question_progress(completed):
            progress.update(questions_task, completed=completed)

        def set_answers_total(total):
            progress.update(answers_task, total=total)

        def update_answer_progress(completed):
            progress.update(answers_task, completed=completed)

        def set_verification_total(total):
            progress.update(verification_task, total=total)

        def update_verification_progress(completed):
            progress.update(verification_task, completed=completed)

        def set_curation_total(total):
            progress.update(curation_task, total=total)

        def update_curation_progress(completed):
            progress.update(curation_task, completed=completed)

        def save_rejected(rejected):
            rejected_samples_list.extend(rejected)

        def save_code_failures(failures_batch):
            code_failures_list.extend(failures_batch)

        # Checkpoint callback (no longer needed in commands.py as pipeline handles it)
        def save_checkpoint():
            pass  # Handled internally by pipeline

        progress_callbacks = {
            "set_questions_total": set_questions_total,
            "questions": update_question_progress,
            "set_answers_total": set_answers_total,
            "answers": update_answer_progress,
            "set_verification_total": set_verification_total,
            "verification": update_verification_progress,
            "set_curation_total": set_curation_total,
            "curation": update_curation_progress,
            "save_rejected": save_rejected,
            "save_code_failures": save_code_failures,
            "save_checkpoint": save_checkpoint,
            "no_cache": no_cache,
        }

        with ModelRegistry(config.models) as model_registry:
            classify_model_name = config.generation.classify_model or config.generation.curate_model
            category_manager = CategoryManager(
                config.categories, model_registry.get_llm_client(classify_model_name)
            )

            pipeline = GenerationPipeline(
                config, model_registry, category_manager, checkpoint_manager
            )

            try:
                samples = pipeline.generate_samples(chunks_by_category, progress_callbacks)
            except Exception as e:
                console.print(f"\n[red]✗ Generation failed: {e}[/red]")
                console.print("[yellow]Progress has been saved to checkpoint[/yellow]")
                raise

    console.print()  # New line after progress bars

    # Trim to target if exceeded
    if len(samples) > config.generation.target_samples:
        samples = samples[: config.generation.target_samples]

    # Save samples
    with open(output_file, "wb") as f:
        pickle.dump(samples, f)

    # Clear checkpoint after successful completion
    checkpoint_manager.clear_checkpoint()

    # Save JSONL for inspection
    with open(output_dir / "samples.jsonl", "w") as f:
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

    # Save rejected samples for inspection
    if rejected_samples_list:
        with open(output_dir / "rejected_samples.jsonl", "w") as f:
            for rejected in rejected_samples_list:
                json.dump(rejected, f, ensure_ascii=False, indent=None)
                f.write("\n")

        console.print(
            f"\n[yellow]ℹ {len(rejected_samples_list)} rejected samples saved to: "
            f"{output_dir / 'rejected_samples.jsonl'}[/yellow]"
        )

    # Save code verification failures for debugging
    if code_failures_list:
        with open(output_dir / "code_verification_failures.jsonl", "w") as f:
            for failure in code_failures_list:
                json.dump(failure, f, ensure_ascii=False, indent=None)
                f.write("\n")

        console.print(
            f"[yellow]ℹ {len(code_failures_list)} code verification failures saved to: "
            f"{output_dir / 'code_verification_failures.jsonl'}[/yellow]"
        )

    # Save summary
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
        "curation_enabled": config.generation.enable_curate_filtering,
        "code_verification_enabled": config.generation.enable_code_verification,
    }

    for sample in samples:
        summary["by_type"][sample.question_type] = (
            summary["by_type"].get(sample.question_type, 0) + 1
        )
        summary["by_category"][sample.category] = summary["by_category"].get(sample.category, 0) + 1

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Display results
    table = Table(title="Generation Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Total Samples", str(len(samples)))
    table.add_row("With Unit Tests", str(summary["samples_with_tests"]))
    if config.generation.enable_curate_filtering:
        table.add_row("Rejected Samples", str(summary["rejected_samples"]))
    if config.generation.enable_code_verification:
        table.add_row("Code Verification Failures", str(summary["code_verification_failures"]))
    table.add_row("Multimodal", str(summary["multimodal_samples"]))
    table.add_row("Text-only", str(summary["text_only_samples"]))

    console.print("\n")
    console.print(table)

    # Show distribution by type
    type_table = Table(title="By Question Type")
    type_table.add_column("Type", style="cyan")
    type_table.add_column("Count", style="green", justify="right")
    for qtype, count in sorted(summary["by_type"].items()):
        type_table.add_row(qtype, str(count))

    console.print("\n")
    console.print(type_table)
    console.print(f"\n[green]✓ Saved to: {output_dir}[/green]")

    if rejected_samples_list:
        console.print(
            f"[yellow]ℹ Review rejected samples at: {output_dir / 'rejected_samples.jsonl'}[/yellow]"
        )

    if code_failures_list:
        console.print(
            f"[yellow]ℹ Review code verification failures at: {output_dir / 'code_verification_failures.jsonl'}[/yellow]"
        )


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

    # Load samples
    generated_dir = Path(config.dataset.generated_dir)
    input_file = generated_dir / "samples.pkl"

    if not input_file.exists():
        console.print("[red]✗ No samples found. Run 'generate' first.[/red]")
        raise typer.Exit(1)

    with open(input_file, "rb") as f:
        samples = pickle.load(f)

    console.print(f"[cyan]Loaded {len(samples)} samples[/cyan]")

    # Setup output
    output_dir = Path(config.dataset.parsed_dir).parent / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build splits
    builder = DatasetBuilder(config.dataset, config.seed)
    train_samples, val_samples, test_samples = builder.stratified_build(samples)

    # Print distribution analysis
    builder.print_split_comparison(train_samples, val_samples, test_samples)

    # Save splits
    with open(output_dir / "train.pkl", "wb") as f:
        pickle.dump(train_samples, f)

    with open(output_dir / "val.pkl", "wb") as f:
        pickle.dump(val_samples, f)

    with open(output_dir / "test.pkl", "wb") as f:
        pickle.dump(test_samples, f)

    # Save distribution statistics
    summary = {
        "total_samples": len(samples),
        "train": builder.get_distribution_stats(train_samples),
        "val": builder.get_distribution_stats(val_samples),
        "test": builder.get_distribution_stats(test_samples),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Display results
    table = Table(title="Dataset Splits")
    table.add_column("Split", style="cyan")
    table.add_column("Samples", style="green", justify="right")
    table.add_column("Multimodal", style="yellow", justify="right")
    table.add_column("Percentage", style="magenta", justify="right")

    total = len(samples)
    train_pct = len(train_samples) / total * 100 if total > 0 else 0
    val_pct = len(val_samples) / total * 100 if total > 0 else 0
    test_pct = len(test_samples) / total * 100 if total > 0 else 0

    train_mm = sum(1 for s in train_samples if s.image_path)
    val_mm = sum(1 for s in val_samples if s.image_path)
    test_mm = sum(1 for s in test_samples if s.image_path)

    table.add_row(
        "Train",
        str(len(train_samples)),
        f"{train_mm} ({train_mm/len(train_samples)*100:.0f}%)" if train_samples else "0",
        f"{train_pct:.1f}%",
    )
    table.add_row(
        "Validation",
        str(len(val_samples)),
        f"{val_mm} ({val_mm/len(val_samples)*100:.0f}%)" if val_samples else "0",
        f"{val_pct:.1f}%",
    )
    table.add_row(
        "Test",
        str(len(test_samples)),
        f"{test_mm} ({test_mm/len(test_samples)*100:.0f}%)" if test_samples else "0",
        f"{test_pct:.1f}%",
    )

    console.print("\n")
    console.print(table)
    console.print(f"[green]✓ Saved to: {output_dir}[/green]")


def export(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    hub_id: Optional[str] = typer.Option(None, "--hub-id", "-h", help="HuggingFace Hub ID"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="HuggingFace token"),
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

    # Load splits
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

    # Export to HuggingFace format
    exporter = HuggingFaceExporter(config.dataset)
    dataset_dict = exporter.export(train_samples, val_samples, test_samples)

    # Save to disk
    output_path = exporter.save_to_disk(dataset_dict)
    console.print(f"[green]✓ Saved dataset to: {output_path}[/green]")

    # Push to hub if requested
    if hub_id:
        console.print(f"\n[cyan]Pushing to HuggingFace Hub: {hub_id}[/cyan]")

        # Filter out empty splits to avoid HuggingFace push_to_hub bug
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

            # Report which splits were pushed
            pushed_splits = list(non_empty_dict.keys())
            empty_splits = [s for s in dataset_dict.keys() if s not in pushed_splits]
            if empty_splits:
                console.print(
                    f"[yellow]Note: Empty splits not pushed: {', '.join(empty_splits)}[/yellow]"
                )
        else:
            console.print("[yellow]⚠ All splits are empty, cannot push to hub[/yellow]")

    # Display final summary
    table = Table(title="Export Complete")
    table.add_column("Split", style="cyan")
    table.add_column("Samples", style="green", justify="right")

    for split in dataset_dict.keys():
        table.add_row(split, str(len(dataset_dict[split])))

    console.print("\n")
    console.print(table)


def pipeline(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
    hub_id: Optional[str] = typer.Option(None, "--hub-id", "-h", help="HuggingFace Hub ID"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="HuggingFace token"),
):
    """Run the complete pipeline: parse -> transcribe -> chunk -> filter -> classify -> generate -> build -> export."""
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
        ("Classify", lambda: classify(config_path, no_cache)),
        ("Generate", lambda: generate(config_path, no_cache)),
        ("Build", lambda: build(config_path)),
        ("Export", lambda: export(config_path, hub_id, token)),
    ]

    for i, (step_name, step_func) in enumerate(steps, 1):
        console.print(f"\n[bold]━━━ Step {i}/8: {step_name} ━━━[/bold]\n")
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
            f"All 8 steps completed successfully\n"
            f"Total time: {minutes}m {seconds}s",
            title="Success",
            border_style="green",
        )
    )
