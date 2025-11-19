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
from synthetic_data.utils import PipelineCache, QualityFilter

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
    """Step 2: Transcribe images using VLM (batch async)."""
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

    # Check for existing transcriptions
    output_file = output_dir / "documents.pkl"
    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Transcriptions already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-transcribe[/yellow]")
        return

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
        return

    console.print(f"[cyan]Found {images_to_transcribe} images to transcribe[/cyan]")
    console.print(
        f"[cyan]Using batch_size={config.generation.vlm_batch_size}, "
        f"concurrency={config.generation.vlm_concurrency}[/cyan]"
    )

    # Transcribe images
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Transcribing images...", total=images_to_transcribe)

        # Progress callback
        def update_transcription_progress(completed):
            progress.update(task, completed=completed)

        with ModelRegistry(config.models) as model_registry:
            try:
                vision_client = model_registry.get_vlm_client(config.generation.vision_model)
                transcriber = ImageTranscriber(
                    vision_client,
                    config.prompts.image_transcription,
                    batch_size=config.generation.vlm_batch_size,
                    max_concurrent=config.generation.vlm_concurrency,
                )

                # Batch transcribe all documents with progress callback
                asyncio.run(
                    transcriber.transcribe_batch_documents_async(
                        all_documents, update_transcription_progress
                    )
                )

                transcribed_count = sum(
                    1 for doc in all_documents for img in doc.images if img.transcription
                )

                console.print(f"[green]✓ Transcribed {transcribed_count} images[/green]")

            except Exception as e:
                console.print(f"[red]✗ Transcription failed: {e}[/red]")
                raise

    # Save transcribed documents
    with open(output_file, "wb") as f:
        pickle.dump(all_documents, f)

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

    # Check cache
    output_file = output_dir / "chunks.pkl"
    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Chunks already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-chunk[/yellow]")
        return

    # Chunk documents with parallel processing
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

        # Chunk documents in parallel
        max_workers = min(8, len(all_documents))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunking tasks
            future_to_doc = {
                executor.submit(chunker.chunk_document, doc): doc for doc in all_documents
            }

            # Collect results as they complete
            for future in as_completed(future_to_doc):
                chunks = future.result()
                all_chunks.extend(chunks)
                progress.update(task, advance=1)

    # Save chunks
    with open(output_file, "wb") as f:
        pickle.dump(all_chunks, f)

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

    # Check cache
    output_file = output_dir / "chunks.pkl"
    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Filtered chunks already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-filter[/yellow]")
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

        # Progress callback
        def update_filter_progress(completed):
            progress.update(task, completed=completed)

        with ModelRegistry(config.models) as model_registry:
            filter_model_name = config.generation.filter_model or config.generation.curate_model
            filter_client = model_registry.get_llm_client(filter_model_name)
            quality_filter = QualityFilter(filter_client)

            total_images_before = sum(len(chunk.images) for chunk in all_chunks)

            console.print(
                f"[cyan]Batch size={config.generation.llm_batch_size}, "
                f"concurrency={config.generation.llm_concurrency}[/cyan]"
            )

            filter_results, debug_info = asyncio.run(
                quality_filter.filter_chunks_batch_async(
                    all_chunks,
                    config.prompts.content_quality_check,
                    config.prompts.image_quality_check,
                    config.prompts.content_filter_system,
                    config.prompts.image_filter_system,
                    batch_size=config.generation.llm_batch_size,
                    max_concurrent=config.generation.llm_concurrency,
                    progress_callback=update_filter_progress,
                    save_debug=True,
                )
            )

            filtered_chunks = [chunk for chunk, passed in filter_results if passed]

            # Save debug information
            if debug_info:
                debug_file = output_dir / "filter_debug.jsonl"
                with open(debug_file, "w") as f:
                    for entry in debug_info:
                        json.dump(entry, f, ensure_ascii=False)
                        f.write("\n")

                # Count rejections
                rejected_count = sum(1 for entry in debug_info if entry["decision"] == "REJECT")
                console.print(f"\n[cyan]ℹ Filter debug info saved: {debug_file}[/cyan]")
                console.print(
                    f"[cyan]  Total filtered: {len(debug_info)} items, "
                    f"{rejected_count} rejected[/cyan]"
                )

    total_images_after = sum(len(chunk.images) for chunk in filtered_chunks)

    with open(output_file, "wb") as f:
        pickle.dump(filtered_chunks, f)

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

    # Check cache
    output_file = output_dir / "chunks_by_category.pkl"
    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Classifications already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-classify[/yellow]")
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

        with ModelRegistry(config.models) as model_registry:
            # Use dedicated classify model if specified, otherwise fall back to curate model
            classify_model_name = config.generation.classify_model or config.generation.curate_model
            classify_client = model_registry.get_llm_client(classify_model_name)
            category_manager = CategoryManager(config.categories, classify_client)

            console.print(
                f"[cyan]Batch size={config.generation.llm_batch_size}, "
                f"concurrency={config.generation.llm_concurrency}[/cyan]"
            )

            # organize_by_category now uses batch processing internally
            chunks_by_category = category_manager.organize_by_category(
                all_chunks,
                config.prompts.category_classification,
                config.prompts.category_classification_system,
                batch_size=config.generation.llm_batch_size,
                max_concurrent=config.generation.llm_concurrency,
                progress_callback=update_classify_progress,
            )

    # Save classified chunks
    with open(output_file, "wb") as f:
        pickle.dump(chunks_by_category, f)

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

    # Check cache
    output_file = output_dir / "samples.pkl"
    if output_file.exists() and not no_cache:
        console.print(f"[cyan]Samples already exist: {output_file}[/cyan]")
        console.print("[yellow]Use --no-cache to re-generate[/yellow]")
        return

    # Generate samples
    console.print(f"[cyan]Target: {config.generation.target_samples} samples[/cyan]")
    console.print(
        f"[cyan]Using batch_size={config.generation.llm_batch_size}, "
        f"concurrency={config.generation.llm_concurrency}[/cyan]"
    )

    pipeline_steps = "Question → Answer"
    if config.generation.enable_curate_filtering:
        pipeline_steps += " → Quality Filter"
    console.print(f"[cyan]Generation pipeline: {pipeline_steps}[/cyan]\n")

    # Storage for rejected samples
    rejected_samples_list = []

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
        curation_task = progress.add_task("Quality filtering...", total=None)

        # Progress callbacks to update progress bars
        def set_questions_total(total):
            progress.update(questions_task, total=total)

        def update_question_progress(completed):
            progress.update(questions_task, completed=completed)

        def set_answers_total(total):
            progress.update(answers_task, total=total)

        def update_answer_progress(completed):
            progress.update(answers_task, completed=completed)

        def set_curation_total(total):
            progress.update(curation_task, total=total)

        def update_curation_progress(completed):
            progress.update(curation_task, completed=completed)

        def save_rejected(rejected):
            rejected_samples_list.extend(rejected)

        progress_callbacks = {
            "set_questions_total": set_questions_total,
            "questions": update_question_progress,
            "set_answers_total": set_answers_total,
            "answers": update_answer_progress,
            "set_curation_total": set_curation_total,
            "curation": update_curation_progress,
            "save_rejected": save_rejected,
        }

        with ModelRegistry(config.models) as model_registry:
            classify_model_name = config.generation.classify_model or config.generation.curate_model
            category_manager = CategoryManager(
                config.categories, model_registry.get_llm_client(classify_model_name)
            )

            pipeline = GenerationPipeline(config, model_registry, category_manager)
            samples = pipeline.generate_samples(chunks_by_category, progress_callbacks)

    console.print()  # New line after progress bars

    # Save samples
    with open(output_file, "wb") as f:
        pickle.dump(samples, f)

    # Save JSONL for inspection
    with open(output_dir / "samples.jsonl", "w") as f:
        for sample in samples:
            json.dump(
                {
                    "question": sample.question,
                    "answer": sample.answer,
                    "category": sample.category,
                    "question_type": sample.question_type,
                    "image_path": sample.image_path,
                    "source_path": sample.source_path,
                },
                f,
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

    # Save summary
    summary = {
        "total_samples": len(samples),
        "rejected_samples": len(rejected_samples_list),
        "multimodal_samples": sum(1 for s in samples if s.image_path),
        "text_only_samples": sum(1 for s in samples if not s.image_path),
        "by_type": {},
        "by_category": {},
        "curation_enabled": config.generation.enable_curate_filtering,
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
    if config.generation.enable_curate_filtering:
        table.add_row("Rejected Samples", str(summary["rejected_samples"]))
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

    # Save splits
    with open(output_dir / "train.pkl", "wb") as f:
        pickle.dump(train_samples, f)

    with open(output_dir / "val.pkl", "wb") as f:
        pickle.dump(val_samples, f)

    with open(output_dir / "test.pkl", "wb") as f:
        pickle.dump(test_samples, f)

    # Save summary
    summary = {
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_percentage": len(train_samples) / len(samples) * 100,
        "val_percentage": len(val_samples) / len(samples) * 100,
        "test_percentage": len(test_samples) / len(samples) * 100,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Display results
    table = Table(title="Dataset Splits")
    table.add_column("Split", style="cyan")
    table.add_column("Samples", style="green", justify="right")
    table.add_column("Percentage", style="yellow", justify="right")

    table.add_row("Train", str(len(train_samples)), f"{summary['train_percentage']:.1f}%")
    table.add_row("Validation", str(len(val_samples)), f"{summary['val_percentage']:.1f}%")
    table.add_row("Test", str(len(test_samples)), f"{summary['test_percentage']:.1f}%")

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
