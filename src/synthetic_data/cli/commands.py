"""CLI commands implementation."""

import hashlib
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from synthetic_data.config import PipelineConfig
from synthetic_data.dataset import DatasetBuilder, HuggingFaceExporter
from synthetic_data.extractors import ContentChunker, DocumentIngestion, ImageTranscriber
from synthetic_data.generators import CategoryManager, GenerationPipeline
from synthetic_data.models import ModelRegistry
from synthetic_data.utils import PipelineCache, QualityFilter

console = Console()


def parse(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache usage"),
    clear_cache: bool = typer.Option(False, "--clear-cache", help="Clear cache before running"),
):
    """Parse documentation and extract content."""
    console.print("[bold cyan]Stage 1: Parsing Documentation[/bold cyan]\n")

    config = PipelineConfig.from_yaml(config_path)
    images_dir = Path(config.dataset.images_dir)

    # Initialize cache
    cache_dir = Path(config.dataset.parsed_dir).parent / ".cache"
    cache = PipelineCache(cache_dir)

    if clear_cache:
        console.print("[yellow]Clearing cache...[/yellow]")
        cache.clear_all()

    # Show cache info
    if not no_cache:
        cache_info = cache.get_cache_info()
        if cache_info:
            console.print("[cyan]Cache status:[/cyan]")
            for stage, info in cache_info.items():
                size_mb = info["size"] / 1024 / 1024
                console.print(f"  {stage}: {info['count']} items, {size_mb:.1f}MB")
            console.print()

    # Parse documents with optional image transcription
    all_documents = []

    if config.generation.enable_image_transcription and config.generation.vision_model:
        console.print("[bold cyan]Image transcription enabled[/bold cyan]")

        with ModelRegistry(config.models) as model_registry:
            try:
                vision_client = model_registry.get_vlm_client(config.generation.vision_model)
                image_transcriber = ImageTranscriber(
                    vision_client,
                    config.prompts.image_transcription,
                    batch_size=config.generation.vlm_batch_size,
                    max_concurrent=config.generation.vlm_concurrency,
                )
                console.print(
                    f"  ✓ Vision model loaded (batch={config.generation.vlm_batch_size}, "
                    f"concurrency={config.generation.vlm_concurrency})\n"
                )
            except Exception as e:
                console.print(f"  Warning: Vision model not available: {e}")
                console.print("  Continuing without image transcription\n")
                image_transcriber = None

            ingestion = DocumentIngestion(
                images_output_dir=images_dir, image_transcriber=image_transcriber
            )

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
                # Count total files across all sources
                total_files = 0
                source_file_counts = {}
                for source in config.sources:
                    source_path = Path(source.path)
                    count = 0
                    if source_path.is_file():
                        count = 1
                    elif source_path.is_dir():
                        for pattern in source.include_patterns:
                            count += len(list(source_path.rglob(pattern)))
                    # Apply max_files limit if set
                    if source.max_files is not None and count > source.max_files:
                        count = source.max_files
                    source_file_counts[source.path] = count
                    total_files += count

                task = progress.add_task(f"Parsing {total_files} files...", total=total_files)
                files_processed = 0

                for source in config.sources:
                    source_path = Path(source.path)

                    # Get list of files for this source
                    source_files = []
                    if source_path.is_file():
                        source_files = [source_path]
                    elif source_path.is_dir():
                        for pattern in source.include_patterns:
                            source_files.extend(list(source_path.rglob(pattern)))

                    # Apply max_files limit if set
                    if source.max_files is not None:
                        source_files = source_files[: source.max_files]

                    # Generate unique cache stage for this source
                    source_cache_id = hashlib.md5(str(source.path).encode()).hexdigest()[:8]
                    cache_stage = f"parse_{source_cache_id}"

                    cache_key = cache.get_stage_cache_key(
                        cache_stage,
                        {
                            "transcription": config.generation.enable_image_transcription,
                            "vision_model": config.generation.vision_model,
                        },
                        source_files,
                    )

                    # Check cache for this specific source
                    if not no_cache and cache.is_cached(cache_stage, cache_key):
                        cached_docs = cache.load_documents(cache_stage, cache_key)
                        if cached_docs:
                            all_documents.extend(cached_docs)
                            files_processed += len(source_files)
                            progress.update(task, advance=len(source_files))
                            console.print(
                                f"  ✓ [{files_processed}/{total_files}] Loaded from cache: "
                                f"{source.path} ({len(cached_docs)} docs, {len(source_files)} files)"
                            )
                            continue

                    # Parse if not cached
                    documents = ingestion.ingest_source(source)
                    all_documents.extend(documents)
                    files_processed += len(source_files)

                    # Save to cache immediately after processing this source
                    if not no_cache and documents:
                        cache.save_documents(cache_stage, cache_key, documents)

                    progress.update(task, advance=len(source_files))
                    console.print(
                        f"  ✓ [{files_processed}/{total_files}] Processed {source.path}: "
                        f"{len(documents)} docs from {len(source_files)} files"
                    )
    else:
        console.print("[bold cyan]Image transcription disabled[/bold cyan]\n")

        ingestion = DocumentIngestion(images_output_dir=images_dir)

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
            # Count total files across all sources
            total_files = 0
            source_file_counts = {}
            for source in config.sources:
                source_path = Path(source.path)
                count = 0
                if source_path.is_file():
                    count = 1
                elif source_path.is_dir():
                    for pattern in source.include_patterns:
                        count += len(list(source_path.rglob(pattern)))
                # Apply max_files limit if set
                if source.max_files is not None and count > source.max_files:
                    count = source.max_files
                source_file_counts[source.path] = count
                total_files += count

            task = progress.add_task(f"Parsing {total_files} files...", total=total_files)
            files_processed = 0

            for source in config.sources:
                source_path = Path(source.path)

                # Get list of files for this source
                source_files = []
                if source_path.is_file():
                    source_files = [source_path]
                elif source_path.is_dir():
                    for pattern in source.include_patterns:
                        source_files.extend(list(source_path.rglob(pattern)))

                # Apply max_files limit if set
                if source.max_files is not None:
                    source_files = source_files[: source.max_files]

                # Generate unique cache stage for this source
                source_cache_id = hashlib.md5(str(source.path).encode()).hexdigest()[:8]
                cache_stage = f"parse_{source_cache_id}"

                cache_key = cache.get_stage_cache_key(
                    cache_stage,
                    {
                        "transcription": False,
                        "vision_model": None,
                    },
                    source_files,
                )

                # Check cache for this specific source
                if not no_cache and cache.is_cached(cache_stage, cache_key):
                    cached_docs = cache.load_documents(cache_stage, cache_key)
                    if cached_docs:
                        all_documents.extend(cached_docs)
                        files_processed += len(source_files)
                        progress.update(task, advance=len(source_files))
                        console.print(
                            f"  ✓ [{files_processed}/{total_files}] Loaded from cache: "
                            f"{source.path} ({len(cached_docs)} docs, {len(source_files)} files)"
                        )
                        continue

                # Parse if not cached
                documents = ingestion.ingest_source(source)
                all_documents.extend(documents)
                files_processed += len(source_files)

                # Save to cache immediately after processing this source
                if not no_cache and documents:
                    cache.save_documents(cache_stage, cache_key, documents)

                progress.update(task, advance=len(source_files))
                console.print(
                    f"  ✓ [{files_processed}/{total_files}] Processed {source.path}: "
                    f"{len(documents)} docs from {len(source_files)} files"
                )

    # Count resolved images and transcriptions
    total_images = sum(len(doc.images) for doc in all_documents)
    resolved_images = sum(1 for doc in all_documents for img in doc.images if img.resolved_path)
    transcribed_images = sum(1 for doc in all_documents for img in doc.images if img.transcription)

    # Display parsing metrics
    parsing_table = Table(title="Parsing Metrics", show_header=True)
    parsing_table.add_column("Metric", style="cyan")
    parsing_table.add_column("Count", style="green", justify="right")
    parsing_table.add_row("Documents Parsed", str(len(all_documents)))
    parsing_table.add_row("Images Found", str(total_images))
    parsing_table.add_row("Images Resolved", str(resolved_images))
    parsing_table.add_row("Images Transcribed", str(transcribed_images))

    console.print("\n")
    console.print(parsing_table)

    # Stage 2: Chunk and filter content
    console.print("\n[bold cyan]Stage 2: Chunking and Filtering Content[/bold cyan]\n")

    chunker = ContentChunker(
        max_length=config.generation.max_context_length,
        overlap=config.generation.chunk_overlap,
    )

    # Check cache for chunks
    chunk_cache_key = cache.get_stage_cache_key(
        "chunk",
        {
            "max_length": config.generation.max_context_length,
            "overlap": config.generation.chunk_overlap,
        },
        [doc.source_path for doc in all_documents],
    )

    all_chunks = []
    if not no_cache and cache.is_cached("chunk", chunk_cache_key):
        all_chunks = cache.load_chunks("chunk", chunk_cache_key)
        if all_chunks:
            console.print(f"  ✓ Loaded {len(all_chunks)} chunks from cache")
    else:
        # Chunk documents
        for doc in all_documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        console.print(f"  ✓ Created {len(all_chunks)} initial chunks")

        # Save to cache
        if not no_cache:
            cache.save_chunks("chunk", chunk_cache_key, all_chunks)

    # Apply quality filtering if enabled
    if config.generation.enable_content_filtering:
        console.print("\n[bold cyan]Applying quality filtering...[/bold cyan]")

        with ModelRegistry(config.models) as model_registry:
            question_client = model_registry.get_llm_client(config.generation.question_model)
            quality_filter = QualityFilter(question_client)

            filtered_chunks = []
            total_images_before = sum(len(chunk.images) for chunk in all_chunks)

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
                task = progress.add_task(
                    f"Filtering {len(all_chunks)} chunks...", total=len(all_chunks)
                )

                for chunk in all_chunks:
                    try:
                        if quality_filter.is_quality_content(
                            chunk, config.prompts.content_quality_check
                        ):
                            # Filter images based on transcription quality
                            if chunk.is_multimodal:
                                quality_images = []
                                for img in chunk.images:
                                    if quality_filter.is_quality_image(
                                        img, config.prompts.image_quality_check
                                    ):
                                        quality_images.append(img)
                                chunk.images = quality_images

                            filtered_chunks.append(chunk)
                    except Exception as e:
                        console.print(f"\n  Warning: Error filtering chunk: {e}")
                        filtered_chunks.append(chunk)

                    progress.update(task, advance=1)

            removed_count = len(all_chunks) - len(filtered_chunks)
            total_images_after = sum(len(chunk.images) for chunk in filtered_chunks)
            images_removed = total_images_before - total_images_after

            console.print(f"  ✓ Filtered {removed_count} low-quality chunks")
            console.print(f"  ✓ Filtered {images_removed} low-quality images")
            console.print(f"  ✓ Kept {len(filtered_chunks)} high-quality chunks")
            all_chunks = filtered_chunks
    else:
        console.print("  Content filtering disabled")

    # Save to parsed directory
    output_dir = Path(config.dataset.parsed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import json

    summary_file = output_dir / "summary.json"
    summary = {
        "total_documents": len(all_documents),
        "total_chunks": len(all_chunks),
        "filtering_enabled": config.generation.enable_content_filtering,
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

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Save complete chunks for reuse
    chunks_file = output_dir / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            chunk_data = {
                "text": chunk.text,  # Save complete text for reuse
                "source_path": str(chunk.source_path),
                "chunk_id": chunk.chunk_id,
                "code_blocks": chunk.code_blocks,
                "images": [
                    {
                        "path": img.path,
                        "resolved_path": img.resolved_path,
                        "alt_text": img.alt_text,
                        "caption": img.caption,
                        "context": img.context,
                        "transcription": img.transcription,
                    }
                    for img in chunk.images
                ],
                "metadata": chunk.metadata,
                "token_estimate": chunk.token_estimate,
            }
            f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")

    console.print(f"\nSummary saved to {summary_file}")
    console.print(f"Chunks saved to {chunks_file}")


def generate(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
):
    """Generate synthetic dataset (full pipeline)."""
    console.print(
        Panel.fit(
            "[bold cyan]Synthetic Dataset Generation[/bold cyan]\n" f"Target: {config_path.name}",
            title="Pipeline",
        )
    )

    config = PipelineConfig.from_yaml(config_path)
    images_dir = Path(config.dataset.images_dir)

    # Display configuration summary
    config_table = Table(title="Configuration", show_header=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Target Samples", f"{config.generation.target_samples:,}")
    config_table.add_row(
        "LLM Batch/Concurrency",
        f"{config.generation.llm_batch_size}/{config.generation.llm_concurrency}",
    )
    config_table.add_row(
        "VLM Batch/Concurrency",
        f"{config.generation.vlm_batch_size}/{config.generation.vlm_concurrency}",
    )
    config_table.add_row("Multimodal Ratio", f"{config.generation.multimodal_ratio:.1%}")
    config_table.add_row(
        "Image Transcription",
        "Enabled" if config.generation.enable_image_transcription else "Disabled",
    )
    config_table.add_row(
        "Content Filtering", "Enabled" if config.generation.enable_content_filtering else "Disabled"
    )
    console.print(config_table)
    console.print()

    # Stage 1: Parse with optional transcription
    console.print("[bold]Stage 1: Parsing documentation[/bold]")

    all_documents = []

    if config.generation.enable_image_transcription and config.generation.vision_model:
        with ModelRegistry(config.models) as model_registry:
            try:
                vision_client = model_registry.get_vlm_client(config.generation.vision_model)
                image_transcriber = ImageTranscriber(
                    vision_client,
                    config.prompts.image_transcription,
                    batch_size=config.generation.vlm_batch_size,
                    max_concurrent=config.generation.vlm_concurrency,
                )
                console.print(
                    f"  ✓ Image transcription enabled (batch={config.generation.vlm_batch_size}, "
                    f"concurrency={config.generation.vlm_concurrency})"
                )
            except Exception as e:
                console.print(f"  Warning: Vision model not available: {e}")
                image_transcriber = None

            ingestion = DocumentIngestion(
                images_output_dir=images_dir, image_transcriber=image_transcriber
            )

            for source in config.sources:
                console.print(f"  Processing {source.path}...")
                documents = ingestion.ingest_source(source)
                all_documents.extend(documents)
    else:
        ingestion = DocumentIngestion(images_output_dir=images_dir)

        for source in config.sources:
            console.print(f"  Processing {source.path}...")
            documents = ingestion.ingest_source(source)
            all_documents.extend(documents)

    transcribed_images = sum(1 for doc in all_documents for img in doc.images if img.transcription)

    # Display parsing summary
    parse_summary = Table(title="Stage 1 Summary")
    parse_summary.add_column("Metric", style="cyan")
    parse_summary.add_column("Count", style="green", justify="right")
    parse_summary.add_row("Documents Parsed", str(len(all_documents)))
    parse_summary.add_row("Images Transcribed", str(transcribed_images))
    console.print(parse_summary)
    console.print()

    # Stage 2: Chunk
    console.print("[bold]Stage 2: Chunking content[/bold]")
    chunker = ContentChunker(
        max_length=config.generation.max_context_length,
        overlap=config.generation.chunk_overlap,
    )

    all_chunks = []
    for doc in all_documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    # Display chunking summary
    chunk_summary = Table(title="Stage 2 Summary")
    chunk_summary.add_column("Metric", style="cyan")
    chunk_summary.add_column("Value", style="green", justify="right")
    chunk_summary.add_row("Total Chunks", str(len(all_chunks)))
    chunk_summary.add_row("Chunks with Code", str(sum(1 for c in all_chunks if c.code_blocks)))
    chunk_summary.add_row("Chunks with Images", str(sum(1 for c in all_chunks if c.images)))
    console.print(chunk_summary)
    console.print()

    # Stage 3: Classify
    console.print("[bold]Stage 3: Classifying by category[/bold]")

    with ModelRegistry(config.models) as model_registry:
        # Get question model for classification
        question_client = model_registry.get_llm_client(config.generation.question_model)

        category_manager = CategoryManager(config.categories, question_client)

        chunks_by_category = category_manager.organize_by_category(
            all_chunks, config.prompts.category_classification
        )

        # Display classification summary
        classification_table = Table(title="Stage 3 Summary - Category Distribution")
        classification_table.add_column("Category", style="cyan")
        classification_table.add_column("Chunks", style="green", justify="right")

        for category, chunks in sorted(
            chunks_by_category.items(), key=lambda x: len(x[1]), reverse=True
        ):
            classification_table.add_row(category, str(len(chunks)))

        console.print(classification_table)
        console.print()

        # Stage 4: Generate
        console.print("[bold]Stage 4: Generating synthetic samples[/bold]")

        pipeline = GenerationPipeline(config, model_registry, category_manager)
        samples = pipeline.generate_samples(chunks_by_category)

    # Display generation summary
    gen_summary = Table(title="Stage 4 Summary - Generated Samples")
    gen_summary.add_column("Metric", style="cyan")
    gen_summary.add_column("Value", style="green", justify="right")
    gen_summary.add_row("Total Samples", str(len(samples)))
    gen_summary.add_row("Multimodal (with images)", str(sum(1 for s in samples if s.image_path)))
    gen_summary.add_row("Text-only", str(sum(1 for s in samples if not s.image_path)))
    gen_summary.add_row("With Code Context", str(sum(1 for s in samples if s.code_context)))

    # Distribution by type
    by_type = {}
    for s in samples:
        by_type[s.question_type] = by_type.get(s.question_type, 0) + 1
    gen_summary.add_row("", "")  # Spacer
    for qtype, count in sorted(by_type.items()):
        gen_summary.add_row(f"  Type: {qtype}", str(count))

    console.print("\n")
    console.print(gen_summary)
    console.print()

    # Save generated samples
    generated_dir = Path(config.dataset.generated_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)

    import json

    with open(generated_dir / "samples.jsonl", "w") as f:
        for sample in samples:
            json.dump(
                {
                    "question": sample.question,
                    "answer": sample.answer,
                    "category": sample.category,
                    "type": sample.question_type,
                    "source": sample.source_path,
                },
                f,
            )
            f.write("\n")

    console.print(f"Samples saved to {generated_dir}")

    # Stage 5: Build dataset
    console.print("[bold]Stage 5: Building dataset splits[/bold]")
    builder = DatasetBuilder(config.dataset, config.seed)

    train_samples, val_samples, test_samples = builder.stratified_build(samples)

    # Display split summary
    split_summary = Table(title="Stage 5 Summary - Dataset Splits")
    split_summary.add_column("Split", style="cyan")
    split_summary.add_column("Samples", style="green", justify="right")
    split_summary.add_column("Percentage", style="yellow", justify="right")
    split_summary.add_row(
        "Train", str(len(train_samples)), f"{len(train_samples)/len(samples):.1%}"
    )
    split_summary.add_row(
        "Validation", str(len(val_samples)), f"{len(val_samples)/len(samples):.1%}"
    )
    split_summary.add_row("Test", str(len(test_samples)), f"{len(test_samples)/len(samples):.1%}")
    console.print(split_summary)
    console.print()

    # Stage 6: Export
    console.print("[bold]Stage 6: Exporting dataset[/bold]")
    exporter = HuggingFaceExporter(config.dataset)

    dataset_dict = exporter.export(train_samples, val_samples, test_samples)
    output_path = exporter.save_to_disk(dataset_dict)

    # Final summary
    console.print("\n")
    console.print(
        Panel.fit(
            f"[bold green]✓ Dataset Generation Complete![/bold green]\n\n"
            f"Generated: {len(samples):,} samples\n"
            f"Multimodal: {sum(1 for s in samples if s.image_path)} ({sum(1 for s in samples if s.image_path)/len(samples):.1%})\n"
            f"Categories: {len(chunks_by_category)}\n"
            f"Output: {output_path}",
            title="Success",
            border_style="green",
        )
    )
    console.print()


def export(
    dataset_path: Path = typer.Option(..., "--dataset", "-d", help="Dataset directory"),
    hub_id: str = typer.Option(..., "--hub-id", "-h", help="HuggingFace Hub ID"),
    token: str | None = typer.Option(None, "--token", "-t", help="HuggingFace token"),
):
    """Export dataset to HuggingFace Hub."""
    console.print("[bold cyan]Exporting to HuggingFace Hub[/bold cyan]\n")

    from datasets import load_from_disk

    console.print(f"Loading dataset from {dataset_path}...")
    dataset_dict = load_from_disk(str(dataset_path))

    console.print(f"Pushing to {hub_id}...")

    dataset_dict.push_to_hub(
        hub_id,
        token=token,
        private=False,
    )

    console.print(f"\n[green]✓ Dataset exported successfully![/green]")
    console.print(f"[green]  Hub: https://huggingface.co/datasets/{hub_id}[/green]\n")
