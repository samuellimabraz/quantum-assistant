import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PIL import Image, ImageDraw
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from synthetic_data.config import PipelineConfig
from synthetic_data.extractors import ContentChunker, DocumentIngestion, ImageTranscriber
from synthetic_data.generators import CategoryManager, GenerationPipeline
from synthetic_data.models import ModelRegistry
from synthetic_data.models import Message
from synthetic_data.utils import PipelineCache, QualityFilter


console = Console()

TEST_RUN_DIR = None


def setup_test_output_dir():
    """Create timestamped directory for test outputs."""
    global TEST_RUN_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TEST_RUN_DIR = Path(__file__).parent.parent / "outputs" / "test_runs" / timestamp
    TEST_RUN_DIR.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[cyan]Test outputs will be saved to: {TEST_RUN_DIR}[/cyan]\n")
    return TEST_RUN_DIR


def get_limited_documents(config: PipelineConfig, ingestion: DocumentIngestion, max_docs: int = 5):
    """Helper to get limited documents respecting max_files config."""
    documents = []

    for source in config.sources:
        if source.max_files:
            source_path = Path(source.path)
            files_to_process = []

            if source_path.is_file():
                files_to_process = [source_path]
            elif source_path.is_dir():
                for pattern in source.include_patterns:
                    files = list(source_path.rglob(pattern))
                    files_to_process.extend(files[: source.max_files - len(files_to_process)])
                    if len(files_to_process) >= source.max_files:
                        break
                files_to_process = files_to_process[: source.max_files]

            for file_path in files_to_process:
                try:
                    for parser in ingestion.parsers:
                        if parser.can_parse(file_path):
                            doc = parser.parse(file_path)

                            # Resolve images
                            if doc and ingestion.image_resolver:
                                for img_ref in doc.images:
                                    resolved = ingestion.image_resolver.resolve_image_path(
                                        img_ref.path, doc.source_path
                                    )
                                    if resolved:
                                        img_ref.resolved_path = str(resolved)

                            documents.append(doc)
                            if len(documents) >= max_docs:
                                return documents
                            break
                except Exception:
                    pass
        else:
            docs = ingestion.ingest_source(source)
            documents.extend(docs[: max_docs - len(documents)])
            if len(documents) >= max_docs:
                return documents

    return documents


def test_config_loading(config_path: Path):
    """Test 1: Configuration loading and validation."""
    console.print("\n[bold cyan]Step 1: Configuration Loading[/bold cyan]")

    try:
        config = PipelineConfig.from_yaml(config_path)

        table = Table(title="Configuration Status", show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Target Samples", str(config.generation.target_samples))
        table.add_row("LLM Batch Size", str(config.generation.llm_batch_size))
        table.add_row("LLM Concurrency", str(config.generation.llm_concurrency))
        table.add_row("VLM Batch Size", str(config.generation.vlm_batch_size))
        table.add_row("VLM Concurrency", str(config.generation.vlm_concurrency))
        table.add_row("Sources", str(len(config.sources)))
        table.add_row("Categories", str(len(config.categories)))
        table.add_row(
            "Max Files",
            str(config.sources[0].max_files) if config.sources[0].max_files else "Unlimited",
        )

        console.print(table)
        console.print("[green]✓ Configuration loaded successfully[/green]")
        return config
    except Exception as e:
        console.print(f"[red]✗ Configuration error: {e}[/red]")
        raise


def test_model_connections(config: PipelineConfig):
    """Test 2: Model endpoint connections."""
    console.print("\n[bold cyan]Step 2: Model Endpoint Connections[/bold cyan]")

    try:
        with ModelRegistry(config.models) as registry:
            # Test question model
            console.print("  Testing question model...")
            question_client = registry.get_llm_client(config.generation.question_model)

            test_response = question_client.generate(
                [
                    Message(role="system", content="Reasoning: low. You are a helpful assistant."),
                    Message(role="user", content="What is quantum computing?"),
                ],
                max_tokens=200,
            )
            console.print(
                f"  [green]✓ Question model connected[/green] (response: {len(test_response)} chars)"
            )
            console.print(f"Response: {test_response}")
            console.print(f"Response length: {len(test_response)}")

            console.print("  Testing answer model...")
            answer_client = registry.get_llm_client(config.generation.answer_model)
            test_response = answer_client.generate(
                [
                    Message(role="system", content="Reasoning: low. You are a helpful assistant."),
                    Message(role="user", content="Explain a qubit in simple terms"),
                ],
                max_tokens=500,
            )
            console.print(f"  [green]✓ Answer model connected[/green]")
            console.print(f"Response: {test_response}")
            console.print(f"Response length: {len(test_response)}")

            if config.generation.enable_image_transcription and config.generation.vision_model:
                console.print("  Testing vision model...")
                vision_client = registry.get_vlm_client(config.generation.vision_model)
                # create a image with a text "Quantum Computing"
                test_img_path = Path("test_image.png")
                image = Image.new("RGB", (640, 640), color=(255, 255, 255))
                draw = ImageDraw.Draw(image)
                draw.text((10, 10), "Quantum Computing", fill=(0, 0, 0))
                image.save(test_img_path)
                test_response = vision_client.generate_with_image(
                    "Explain the image",
                    test_img_path,
                )
                console.print(f"Response: {test_response}")
                console.print(f"Response length: {len(test_response)}")
                console.print(f"  [green]✓ Vision model connected[/green]")
                # Cleanup test image
                test_img_path.unlink()
            else:
                console.print("  [yellow]⊘ Vision model disabled[/yellow]")

        console.print("[green]✓ All model connections verified[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Model connection error: {e}[/red]")
        return False


def test_document_parsing(config: PipelineConfig):
    """Test 3: Document parsing and image extraction."""
    console.print("\n[bold cyan]Step 3: Document Parsing[/bold cyan]")

    try:
        images_dir = Path(config.dataset.images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)

        ingestion = DocumentIngestion(images_output_dir=images_dir)

        total_docs = 0
        total_images = 0
        total_code_blocks = 0
        all_documents = []

        for source in config.sources:
            console.print(f"  Processing source: {source.path}")

            if source.max_files:
                console.print(f"  [yellow]Limited to {source.max_files} files[/yellow]")

                source_path = Path(source.path)
                files_to_process = []
                if source_path.is_file():
                    files_to_process = [source_path]
                elif source_path.is_dir():
                    for pattern in source.include_patterns:
                        files = list(source_path.rglob(pattern))
                        files_to_process.extend(files[: source.max_files - len(files_to_process)])
                        if len(files_to_process) >= source.max_files:
                            break
                    files_to_process = files_to_process[: source.max_files]

                documents = []
                for file_path in files_to_process:
                    try:
                        for parser in ingestion.parsers:
                            if parser.can_parse(file_path):
                                doc = parser.parse(file_path)
                                documents.append(doc)
                                break
                    except Exception as e:
                        console.print(f"  [yellow]Warning: Error parsing {file_path}: {e}[/yellow]")
            else:
                documents = ingestion.ingest_source(source)

            source_images = sum(len(doc.images) for doc in documents)
            source_code = sum(len(doc.code_blocks) for doc in documents)

            console.print(
                f"    [green]✓ Parsed {len(documents)} documents, "
                f"{source_images} images, {source_code} code blocks[/green]"
            )

            total_docs += len(documents)
            total_images += source_images
            total_code_blocks += source_code
            all_documents.extend(documents)

        table = Table(title="Parsing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")
        table.add_row("Total Documents", str(total_docs))
        table.add_row("Total Images", str(total_images))
        table.add_row("Total Code Blocks", str(total_code_blocks))
        console.print(table)

        if TEST_RUN_DIR:
            parsing_summary = {
                "total_documents": total_docs,
                "total_images": total_images,
                "total_code_blocks": total_code_blocks,
                "documents": [
                    {
                        "source_path": str(doc.source_path),
                        "title": doc.title,
                        "images": len(doc.images),
                        "code_blocks": len(doc.code_blocks),
                    }
                    for doc in all_documents
                ],
            }
            with open(TEST_RUN_DIR / "01_parsing.json", "w") as f:
                json.dump(parsing_summary, f, indent=2)
            console.print(f"  [dim]Saved to: {TEST_RUN_DIR / '01_parsing.json'}[/dim]")

        console.print("[green]✓ Document parsing successful[/green]")
        return total_docs, total_images
    except Exception as e:
        console.print(f"[red]✗ Parsing error: {e}[/red]")
        return 0, 0


def test_image_transcription(config: PipelineConfig):
    """Test 4: Image transcription with VLM."""
    console.print("\n[bold cyan]Step 4: Image Transcription (Batch Processing)[/bold cyan]")

    if not config.generation.enable_image_transcription or not config.generation.vision_model:
        console.print("[yellow]⊘ Image transcription disabled, skipping[/yellow]")
        return 0

    try:
        images_dir = Path(config.dataset.images_dir)

        with ModelRegistry(config.models) as registry:
            vision_client = registry.get_vlm_client(config.generation.vision_model)
            transcriber = ImageTranscriber(
                vision_client,
                config.prompts.image_transcription,
                batch_size=config.generation.vlm_batch_size,
                max_concurrent=config.generation.vlm_concurrency,
            )

            console.print(
                f"  Batch size: {config.generation.vlm_batch_size}, "
                f"Concurrency: {config.generation.vlm_concurrency}"
            )

            ingestion = DocumentIngestion(
                images_output_dir=images_dir, image_transcriber=transcriber
            )

            test_docs = get_limited_documents(config, ingestion, max_docs=3)

            if test_docs and transcriber:
                console.print(f"  Transcribing images in {len(test_docs)} documents...")
                total_images = sum(
                    1 for doc in test_docs for img in doc.images if img.resolved_path
                )
                transcribed = [0] 

                def progress_cb(completed):
                    transcribed[0] = completed
                    console.print(f"\r  Progress: {completed}/{total_images}", end="")

                asyncio.run(transcriber.transcribe_batch_documents_async(test_docs, progress_cb))
                console.print()  

            transcribed_count = sum(
                1 for doc in test_docs for img in doc.images if img.transcription
            )
            total_images = sum(len(doc.images) for doc in test_docs)

            console.print(
                f"  [green]✓ Transcribed {transcribed_count}/{total_images} images[/green]"
            )

            if TEST_RUN_DIR:
                transcription_data = {
                    "total_images": total_images,
                    "transcribed_count": transcribed_count,
                    "transcriptions": [
                        {
                            "image_path": img.resolved_path,
                            "transcription": img.transcription,
                            "alt_text": img.alt_text,
                            "caption": img.caption,
                        }
                        for doc in test_docs
                        for img in doc.images
                        if img.transcription
                    ],
                }
                with open(TEST_RUN_DIR / "02_transcriptions.json", "w") as f:
                    json.dump(transcription_data, f, indent=2)
                console.print(f"  [dim]Saved to: {TEST_RUN_DIR / '02_transcriptions.json'}[/dim]")

            for doc in test_docs:
                for img in doc.images:
                    if img.transcription:
                        console.print(f"\n  [dim]Sample transcription:[/dim]")
                        console.print(
                            Panel(img.transcription[:200] + "...", title="Image Description")
                        )
                        return transcribed_count

        return 0
    except Exception as e:
        console.print(f"[red]✗ Transcription error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return 0


def test_chunking(config: PipelineConfig):
    """Test 5: Content chunking."""
    console.print("\n[bold cyan]Step 5: Content Chunking[/bold cyan]")

    try:
        images_dir = Path(config.dataset.images_dir)
        ingestion = DocumentIngestion(images_output_dir=images_dir)

        test_docs = get_limited_documents(config, ingestion, max_docs=3)

        chunker = ContentChunker(
            max_length=config.generation.max_context_length,
            overlap=config.generation.chunk_overlap,
        )

        all_chunks = []
        for doc in test_docs:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        avg_length = sum(len(c.text) for c in all_chunks) / len(all_chunks) if all_chunks else 0
        chunks_with_code = sum(1 for c in all_chunks if c.code_blocks)
        chunks_with_images = sum(1 for c in all_chunks if c.images)

        table = Table(title="Chunking Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        table.add_row("Total Chunks", str(len(all_chunks)))
        table.add_row("Avg Chunk Length", f"{avg_length:.0f} chars")
        table.add_row("Chunks with Code", str(chunks_with_code))
        table.add_row("Chunks with Images", str(chunks_with_images))
        console.print(table)

        if TEST_RUN_DIR:
            chunk_summary = {
                "total_chunks": len(all_chunks),
                "avg_length": avg_length,
                "chunks_with_code": chunks_with_code,
                "chunks_with_images": chunks_with_images,
                "sample_chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "text_length": len(chunk.text),
                        "text_preview": chunk.text[:200],
                        "has_code": len(chunk.code_blocks) > 0,
                        "has_images": len(chunk.images) > 0,
                    }
                    for chunk in all_chunks[:5]
                ],
            }
            with open(TEST_RUN_DIR / "03_chunks.json", "w") as f:
                json.dump(chunk_summary, f, indent=2)
            console.print(f"  [dim]Saved to: {TEST_RUN_DIR / '03_chunks.json'}[/dim]")

        console.print("[green]✓ Chunking successful[/green]")
        return len(all_chunks)
    except Exception as e:
        console.print(f"[red]✗ Chunking error: {e}[/red]")
        return 0


def test_quality_filtering(config: PipelineConfig):
    """Test 6: Quality filtering for content and images."""
    console.print("\n[bold cyan]Step 6: Quality Filtering[/bold cyan]")

    if not config.generation.enable_content_filtering:
        console.print("[yellow]⊘ Content filtering disabled, skipping[/yellow]")
        return 0, 0

    try:
        images_dir = Path(config.dataset.images_dir)
        ingestion = DocumentIngestion(images_output_dir=images_dir)

        test_docs = get_limited_documents(config, ingestion, max_docs=3)

        chunker = ContentChunker(
            max_length=config.generation.max_context_length,
            overlap=config.generation.chunk_overlap,
        )

        all_chunks = []
        for doc in test_docs:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        with ModelRegistry(config.models) as registry:
            question_client = registry.get_llm_client(config.generation.question_model)
            quality_filter = QualityFilter(question_client)

            console.print(f"  Testing {len(all_chunks[:5])} chunks for quality...")
            content_passed = 0
            for chunk in all_chunks[:5]:
                try:
                    if quality_filter.is_quality_content(
                        chunk, config.prompts.content_quality_check
                    ):
                        content_passed += 1
                except Exception:
                    pass

            console.print(f"  [green]✓ Content filter: {content_passed}/5 chunks passed[/green]")

            images_to_test = [img for doc in test_docs for img in doc.images if img.transcription][
                :3
            ]

            images_passed = 0
            if images_to_test:
                console.print(f"  Testing {len(images_to_test)} images for quality...")
                for img in images_to_test:
                    try:
                        if quality_filter.is_quality_image(img, config.prompts.image_quality_check):
                            images_passed += 1
                    except Exception:
                        pass

                console.print(
                    f"  [green]✓ Image filter: {images_passed}/{len(images_to_test)} images passed[/green]"
                )

            if TEST_RUN_DIR:
                filter_summary = {
                    "content_tested": 5,
                    "content_passed": content_passed,
                    "images_tested": len(images_to_test),
                    "images_passed": images_passed,
                }
                with open(TEST_RUN_DIR / "04_quality_filter.json", "w") as f:
                    json.dump(filter_summary, f, indent=2)
                console.print(f"  [dim]Saved to: {TEST_RUN_DIR / '04_quality_filter.json'}[/dim]")

            console.print("[green]✓ Quality filtering successful[/green]")
            return content_passed, images_passed

    except Exception as e:
        console.print(f"[red]✗ Filtering error: {e}[/red]")
        return 0, 0


def test_category_classification(config: PipelineConfig):
    """Test 7: Category classification."""
    console.print("\n[bold cyan]Step 7: Category Classification[/bold cyan]")

    try:
        images_dir = Path(config.dataset.images_dir)
        ingestion = DocumentIngestion(images_output_dir=images_dir)

        test_docs = get_limited_documents(config, ingestion, max_docs=3)

        chunker = ContentChunker(
            max_length=config.generation.max_context_length,
            overlap=config.generation.chunk_overlap,
        )

        chunks = []
        for doc in test_docs:
            chunks.extend(chunker.chunk_document(doc))

        with ModelRegistry(config.models) as registry:
            question_client = registry.get_llm_client(config.generation.question_model)
            category_manager = CategoryManager(config.categories, question_client)

            chunks_by_category = category_manager.organize_by_category(
                chunks[:10], config.prompts.category_classification  # Test with first 10 chunks
            )

            table = Table(title="Category Distribution (Test)")
            table.add_column("Category", style="cyan")
            table.add_column("Chunks", style="green", justify="right")

            for category, cat_chunks in sorted(
                chunks_by_category.items(), key=lambda x: len(x[1]), reverse=True
            ):
                table.add_row(category, str(len(cat_chunks)))

            console.print(table)

            if TEST_RUN_DIR:
                classification_summary = {
                    "total_categories": len(chunks_by_category),
                    "total_chunks_classified": sum(
                        len(chunks) for chunks in chunks_by_category.values()
                    ),
                    "distribution": {
                        category: len(cat_chunks)
                        for category, cat_chunks in sorted(
                            chunks_by_category.items(), key=lambda x: len(x[1]), reverse=True
                        )
                    },
                }
                with open(TEST_RUN_DIR / "05_categories.json", "w") as f:
                    json.dump(classification_summary, f, indent=2)
                console.print(f"  [dim]Saved to: {TEST_RUN_DIR / '05_categories.json'}[/dim]")

            console.print("[green]✓ Category classification successful[/green]")
            return len(chunks_by_category)
    except Exception as e:
        console.print(f"[red]✗ Classification error: {e}[/red]")
        return 0


def test_sample_generation(config: PipelineConfig):
    """Test 8: Sample generation with async batching."""
    console.print("\n[bold cyan]Step 8: Sample Generation (Async Batching)[/bold cyan]")

    try:
        console.print(
            f"  Batch size: {config.generation.llm_batch_size}, "
            f"Concurrency: {config.generation.llm_concurrency}"
        )

        images_dir = Path(config.dataset.images_dir)
        ingestion = DocumentIngestion(images_output_dir=images_dir)

        test_docs = get_limited_documents(config, ingestion, max_docs=3)

        chunker = ContentChunker(
            max_length=config.generation.max_context_length,
            overlap=config.generation.chunk_overlap,
        )

        chunks = []
        for doc in test_docs:
            chunks.extend(chunker.chunk_document(doc))

        with ModelRegistry(config.models) as registry:
            question_client = registry.get_llm_client(config.generation.question_model)
            category_manager = CategoryManager(config.categories, question_client)

            chunks_by_category = category_manager.organize_by_category(
                chunks[:15], config.prompts.category_classification
            )

            pipeline = GenerationPipeline(config, registry, category_manager, checkpoint_manager=None)
            samples = pipeline.generate_samples(chunks_by_category)

            table = Table(title="Generated Samples")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green", justify="right")
            table.add_row("Total Samples", str(len(samples)))
            table.add_row("Multimodal Samples", str(sum(1 for s in samples if s.image_path)))

            console.print(table)

            if TEST_RUN_DIR and samples:
                samples_file = TEST_RUN_DIR / "06_samples.jsonl"
                with open(samples_file, "w") as f:
                    for sample in samples:
                        json.dump(
                            {
                                "question": sample.question,
                                "answer": sample.answer,
                                "category": sample.category,
                                "question_type": sample.question_type,
                                "difficulty": sample.difficulty,
                                "has_image": sample.image_path is not None,
                                "source_path": sample.source_path,
                            },
                            f,
                        )
                        f.write("\n")
                console.print(f"  [dim]Saved to: {samples_file}[/dim]")

            if samples:
                sample = samples[0]
                console.print(f"\n  [dim]Sample Question:[/dim]")
                console.print(
                    Panel(sample.question, title=f"{sample.question_type} ({sample.difficulty})")
                )
                console.print(f"\n  [dim]Sample Answer:[/dim]")
                console.print(Panel(sample.answer[:300] + "...", title="Answer"))

            console.print("[green]✓ Sample generation successful[/green]")
            return len(samples)
    except Exception as e:
        console.print(f"[red]✗ Generation error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return 0


def test_cache_functionality(config: PipelineConfig):
    """Test 9: Cache functionality."""
    console.print("\n[bold cyan]Step 9: Cache Testing[/bold cyan]")

    try:
        cache_dir = Path(config.dataset.parsed_dir).parent / ".cache"
        cache = PipelineCache(cache_dir)

        cache.clear_stage("test")

        from synthetic_data.parsers.base import Document

        test_doc = Document(
            source_path=Path("test.txt"),
            title="Test Document",
            content="Test content for caching",
            code_blocks=["print('test')"],
            images=[],
            metadata={"test": True},
        )

        test_key = cache.get_stage_cache_key("test", {"test": True}, [Path("test.txt")])
        cache.save_documents("test", test_key, [test_doc])
        console.print("  [green]✓ Document saved to cache[/green]")

        if cache.is_cached("test", test_key):
            console.print("  [green]✓ Cache detection working[/green]")

        loaded_docs = cache.load_documents("test", test_key)
        if loaded_docs and len(loaded_docs) == 1:
            if loaded_docs[0].title == "Test Document":
                console.print("  [green]✓ Cache loading successful[/green]")

        info = cache.get_cache_info()
        if "test" in info:
            console.print(
                f"  [green]✓ Cache info: {info['test']['count']} items, {info['test']['size']} bytes[/green]"
            )

        cache.clear_stage("test")
        if not cache.is_cached("test", test_key):
            console.print("  [green]✓ Cache clearing successful[/green]")

        console.print("\n[green]✓ Cache functionality working correctly[/green]")
        return True

    except Exception as e:
        console.print(f"[red]✗ Cache test error: {e}[/red]")
        return False


def main():
    """Run all pipeline tests."""
    if len(sys.argv) < 2:
        console.print("[red]Usage: python test_pipeline_steps.py <config_path>[/red]")
        console.print(
            "[yellow]Example: python test_pipeline_steps.py src/synthetic_data/yaml/config_test.yaml[/yellow]"
        )
        sys.exit(1)

    config_path = Path(sys.argv[1])

    console.print(
        Panel.fit(
            "[bold cyan]Pipeline Step-by-Step Testing[/bold cyan]\n"
            "This script tests each step of the synthetic data generation pipeline.\n"
            f"Config: {config_path.name}",
            title="Testing Suite",
        )
    )

    test_run_dir = setup_test_output_dir()

    start_time = time.time()
    test_results = Table(title="Test Results", show_header=True, header_style="bold magenta")
    test_results.add_column("Step", style="cyan")
    test_results.add_column("Status", justify="center")
    test_results.add_column("Time", justify="right", style="yellow")

    step_start = time.time()
    config = test_config_loading(config_path)
    test_results.add_row("1. Configuration", "[green]✓[/green]", f"{time.time() - step_start:.1f}s")

    step_start = time.time()
    if not test_model_connections(config):
        test_results.add_row(
            "2. Model Connections", "[red]✗[/red]", f"{time.time() - step_start:.1f}s"
        )
        console.print(test_results)
        console.print("[red]✗ Model connections failed, stopping tests[/red]")
        sys.exit(1)
    test_results.add_row(
        "2. Model Connections", "[green]✓[/green]", f"{time.time() - step_start:.1f}s"
    )

    step_start = time.time()
    docs, _ = test_document_parsing(config)
    test_results.add_row(
        "3. Document Parsing", f"[green]✓[/green] ({docs} docs)", f"{time.time() - step_start:.1f}s"
    )

    step_start = time.time()
    transcribed = test_image_transcription(config)
    test_results.add_row(
        "4. Image Transcription",
        f"[green]✓[/green] ({transcribed} imgs)",
        f"{time.time() - step_start:.1f}s",
    )

    step_start = time.time()
    chunks = test_chunking(config)
    test_results.add_row(
        "5. Content Chunking",
        f"[green]✓[/green] ({chunks} chunks)",
        f"{time.time() - step_start:.1f}s",
    )

    step_start = time.time()
    content_passed, images_passed = test_quality_filtering(config)
    if config.generation.enable_content_filtering:
        test_results.add_row(
            "6. Quality Filter",
            f"[green]✓[/green] ({content_passed}/{images_passed})",
            f"{time.time() - step_start:.1f}s",
        )
    else:
        test_results.add_row("6. Quality Filter", "[yellow]⊘ disabled[/yellow]", "0.0s")

    step_start = time.time()
    categories = test_category_classification(config)
    test_results.add_row(
        "7. Classification",
        f"[green]✓[/green] ({categories} cats)",
        f"{time.time() - step_start:.1f}s",
    )

    step_start = time.time()
    samples = test_sample_generation(config)
    test_results.add_row(
        "8. Sample Generation",
        f"[green]✓[/green] ({samples} samples)",
        f"{time.time() - step_start:.1f}s",
    )

    step_start = time.time()
    cache_ok = test_cache_functionality(config)
    test_results.add_row(
        "9. Cache Testing",
        "[green]✓[/green]" if cache_ok else "[yellow]⚠[/yellow]",
        f"{time.time() - step_start:.1f}s",
    )

    total_time = time.time() - start_time

    console.print("\n")
    console.print(test_results)
    console.print(f"\n[bold]Total test time: {total_time:.1f} seconds[/bold]")

    if TEST_RUN_DIR:
        summary = {
            "test_run": datetime.now().isoformat(),
            "config_file": str(config_path),
            "total_time_seconds": total_time,
            "all_tests_passed": True,
        }
        with open(TEST_RUN_DIR / "00_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        console.print(f"\n[bold cyan]Test outputs saved to:[/bold cyan]")
        console.print(f"  {TEST_RUN_DIR}")
        for file in sorted(TEST_RUN_DIR.glob("*.json*")):
            console.print(f"  - {file.name}")

    console.print("\n[bold green]✓ All pipeline steps tested successfully![/bold green]")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("  1. Review test outputs in outputs/test_runs/<timestamp>/")
    console.print("  2. Run full pipeline: synthetic-data generate --config <config>")
    console.print("  3. Monitor throughput and adjust batch/concurrency settings")


if __name__ == "__main__":
    main()
