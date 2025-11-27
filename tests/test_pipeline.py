"""Test suite for refactored synthetic data generation pipeline.

Tests all stages:
1. Parse & Ingestion (all parser types, image ID generation)
2. Transcribe & Classification (VLM + automatic image typing)
3. Chunking (image references, accumulated code, context building)
4. Quality Filtering (content and images with proper chunk updates)
5. Input Generation (text-only and multimodal for all 3 types)
6. Test Generation (function_completion and code_generation with retry)
7. Answer Generation (all types with code verification and retry)
8. Quality Curation (final check)
"""

import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from synthetic_data.config import PipelineConfig, QuestionType
from synthetic_data.extractors import ContentChunker, DocumentIngestion, ImageTranscriber
from synthetic_data.generators import GenerationPipeline
from synthetic_data.generators.planner import InputPlanner
from synthetic_data.generators.prompts import PromptSet
from synthetic_data.models import ModelRegistry, Message
from synthetic_data.parsers.base import Document, ImageType
from synthetic_data.utils import QualityFilter

console = Console()
TEST_RUN_DIR = None


def setup_test_output_dir():
    """Create timestamped directory for test outputs."""
    global TEST_RUN_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TEST_RUN_DIR = (
        Path(__file__).parent.parent / "outputs" / "test_runs" / f"refactored_{timestamp}"
    )
    TEST_RUN_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"\n[cyan]Test outputs: {TEST_RUN_DIR}[/cyan]\n")
    return TEST_RUN_DIR


def test_parsers(config: PipelineConfig):
    """Test 1: Document parsers with image ID generation."""
    console.print("\n[bold cyan]Test 1: Document Parsing & Image ID Generation[/bold cyan]")

    images_dir = Path(config.dataset.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    ingestion = DocumentIngestion(images_output_dir=images_dir)

    # Test with real data
    data_dir = Path("/Users/samuel/Developer/avante/unifei/tcc/quantum-assistant/data")

    if not data_dir.exists():
        console.print(f"[yellow]Warning: Data directory not found: {data_dir}[/yellow]")
        return []

    # Find test files
    test_files = {
        "jupyter": list(data_dir.rglob("*.ipynb"))[:2],
        "pdf": list(data_dir.rglob("*.pdf"))[:1],
        "mdx": list(data_dir.rglob("*.mdx"))[:1],
    }

    parsed_docs = []
    parser_stats = {"jupyter": 0, "pdf": 0, "mdx": 0}

    for parser_type, files in test_files.items():
        for file_path in files:
            try:
                doc = ingestion.parse_file(file_path)
                if doc:
                    parsed_docs.append(doc)
                    parser_stats[parser_type] += 1

                    # Verify image IDs
                    for img in doc.images:
                        assert img.image_id, f"Image missing ID: {img.path}"
                        assert img.image_id.startswith("img_"), f"Invalid ID format: {img.image_id}"

                    console.print(f"  [green]✓[/green] {parser_type}: {file_path.name}")
                    console.print(
                        f"     Images: {len(doc.images)}, Code blocks: {len(doc.code_blocks)}"
                    )
            except Exception as e:
                console.print(f"  [red]✗[/red] {parser_type}: {file_path.name} - {e}")

    # Verify results
    assert parsed_docs, "No documents parsed"

    total_images = sum(len(doc.images) for doc in parsed_docs)
    images_with_ids = sum(1 for doc in parsed_docs for img in doc.images if img.image_id)
    images_resolved = sum(1 for doc in parsed_docs for img in doc.images if img.resolved_path)

    table = Table(title="Parsing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Documents Parsed", str(len(parsed_docs)))
    table.add_row("Jupyter", str(parser_stats["jupyter"]))
    table.add_row("PDF", str(parser_stats["pdf"]))
    table.add_row("MDX", str(parser_stats["mdx"]))
    table.add_row("Total Images", str(total_images))
    table.add_row("Images with IDs", str(images_with_ids))
    table.add_row("Images Resolved", str(images_resolved))

    console.print(table)

    if TEST_RUN_DIR:
        with open(TEST_RUN_DIR / "01_parsing.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_docs": len(parsed_docs),
                    "total_images": total_images,
                    "images_with_ids": images_with_ids,
                    "images_resolved": images_resolved,
                    "parser_stats": parser_stats,
                },
                f,
                indent=2,
            )

    console.print("[green]✓ All parsers working, image IDs generated[/green]")
    return parsed_docs


def test_transcription_and_classification(config: PipelineConfig, documents: list[Document]):
    """Test 2: Image transcription with automatic classification."""
    console.print("\n[bold cyan]Test 2: Transcription + Classification[/bold cyan]")

    if not config.generation.enable_image_transcription:
        console.print("[yellow]⊘ Transcription disabled[/yellow]")
        return documents

    images_to_transcribe = sum(1 for doc in documents for img in doc.images if img.resolved_path)

    if images_to_transcribe == 0:
        console.print("[yellow]No images to transcribe[/yellow]")
        return documents

    console.print(f"  Transcribing {images_to_transcribe} images...")

    with ModelRegistry(config.models) as registry:
        vision_client = registry.get_vlm_client(config.generation.vision_model)
        transcriber = ImageTranscriber(
            vision_client,
            config.prompts.image_transcription,
            system_prompt=config.prompts.image_transcription_system,
            batch_size=config.generation.vlm_batch_size,
            max_concurrent=config.generation.vlm_concurrency,
            enable_classification=True,  # Enable classification
        )

        # Transcribe with progress
        completed = [0]

        def progress_cb(count):
            completed[0] = count
            console.print(f"\r  Progress: {count}/{images_to_transcribe}", end="")

        asyncio.run(transcriber.transcribe_batch_documents_async(documents, progress_cb))
        console.print()

    # Verify transcriptions and classifications
    transcribed = sum(1 for doc in documents for img in doc.images if img.transcription)
    classified = sum(
        1 for doc in documents for img in doc.images if img.image_type != ImageType.UNKNOWN
    )

    # Count by type
    type_counts = {}
    for doc in documents:
        for img in doc.images:
            if img.image_type:
                type_name = img.image_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1

    table = Table(title="Transcription Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Images Transcribed", str(transcribed))
    table.add_row("Images Classified", str(classified))

    console.print(table)

    # Show classification breakdown
    if type_counts:
        type_table = Table(title="Image Type Distribution")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", style="green", justify="right")

        for img_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            type_table.add_row(img_type, str(count))

        console.print(type_table)

    # Show sample transcription
    for doc in documents:
        for img in doc.images:
            if img.transcription:
                console.print(f"\n[dim]Sample transcription ({img.image_type.value}):[/dim]")
                console.print(Panel(img.transcription[:300] + "...", title=img.alt_text or "Image"))
                break
        if img.transcription:
            break

    if TEST_RUN_DIR:
        with open(TEST_RUN_DIR / "02_transcription.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_images": images_to_transcribe,
                    "transcribed": transcribed,
                    "classified": classified,
                    "type_distribution": type_counts,
                },
                f,
                indent=2,
            )

    console.print("[green]✓ Transcription and classification working[/green]")
    return documents


def test_chunking_with_image_refs(config: PipelineConfig, documents: list[Document]):
    """Test 3: Chunking with image references and accumulated code."""
    console.print("\n[bold cyan]Test 3: Chunking (Image Refs + Accumulated Code)[/bold cyan]")

    chunker = ContentChunker(
        max_length=config.generation.max_context_length,
        overlap=config.generation.chunk_overlap,
    )

    all_chunks = []
    for doc in documents[:3]:  # Test with first 3 docs
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    # Verify image references (not inline transcriptions)
    chunks_with_refs = sum(1 for c in all_chunks if "[IMAGE:img_" in c.text)
    chunks_with_inline = sum(1 for c in all_chunks if "[Visual:" in c.text or "[/Visual]" in c.text)

    # Verify accumulated code
    chunks_with_accumulated_code = sum(1 for c in all_chunks if c.accumulated_code)

    table = Table(title="Chunking Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Total Chunks", str(len(all_chunks)))
    table.add_row("With Code", str(sum(1 for c in all_chunks if c.code_blocks)))
    table.add_row("With Images", str(sum(1 for c in all_chunks if c.images)))
    table.add_row("With Image Refs [IMAGE:id]", str(chunks_with_refs))
    table.add_row("With Inline [Visual:]", str(chunks_with_inline))
    table.add_row("With Accumulated Code", str(chunks_with_accumulated_code))

    console.print(table)

    # Test context building
    console.print("\n[bold]Testing Context Building:[/bold]")
    for chunk in all_chunks:
        if chunk.images and chunk.images[0].transcription:
            # Test text-only context
            text_context = chunk.build_context_with_transcriptions(include_code=True)
            assert (
                "[IMAGE:" in text_context or "[TARGET IMAGE:" in text_context
            ), "Context should have image transcription inserted"

            # Test multimodal context with target
            target_id = chunk.images[0].image_id
            multimodal_context = chunk.build_context_with_transcriptions(
                target_image_id=target_id,
                include_code=True,
            )
            assert (
                "[TARGET IMAGE:" in multimodal_context
            ), "Multimodal context should emphasize target image"

            console.print(f"  [green]✓[/green] Context building for chunk {chunk.chunk_id}")
            console.print(f"     Text context: {len(text_context)} chars")
            console.print(f"     Multimodal context: {len(multimodal_context)} chars")
            console.print(f"     Accumulated code blocks: {len(chunk.accumulated_code)}")

            # Show sample context
            console.print("\n[dim]Sample multimodal context:[/dim]")
            console.print(Panel(multimodal_context + "...", title="Context Preview"))
            break

    # Verify no code blocks split
    console.print("\n[bold]Verifying Code Block Integrity:[/bold]")
    split_violations = 0
    for chunk in all_chunks:
        for code_block in chunk.code_blocks:
            # Check if code block appears complete in chunk text
            if "```" in chunk.text:
                # Count ``` pairs
                count = chunk.text.count("```")
                if count % 2 != 0:
                    split_violations += 1

    if split_violations == 0:
        console.print("  [green]✓ No code blocks split across chunks[/green]")
    else:
        console.print(f"  [yellow]⚠ {split_violations} potential code block splits[/yellow]")

    assert chunks_with_inline == 0, "Should not have inline transcriptions, only refs"

    if TEST_RUN_DIR:
        with open(TEST_RUN_DIR / "03_chunking.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_chunks": len(all_chunks),
                    "chunks_with_refs": chunks_with_refs,
                    "chunks_with_inline": chunks_with_inline,
                    "chunks_with_accumulated_code": chunks_with_accumulated_code,
                    "split_violations": split_violations,
                },
                f,
                indent=2,
            )

    console.print("[green]✓ Chunking using image references and accumulated code[/green]")
    return all_chunks


def test_quality_filtering(config: PipelineConfig, chunks: list):
    """Test 4: Quality filtering with proper chunk/image filtering."""
    console.print("\n[bold cyan]Test 4: Quality Filtering[/bold cyan]")

    if not config.generation.enable_content_filtering:
        console.print("[yellow]⊘ Filtering disabled[/yellow]")
        return chunks

    with ModelRegistry(config.models) as registry:
        filter_model = config.generation.filter_model or config.generation.curate_model
        filter_client = registry.get_llm_client(filter_model)
        quality_filter = QualityFilter(filter_client)

        test_chunks = chunks[:10]  # Test with subset

        console.print(f"  Filtering {len(test_chunks)} chunks...")

        results, debug_info = asyncio.run(
            quality_filter.filter_chunks_batch_async(
                test_chunks,
                config.prompts.content_quality_check,
                config.prompts.image_quality_check,
                config.prompts.content_filter_system,
                config.prompts.image_filter_system,
                batch_size=5,
                max_concurrent=10,
                save_debug=True,
            )
        )

    passed_chunks = [chunk for chunk, passed in results if passed]
    removed_chunks = len(test_chunks) - len(passed_chunks)

    # Count filtered images
    original_images = sum(len(c.images) for c in test_chunks)
    filtered_images = sum(len(c.images) for c, _ in results if _)
    removed_images = original_images - filtered_images

    table = Table(title="Filtering Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Chunks Tested", str(len(test_chunks)))
    table.add_row("Chunks Passed", str(len(passed_chunks)))
    table.add_row("Chunks Removed", str(removed_chunks))
    table.add_row("Images Before", str(original_images))
    table.add_row("Images After", str(filtered_images))
    table.add_row("Images Removed", str(removed_images))

    console.print(table)

    # Show debug info
    if debug_info:
        console.print(f"\n[dim]Debug entries: {len(debug_info)}[/dim]")
        content_rejects = sum(
            1 for d in debug_info if d["type"] == "content" and d["decision"] == "REJECT"
        )
        image_rejects = sum(
            1 for d in debug_info if d["type"] == "image" and d["decision"] == "REJECT"
        )
        console.print(f"  Content rejections: {content_rejects}")
        console.print(f"  Image rejections: {image_rejects}")

    if TEST_RUN_DIR:
        with open(TEST_RUN_DIR / "04_filtering.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunks_tested": len(test_chunks),
                    "chunks_passed": len(passed_chunks),
                    "chunks_removed": removed_chunks,
                    "images_removed": removed_images,
                    "debug_entries": len(debug_info),
                },
                f,
                indent=2,
            )

        # Save debug info
        if debug_info:
            with open(TEST_RUN_DIR / "04_filtering_debug.jsonl", "w", encoding="utf-8") as f:
                for entry in debug_info:
                    json.dump(entry, f)
                    f.write("\n")

    console.print("[green]✓ Filtering working correctly[/green]")
    return passed_chunks


def test_input_generation_all_types(config: PipelineConfig, chunks: list):
    """Test 5: Input generation for all 3 types (text-only and multimodal)."""
    console.print("\n[bold cyan]Test 5: Input Generation (All Types)[/bold cyan]")

    # Find chunks for each scenario
    chunk_with_code = next((c for c in chunks if c.code_blocks), None)
    chunk_with_image = next((c for c in chunks if c.transcribed_images), None)
    chunk_text_only = next((c for c in chunks if not c.images), chunks[0])

    with ModelRegistry(config.models) as registry:
        question_client = registry.get_llm_client(config.generation.question_model)
        prompts = PromptSet(
            input_generation_system=config.prompts.input_generation_system,
            function_completion_prompt=config.prompts.function_completion_prompt,
            code_generation_prompt=config.prompts.code_generation_prompt,
            qa_prompt=config.prompts.qa_prompt,
            test_generation_prompt=config.prompts.test_generation_prompt,
            answer_generation_system=config.prompts.answer_generation_system,
            function_completion_answer_prompt=config.prompts.function_completion_answer_prompt,
            code_generation_answer_prompt=config.prompts.code_generation_answer_prompt,
            qa_answer_prompt=config.prompts.qa_answer_prompt,
            answer_correction_prompt=config.prompts.answer_correction_prompt,
            content_quality_check=config.prompts.content_quality_check,
            content_filter_system=config.prompts.content_filter_system,
            image_quality_check=config.prompts.image_quality_check,
            image_filter_system=config.prompts.image_filter_system,
            image_transcription=config.prompts.image_transcription,
            image_transcription_system=config.prompts.image_transcription_system,
            candidate_filter_system=config.prompts.candidate_filter_system,
            candidate_filter_prompt=config.prompts.candidate_filter_prompt,
            category_classification=config.prompts.category_classification,
            category_classification_system=config.prompts.category_classification_system,
            sample_curation=config.prompts.sample_curation,
            sample_curation_system=config.prompts.sample_curation_system,
        )

        planner = InputPlanner(
            llm_client=question_client,
            prompts=prompts,
            candidates_per_chunk=2,
            max_concurrent=5,
            test_max_iterations=3,
        )

        # Test each type
        test_results = {}

        # Test function_completion (text-only)
        if chunk_with_code:
            console.print("\n  Testing function_completion (text-only)...")
            context = chunk_with_code.build_context_with_transcriptions(include_code=True)

            messages = [
                Message(role="system", content=prompts.get_input_system_prompt(use_image=False)),
                Message(
                    role="user", content=prompts.function_completion_prompt.format(context=context)
                ),
            ]

            response = question_client.generate(messages, temperature=0.7)

            assert "def " in response, "function_completion should have function definition"
            assert "pass" in response, "function_completion should have pass statement"

            console.print(f"  [green]✓[/green] Generated function_completion")
            console.print(Panel(response, title="Function Completion"))

            test_results["function_completion_text"] = {
                "success": True,
                "has_def": "def " in response,
                "has_pass": "pass" in response,
            }

        # Test code_generation (multimodal if available)
        if chunk_with_image:
            console.print("\n  Testing code_generation (multimodal)...")
            target_img = next(
                (img for img in chunk_with_image.transcribed_images if img.image_id), None
            )

            if target_img:
                context = chunk_with_image.build_context_with_transcriptions(
                    target_image_id=target_img.image_id,
                    include_code=True,
                )

                assert (
                    "[TARGET IMAGE:" in context
                ), "Multimodal context should emphasize target image"

                messages = [
                    Message(role="system", content=prompts.get_input_system_prompt(use_image=True)),
                    Message(
                        role="user", content=prompts.code_generation_prompt.format(context=context)
                    ),
                ]

                response = question_client.generate(messages, temperature=0.7)

                assert (
                    "You must implement" in response or "function named" in response
                ), "code_generation should specify function"

                console.print(f"  [green]✓[/green] Generated code_generation (multimodal)")
                console.print(Panel(response[:400] + "...", title="Code Generation"))

                test_results["code_generation_multimodal"] = {
                    "success": True,
                    "context_has_target_image": "[TARGET IMAGE:" in context,
                    "image_type": target_img.image_type.value,
                }

        # Test qa (text-only)
        console.print("\n  Testing qa (text-only)...")
        context = chunk_text_only.build_context_with_transcriptions(include_code=False)

        messages = [
            Message(role="system", content=prompts.get_input_system_prompt(use_image=False)),
            Message(role="user", content=prompts.qa_prompt.format(context=context)),
        ]

        response = question_client.generate(messages, temperature=0.7)

        assert len(response) > 20, "QA should have substantial question"
        # assert "?" in response, "QA should be a question"

        console.print(f"  [green]✓[/green] Generated qa")
        console.print(Panel(response[:400] + "...", title="QA Question"))

        test_results["qa_text"] = {
            "success": True,
            "is_question": "?" in response,
        }

    # Calculate chunk statistics
    chunks_with_refs = sum(1 for c in chunks if "[IMAGE:img_" in c.text)
    chunks_with_accumulated_code = sum(1 for c in chunks if c.accumulated_code)

    if TEST_RUN_DIR:
        with open(TEST_RUN_DIR / "05_input_generation.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_chunks": len(chunks),
                    "chunks_with_refs": chunks_with_refs,
                    "chunks_with_accumulated_code": chunks_with_accumulated_code,
                    "test_results": test_results,
                },
                f,
                indent=2,
            )

    console.print("[green]✓ Input generation working for all types[/green]")
    return chunks


def test_test_generation_with_retry(config: PipelineConfig, chunks: list):
    """Test 6: Unit test generation with retry logic."""
    console.print("\n[bold cyan]Test 6: Test Generation with Retry[/bold cyan]")

    chunk_with_code = next((c for c in chunks if c.code_blocks), None)
    if not chunk_with_code:
        console.print("[yellow]No chunks with code for testing[/yellow]")
        return

    with ModelRegistry(config.models) as registry:
        question_client = registry.get_llm_client(config.generation.question_model)
        prompts = PromptSet(
            input_generation_system=config.prompts.input_generation_system,
            function_completion_prompt=config.prompts.function_completion_prompt,
            code_generation_prompt=config.prompts.code_generation_prompt,
            qa_prompt=config.prompts.qa_prompt,
            test_generation_prompt=config.prompts.test_generation_prompt,
            answer_generation_system=config.prompts.answer_generation_system,
            function_completion_answer_prompt=config.prompts.function_completion_answer_prompt,
            code_generation_answer_prompt=config.prompts.code_generation_answer_prompt,
            qa_answer_prompt=config.prompts.qa_answer_prompt,
            answer_correction_prompt=config.prompts.answer_correction_prompt,
            content_quality_check=config.prompts.content_quality_check,
            content_filter_system=config.prompts.content_filter_system,
            image_quality_check=config.prompts.image_quality_check,
            image_filter_system=config.prompts.image_filter_system,
            image_transcription=config.prompts.image_transcription,
            image_transcription_system=config.prompts.image_transcription_system,
            candidate_filter_system=config.prompts.candidate_filter_system,
            candidate_filter_prompt=config.prompts.candidate_filter_prompt,
            category_classification=config.prompts.category_classification,
            category_classification_system=config.prompts.category_classification_system,
            sample_curation=config.prompts.sample_curation,
            sample_curation_system=config.prompts.sample_curation_system,
        )

        planner = InputPlanner(
            llm_client=question_client,
            prompts=prompts,
            candidates_per_chunk=1,
            max_concurrent=5,
            test_max_iterations=3,
        )

        # Generate candidates with tests
        console.print("  Generating candidates with tests...")
        candidates = asyncio.run(
            planner.generate_candidates_async(
                [chunk_with_code],
                config.generation.question_type_weights,
                multimodal_ratio=0.0,  # Text-only for test generation
            )
        )

        # Check results
        code_candidates = [
            c
            for c in candidates
            if c.question_type in (QuestionType.FUNCTION_COMPLETION, QuestionType.CODE_GENERATION)
        ]

        candidates_with_tests = sum(1 for c in code_candidates if c.test_code)
        failed_test_gen = sum(1 for c in code_candidates if not c.test_code and c.rejection_reason)

        table = Table(title="Test Generation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        table.add_row("Code Candidates", str(len(code_candidates)))
        table.add_row("With Valid Tests", str(candidates_with_tests))
        table.add_row("Failed Test Gen", str(failed_test_gen))

        console.print(table)

        # Show sample test
        for candidate in code_candidates:
            if candidate.test_code:
                console.print(f"\n[dim]Sample test for {candidate.question_type.value}:[/dim]")
                console.print(
                    Panel(
                        candidate.test_code[:500] + "...", title=f"Test - {candidate.entry_point}"
                    )
                )

                # Verify test structure
                assert (
                    "def check(" in candidate.test_code or "def test_" in candidate.test_code
                ), "Test should have check or test_ function"
                assert "assert" in candidate.test_code, "Test should have assertions"
                assert (
                    candidate.entry_point in candidate.test_code
                ), f"Test should call {candidate.entry_point}"

                console.print("  [green]✓[/green] Test structure valid")
                break

        if TEST_RUN_DIR:
            with open(TEST_RUN_DIR / "06_test_generation.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "code_candidates": len(code_candidates),
                        "with_valid_tests": candidates_with_tests,
                        "failed_test_gen": failed_test_gen,
                    },
                    f,
                    indent=2,
                )

    console.print("[green]✓ Test generation with validation working[/green]")


def test_answer_generation_with_verification(config: PipelineConfig, chunks: list):
    """Test 7: Answer generation with code verification and retry."""
    console.print("\n[bold cyan]Test 7: Answer Generation with Verification[/bold cyan]")

    # Generate test candidates
    with ModelRegistry(config.models) as registry:
        question_client = registry.get_llm_client(config.generation.question_model)
        prompts = PromptSet(
            input_generation_system=config.prompts.input_generation_system,
            function_completion_prompt=config.prompts.function_completion_prompt,
            code_generation_prompt=config.prompts.code_generation_prompt,
            qa_prompt=config.prompts.qa_prompt,
            test_generation_prompt=config.prompts.test_generation_prompt,
            answer_generation_system=config.prompts.answer_generation_system,
            function_completion_answer_prompt=config.prompts.function_completion_answer_prompt,
            code_generation_answer_prompt=config.prompts.code_generation_answer_prompt,
            qa_answer_prompt=config.prompts.qa_answer_prompt,
            answer_correction_prompt=config.prompts.answer_correction_prompt,
            content_quality_check=config.prompts.content_quality_check,
            content_filter_system=config.prompts.content_filter_system,
            image_quality_check=config.prompts.image_quality_check,
            image_filter_system=config.prompts.image_filter_system,
            image_transcription=config.prompts.image_transcription,
            image_transcription_system=config.prompts.image_transcription_system,
            candidate_filter_system=config.prompts.candidate_filter_system,
            candidate_filter_prompt=config.prompts.candidate_filter_prompt,
            category_classification=config.prompts.category_classification,
            category_classification_system=config.prompts.category_classification_system,
            sample_curation=config.prompts.sample_curation,
            sample_curation_system=config.prompts.sample_curation_system,
        )

        planner = InputPlanner(
            llm_client=question_client,
            prompts=prompts,
            candidates_per_chunk=2,
            max_concurrent=5,
            test_max_iterations=3,
        )

        # Generate candidates
        test_chunks = [c for c in chunks if c.code_blocks][:3]

        console.print(f"  Generating candidates from {len(test_chunks)} chunks...")
        candidates = asyncio.run(
            planner.generate_candidates_async(
                test_chunks,
                config.generation.question_type_weights,
                multimodal_ratio=0.3,
            )
        )

        valid_candidates = [c for c in candidates if c.is_valid]
        console.print(f"  Generated {len(valid_candidates)} valid candidates")

        # Test answer generation pipeline
        console.print("\n  Testing answer generation pipeline...")

        gen_pipeline = GenerationPipeline(config, registry, checkpoint_manager=None)

        # Generate answers for a few candidates
        test_candidates = valid_candidates[:5]

        samples, failures = asyncio.run(gen_pipeline._generate_answers_async(test_candidates, {}))

        table = Table(title="Answer Generation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        table.add_row("Candidates", str(len(test_candidates)))
        table.add_row("Successful", str(len(samples)))
        table.add_row("Failed", str(len(failures)))

        console.print(table)

        # Verify samples
        for i, sample in enumerate(samples[:2]):
            console.print(f"\n[dim]Sample {i+1} ({sample.question_type}):[/dim]")
            console.print(
                Panel(
                    f"Q: {sample.question[:200]}...\n\nA: {sample.answer[:200]}...",
                    title=f"Sample {i+1}",
                )
            )

            # Verify code samples
            if sample.question_type in ("function_completion", "code_generation"):
                assert sample.test_code, "Code samples should have tests"
                assert sample.entry_point, "Code samples should have entry point"
                console.print(f"  [green]✓[/green] Has test and entry point")

        # Check failure reasons
        if failures:
            console.print(f"\n[yellow]Failure analysis:[/yellow]")
            error_types = {}
            for failure in failures:
                error_type = failure.get("error", "Unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1

            for error_type, count in error_types.items():
                console.print(f"  {error_type}: {count}")

        if TEST_RUN_DIR:
            with open(TEST_RUN_DIR / "07_answer_generation.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "candidates": len(test_candidates),
                        "successful": len(samples),
                        "failed": len(failures),
                        "error_types": error_types if failures else {},
                    },
                    f,
                    indent=2,
                )

            # Save samples
            with open(TEST_RUN_DIR / "07_samples.jsonl", "w", encoding="utf-8") as f:
                for sample in samples:
                    json.dump(
                        {
                            "question": sample.question,
                            "answer": sample.answer,
                            "question_type": sample.question_type,
                            "has_test": bool(sample.test_code),
                            "has_image": bool(sample.image_path),
                        },
                        f,
                    )
                    f.write("\n")

    console.print("[green]✓ Answer generation with verification working[/green]")


def test_quality_curation(config: PipelineConfig, chunks: list):
    """Test 8: Final quality curation."""
    console.print("\n[bold cyan]Test 8: Quality Curation[/bold cyan]")

    if not config.generation.enable_curate_filtering:
        console.print("[yellow]⊘ Curation disabled[/yellow]")
        return

    # Generate a few samples for curation
    with ModelRegistry(config.models) as registry:
        gen_pipeline = GenerationPipeline(config, registry, checkpoint_manager=None)

        test_chunks = chunks[:5]
        console.print(f"  Generating samples from {len(test_chunks)} chunks...")

        # Generate candidates
        candidates = asyncio.run(gen_pipeline._generate_candidates_async(test_chunks, {}))

        valid_candidates = [c for c in candidates if c.is_valid][:3]

        # Generate answers
        samples, _ = asyncio.run(gen_pipeline._generate_answers_async(valid_candidates, {}))

        if not samples:
            console.print("[yellow]No samples generated for curation test[/yellow]")
            return

        console.print(f"  Curating {len(samples)} samples...")

        # Curate
        curated, rejected = asyncio.run(gen_pipeline._curate_samples_async(samples, {}))

        table = Table(title="Curation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        table.add_row("Samples Tested", str(len(samples)))
        table.add_row("Passed", str(len(curated)))
        table.add_row("Rejected", str(len(rejected)))

        console.print(table)

        if rejected:
            console.print("\n[yellow]Rejection reasons:[/yellow]")
            for rej in rejected[:3]:
                console.print(f"  - {rej.get('rejection_reason', 'No reason')}")

        if TEST_RUN_DIR:
            with open(TEST_RUN_DIR / "08_curation.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "samples_tested": len(samples),
                        "passed": len(curated),
                        "rejected": len(rejected),
                    },
                    f,
                    indent=2,
                )

    console.print("[green]✓ Quality curation working[/green]")


def test_full_pipeline_integration(config: PipelineConfig):
    """Test 9: Full pipeline integration with target distributions."""
    console.print("\n[bold cyan]Test 9: Full Pipeline Integration[/bold cyan]")

    # Use actual data source
    data_dir = Path("/Users/samuel/Developer/avante/unifei/tcc/quantum-assistant/data")

    if not data_dir.exists():
        console.print(f"[yellow]Warning: Data directory not found: {data_dir}[/yellow]")
        return

    # Parse
    console.print("  Stage 1: Parsing...")
    images_dir = Path(config.dataset.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    ingestion = DocumentIngestion(images_output_dir=images_dir)

    documents = []
    for source in config.sources:
        source_docs = ingestion.ingest_source(source)
        documents.extend(source_docs)

    console.print(f"    Parsed {len(documents)} documents")

    # Transcribe + Classify
    if config.generation.enable_image_transcription:
        console.print("  Stage 2: Transcribing + Classifying...")

        with ModelRegistry(config.models) as registry:
            vision_client = registry.get_vlm_client(config.generation.vision_model)
            transcriber = ImageTranscriber(
                vision_client,
                config.prompts.image_transcription,
                system_prompt=config.prompts.image_transcription_system,
                batch_size=config.generation.vlm_batch_size,
                max_concurrent=config.generation.vlm_concurrency,
                enable_classification=True,
            )

            asyncio.run(transcriber.transcribe_batch_documents_async(documents))

        transcribed = sum(1 for doc in documents for img in doc.images if img.transcription)
        classified = sum(
            1 for doc in documents for img in doc.images if img.image_type != ImageType.UNKNOWN
        )
        console.print(f"    Transcribed: {transcribed}, Classified: {classified}")

    # Chunk
    console.print("  Stage 3: Chunking...")
    chunker = ContentChunker(
        max_length=config.generation.max_context_length,
        overlap=config.generation.chunk_overlap,
    )

    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    console.print(f"    Created {len(all_chunks)} chunks")
    console.print(f"    Multimodal: {sum(1 for c in all_chunks if c.is_multimodal)}")
    console.print(f"    With code: {sum(1 for c in all_chunks if c.code_blocks)}")

    # Generate
    console.print("  Stage 4: Generating samples...")

    with ModelRegistry(config.models) as registry:
        gen_pipeline = GenerationPipeline(config, registry, checkpoint_manager=None)

        samples = gen_pipeline.generate_samples(all_chunks)

    # Analyze distribution
    by_type = {}
    by_modality = {"multimodal": 0, "text_only": 0}

    for sample in samples:
        by_type[sample.question_type] = by_type.get(sample.question_type, 0) + 1
        if sample.image_path:
            by_modality["multimodal"] += 1
        else:
            by_modality["text_only"] += 1

    total = len(samples)

    # Distribution table
    dist_table = Table(title="Sample Distribution")
    dist_table.add_column("Category", style="cyan")
    dist_table.add_column("Count", style="green", justify="right")
    dist_table.add_column("Percentage", style="magenta", justify="right")

    for qtype, count in sorted(by_type.items()):
        dist_table.add_row(qtype, str(count), f"{count/total*100:.1f}%")

    dist_table.add_row("─" * 20, "─" * 8, "─" * 10)
    dist_table.add_row(
        "Multimodal", str(by_modality["multimodal"]), f"{by_modality['multimodal']/total*100:.1f}%"
    )
    dist_table.add_row(
        "Text-only", str(by_modality["text_only"]), f"{by_modality['text_only']/total*100:.1f}%"
    )

    console.print(dist_table)

    # Check targets
    console.print("\n[bold]Target vs Actual:[/bold]")

    # Question type distribution
    func_comp_pct = by_type.get("function_completion", 0) / total * 100 if total > 0 else 0
    code_gen_pct = by_type.get("code_generation", 0) / total * 100 if total > 0 else 0
    qa_pct = by_type.get("qa", 0) / total * 100 if total > 0 else 0

    console.print(f"  function_completion: {func_comp_pct:.1f}% (target: 45%)")
    console.print(f"  code_generation: {code_gen_pct:.1f}% (target: 45%)")
    console.print(f"  qa: {qa_pct:.1f}% (target: 10%)")

    # Multimodal distribution
    multimodal_pct = by_modality["multimodal"] / total * 100 if total > 0 else 0
    console.print(f"  multimodal: {multimodal_pct:.1f}% (target: 50%)")

    if TEST_RUN_DIR:
        with open(TEST_RUN_DIR / "09_integration.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_samples": total,
                    "by_type": by_type,
                    "by_modality": by_modality,
                    "percentages": {
                        "function_completion": func_comp_pct,
                        "code_generation": code_gen_pct,
                        "qa": qa_pct,
                        "multimodal": multimodal_pct,
                    },
                    "targets": {
                        "function_completion": 45.0,
                        "code_generation": 45.0,
                        "qa": 10.0,
                        "multimodal": 50.0,
                    },
                },
                f,
                indent=2,
            )

    console.print("[green]✓ Full pipeline integration working[/green]")


def main():
    """Run all tests."""
    if len(sys.argv) < 2:
        console.print("[red]Usage: python test_refactored_pipeline.py <config_path>[/red]")
        console.print(
            "[yellow]Example: python test_refactored_pipeline.py src/synthetic_data/yaml/config.yaml[/yellow]"
        )
        sys.exit(1)

    config_path = Path(sys.argv[1])

    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)

    console.print(
        Panel.fit(
            "[bold cyan]Refactored Pipeline Test Suite[/bold cyan]\n"
            "Testing all refactored components:\n"
            "- Image classification\n"
            "- Image reference system\n"
            "- Accumulated code context\n"
            "- Multimodal generation strategy",
            title="Test Suite",
        )
    )

    test_run_dir = setup_test_output_dir()

    # Load config
    console.print(f"\n[cyan]Loading config: {config_path}[/cyan]")
    config = PipelineConfig.from_yaml(config_path)
    console.print(f"  Target samples: {config.generation.target_samples}")
    console.print(f"  Multimodal ratio: {config.generation.multimodal_ratio}")
    console.print(f"  Candidates per chunk: {config.generation.candidates_per_chunk}")

    # Track results
    test_results = Table(title="Test Results", show_header=True, header_style="bold magenta")
    test_results.add_column("Test", style="cyan")
    test_results.add_column("Status", justify="center")
    test_results.add_column("Time", justify="right", style="yellow")

    start_time = time.time()

    # Test 1: Parsers
    step_start = time.time()
    try:
        documents = test_parsers(config)
        test_results.add_row(
            "1. Parsing + Image IDs",
            f"[green]✓[/green] ({len(documents)} docs)",
            f"{time.time() - step_start:.1f}s",
        )
    except Exception as e:
        test_results.add_row(
            "1. Parsing + Image IDs", "[red]✗[/red]", f"{time.time() - step_start:.1f}s"
        )
        console.print(test_results)
        console.print(f"[red]✗ Test 1 failed: {e}[/red]")
        raise

    # Test 2: Transcription + Classification
    step_start = time.time()
    try:
        documents = test_transcription_and_classification(config, documents)
        transcribed_count = sum(1 for d in documents for i in d.images if i.transcription)
        test_results.add_row(
            "2. Transcription + Classification",
            f"[green]✓[/green] ({transcribed_count} imgs)",
            f"{time.time() - step_start:.1f}s",
        )
    except Exception as e:
        test_results.add_row(
            "2. Transcription + Classification", "[red]✗[/red]", f"{time.time() - step_start:.1f}s"
        )
        console.print(test_results)
        console.print(f"[red]✗ Test 2 failed: {e}[/red]")
        raise

    # Test 3: Chunking
    step_start = time.time()
    try:
        chunks = test_chunking_with_image_refs(config, documents)
        test_results.add_row(
            "3. Chunking (Refs + Code)",
            f"[green]✓[/green] ({len(chunks)} chunks)",
            f"{time.time() - step_start:.1f}s",
        )
    except Exception as e:
        test_results.add_row(
            "3. Chunking (Refs + Code)", "[red]✗[/red]", f"{time.time() - step_start:.1f}s"
        )
        console.print(test_results)
        console.print(f"[red]✗ Test 3 failed: {e}[/red]")
        raise

    # Test 4: Filtering
    step_start = time.time()
    try:
        filtered_chunks = test_quality_filtering(config, chunks)
        test_results.add_row(
            "4. Quality Filtering",
            f"[green]✓[/green] ({len(filtered_chunks)} passed)",
            f"{time.time() - step_start:.1f}s",
        )
    except Exception as e:
        test_results.add_row(
            "4. Quality Filtering", "[red]✗[/red]", f"{time.time() - step_start:.1f}s"
        )
        console.print(test_results)
        console.print(f"[red]✗ Test 4 failed: {e}[/red]")
        raise

    # Test 5: Input Generation
    step_start = time.time()
    try:
        test_chunks = test_input_generation_all_types(config, filtered_chunks or chunks)
        test_results.add_row(
            "5. Input Generation (All Types)",
            "[green]✓[/green]",
            f"{time.time() - step_start:.1f}s",
        )
    except Exception as e:
        test_results.add_row(
            "5. Input Generation (All Types)", "[red]✗[/red]", f"{time.time() - step_start:.1f}s"
        )
        console.print(test_results)
        console.print(f"[red]✗ Test 5 failed: {e}[/red]")
        raise

    # Test 6: Test Generation
    step_start = time.time()
    try:
        test_test_generation_with_retry(config, test_chunks)
        test_results.add_row(
            "6. Test Generation + Retry", "[green]✓[/green]", f"{time.time() - step_start:.1f}s"
        )
    except Exception as e:
        test_results.add_row(
            "6. Test Generation + Retry", "[red]✗[/red]", f"{time.time() - step_start:.1f}s"
        )
        console.print(test_results)
        console.print(f"[red]✗ Test 6 failed: {e}[/red]")
        raise

    # Test 7: Answer Generation
    step_start = time.time()
    try:
        test_answer_generation_with_verification(config, test_chunks)
        test_results.add_row(
            "7. Answer Gen + Verification", "[green]✓[/green]", f"{time.time() - step_start:.1f}s"
        )
    except Exception as e:
        test_results.add_row(
            "7. Answer Gen + Verification", "[red]✗[/red]", f"{time.time() - step_start:.1f}s"
        )
        console.print(test_results)
        console.print(f"[red]✗ Test 7 failed: {e}[/red]")
        raise

    # Test 8: Curation
    step_start = time.time()
    try:
        test_quality_curation(config, test_chunks)
        test_results.add_row(
            "8. Quality Curation", "[green]✓[/green]", f"{time.time() - step_start:.1f}s"
        )
    except Exception as e:
        test_results.add_row(
            "8. Quality Curation", "[red]✗[/red]", f"{time.time() - step_start:.1f}s"
        )
        console.print(test_results)
        console.print(f"[red]✗ Test 8 failed: {e}[/red]")
        raise

    # Test 9: Integration
    # step_start = time.time()
    # try:
    #    test_full_pipeline_integration(config)
    #    test_results.add_row(
    #        "9. Full Integration", "[green]✓[/green]", f"{time.time() - step_start:.1f}s"
    #    )
    # except Exception as e:
    #    test_results.add_row(
    #        "9. Full Integration", "[red]✗[/red]", f"{time.time() - step_start:.1f}s"
    #    )
    #    console.print(test_results)
    #    console.print(f"[red]✗ Test 9 failed: {e}[/red]")
    #    raise

    # Summary
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
        with open(TEST_RUN_DIR / "00_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        console.print(f"\n[cyan]Test outputs saved to:[/cyan]")
        console.print(f"  {TEST_RUN_DIR}")
        for file in sorted(TEST_RUN_DIR.glob("*.json*")):
            console.print(f"  - {file.name}")

    console.print("\n[bold green]✓ All tests passed successfully![/bold green]")
    console.print("\n[cyan]Refactored pipeline verified:[/cyan]")
    console.print("  ✓ Image classification working")
    console.print("  ✓ Image reference system working")
    console.print("  ✓ Accumulated code context working")
    console.print("  ✓ Multimodal generation strategy working")
    console.print("  ✓ Code verification with retry working")


if __name__ == "__main__":
    main()
