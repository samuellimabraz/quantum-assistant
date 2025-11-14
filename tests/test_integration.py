"""Integration tests for synthetic data pipeline."""

from pathlib import Path

from synthetic_data.config import PipelineConfig
from synthetic_data.extractors import DocumentIngestion, ContentChunker
from synthetic_data.models import LLMClient, Message
from synthetic_data.parsers import JupyterParser, MDXParser, PDFParser


def test_endpoint_connection():
    """Test connection to VLM/LLM endpoint."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    base_url = os.getenv("VISION_MODEL_BASE_URL")
    api_key = os.getenv("VISION_MODEL_API_KEY")
    model_name = os.getenv("VISION_MODEL_NAME")

    if not base_url or not api_key:
        print("Skipping endpoint test - environment variables not set")
        print("Set VISION_MODEL_BASE_URL and VISION_MODEL_API_KEY in .env file")
        return

    client = LLMClient(
        base_url=f"{base_url}",
        api_key=api_key,
        model_name=model_name,
    )

    try:
        response = client.generate(
            [Message(role="user", content="Hello, this is a test message.")],
            max_tokens=100,
            temperature=0.1,
        )
        #print(response)
        assert response
        print(f"  Endpoint: {base_url}")
        print(f"  Model: {model_name}")
        print(f"  Response: {response}")
        print(f"  ✓ Endpoint connection successful")
    except Exception as e:
        print(f"  ✗ Endpoint test failed: {e}")
        raise
    finally:
        client.close()


def test_parse_documentation():
    """Test parsing actual documentation from data folder."""
    data_dir = Path("/Users/samuel/Developer/avante/unifei/tcc/quantum-assistant/data")

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    parsers = {
        ".ipynb": JupyterParser(),
        ".mdx": MDXParser(),
        ".pdf": PDFParser(),
    }

    for extension, parser in parsers.items():
        files = list(data_dir.rglob(f"*{extension}"))
        if files:
            print(f"\nTesting {extension} parser with {len(files)} files")

            test_file = files[0]
            try:
                doc = parser.parse(test_file)

                print(f"  File: {test_file.name}")
                print(f"  Title: {doc.title[:80] if doc.title else 'No title'}")
                print(f"  Content length: {len(doc.content)}")
                print(f"  Code blocks: {len(doc.code_blocks)}")
                print(f"  Images: {len(doc.images)}")

                if doc.images:
                    print(f"  First image: {doc.images[0].path}")

                assert len(doc.content) > 0, "No content extracted"
                print(f"  ✓ Parser works correctly")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                raise


def test_chunking():
    """Test content chunking."""
    data_dir = Path("/Users/samuel/Developer/avante/unifei/tcc/quantum-assistant/data")

    if not data_dir.exists():
        return

    # Find first notebook
    notebooks = list(data_dir.rglob("*.ipynb"))
    if not notebooks:
        print("No notebooks found")
        return

    parser = JupyterParser()
    doc = parser.parse(notebooks[0])

    chunker = ContentChunker(max_length=2048, overlap=200)
    chunks = chunker.chunk_document(doc)

    print(f"\nChunking test:")
    print(f"  Document: {notebooks[0].name}")
    print(f"  Original length: {len(doc.content)}")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Avg chunk size: {sum(len(c.text) for c in chunks) // len(chunks)}")

    assert len(chunks) > 0, "No chunks created"
    print(f"  ✓ Chunking works correctly")


def test_config_loading():
    """Test loading actual configuration."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    config_path = Path("src/synthetic_data/yaml/config.yaml")

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return

    config = PipelineConfig.from_yaml(config_path)

    print(f"Configuration test:")
    print(f"  Sources: {len(config.sources)}")
    print(f"  Categories: {len(config.categories)}")
    print(f"  Target samples: {config.generation.target_samples}")
    print(f"  Question types: {[qt.value for qt in config.generation.question_types]}")
    print(f"  Models: {len(config.models.endpoints)}")

    assert len(config.sources) > 0
    assert len(config.categories) == 14, f"Expected 14 categories, got {len(config.categories)}"
    print(f"  ✓ Configuration valid")


def test_full_ingestion():
    """Test full document ingestion pipeline."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    config_path = Path("src/synthetic_data/yaml/config.yaml")

    if not config_path.exists():
        print("Config not found")
        return

    config = PipelineConfig.from_yaml(config_path)

    # Check if source paths exist
    print(f"\nIngestion test:")
    for source in config.sources:
        source_path = Path(source.path)
        print(f"  Checking source: {source.path}")
        print(f"    Exists: {source_path.exists()}")
        if source_path.exists():
            print(f"    Is directory: {source_path.is_dir()}")
            if source_path.is_dir():
                # Count files matching patterns
                for pattern in source.include_patterns:
                    files = list(source_path.rglob(pattern))
                    print(f"    {pattern}: {len(files)} files")

    ingestion = DocumentIngestion()

    all_docs = []
    for source in config.sources:
        try:
            docs = ingestion.ingest_source(source)
            all_docs.extend(docs)
            print(f"  Ingested from {source.path}: {len(docs)} documents")
        except Exception as e:
            print(f"  Error ingesting {source.path}: {e}")

    if len(all_docs) == 0:
        print("  Warning: No documents ingested - check paths in config.yaml")
        return

    print(f"  Total documents: {len(all_docs)}")

    # Group by type
    by_type = {}
    for doc in all_docs:
        ext = doc.source_path.suffix
        by_type[ext] = by_type.get(ext, 0) + 1

    for ext, count in sorted(by_type.items()):
        print(f"  {ext}: {count}")

    # Check images
    total_images = sum(len(doc.images) for doc in all_docs)
    docs_with_images = sum(1 for doc in all_docs if doc.has_images)

    print(f"  Total images: {total_images}")
    print(f"  Docs with images: {docs_with_images}")

    print(f"  ✓ Ingestion works correctly")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    print("Running integration tests...")
    print("=" * 60)

    try:
        test_config_loading()
    except Exception as e:
        print(f"✗ Config test failed: {e}")

    print("=" * 60)

    try:
        test_parse_documentation()
    except Exception as e:
        print(f"✗ Parse test failed: {e}")

    print("=" * 60)

    try:
        test_chunking()
    except Exception as e:
        print(f"✗ Chunking test failed: {e}")

    print("=" * 60)

    try:
        test_full_ingestion()
    except Exception as e:
        print(f"✗ Ingestion test failed: {e}")

    print("=" * 60)

    try:
        test_endpoint_connection()
    except Exception as e:
        print(f"✗ Endpoint test failed: {e}")

    print("=" * 60)
    print("\nTests completed!")
