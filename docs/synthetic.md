# Synthetic Data Generation - Technical Documentation

## Architecture

### Image Resolution

The system includes an `ImageResolver` that handles different image source types:

- **PDF images**: `pdf:file.pdf:page0:img0` → Extracts actual bytes using PyMuPDF and saves to `outputs/images/`
- **Notebook attachments**: `attachment:image.png` → Extracts base64 data and saves to `outputs/images/`
- **Documentation paths**: `/learning/images/...` → Resolves to `data/.../public/learning/images/...` (both MDX and Jupyter)
- **Direct/relative paths**: Already on disk → Returns absolute path

All resolved image paths are stored in `ImageReference.resolved_path` for use during generation.

### Class Diagram

```mermaid
classDiagram
    class PipelineConfig {
        +List~SourceConfig~ sources
        +List~CategoryConfig~ categories
        +ModelConfig models
        +PromptsConfig prompts
        +GenerationConfig generation
        +DatasetConfig dataset
        +from_yaml() PipelineConfig
    }
    
    class DocumentParser {
        <<interface>>
        +can_parse(Path) bool
        +parse(Path) Document
    }
    
    class JupyterParser {
        +parse(Path) Document
    }
    
    class MDXParser {
        +parse(Path) Document
    }
    
    class PDFParser {
        +parse(Path) Document
    }
    
    class DocumentIngestion {
        +List~DocumentParser~ parsers
        +ingest_source(SourceConfig) List~Document~
    }
    
    class ContentChunker {
        +int max_length
        +int overlap
        +chunk_document(Document) List~Chunk~
    }
    
    class CategoryManager {
        +Dict~str,CategoryConfig~ categories
        +LLMClient llm_client
        +classify_chunk(Chunk, str) str
        +organize_by_category(List~Chunk~) Dict
    }
    
    class ModelRegistry {
        +ModelConfig config
        +Dict~str,LLMClient~ clients
        +get_llm_client(str) LLMClient
        +get_vlm_client(str) VLMClient
    }
    
    class GenerationPipeline {
        +PipelineConfig config
        +ModelRegistry registry
        +CategoryManager category_manager
        +generate_samples(Dict) List~Sample~
    }
    
    class DatasetBuilder {
        +stratified_build(List~Sample~) Tuple
    }
    
    class HuggingFaceExporter {
        +export(List, List, List) DatasetDict
        +save_to_disk(DatasetDict) Path
    }
    
    DocumentParser <|-- JupyterParser
    DocumentParser <|-- MDXParser
    DocumentParser <|-- PDFParser
    DocumentIngestion o-- DocumentParser
    GenerationPipeline --> ModelRegistry
    GenerationPipeline --> CategoryManager
    CategoryManager --> LLMClient
    ModelRegistry --> LLMClient
    ModelRegistry --> VLMClient
```

### Pipeline Flow

```mermaid
flowchart TD
    Start([Source Documents]) --> Parse
    
    subgraph Parse [Stage 1: Parse]
        P1[Jupyter Parser] --> D1[Documents]
        P2[MDX Parser] --> D1
        P3[PDF Parser] --> D1
    end
    
    D1 --> Chunk[Stage 2: Chunk]
    Chunk --> Chunks[Content Chunks]
    
    Chunks --> Classify[Stage 3: Classify]
    
    subgraph Classify [LLM Classification]
        CL1{LLM Available?}
        CL1 -->|Yes| CL2[LLM Classify]
        CL1 -->|No| CL3[Keyword Match]
        CL2 --> CL4[Category Assignment]
        CL3 --> CL4
    end
    
    CL4 --> Generate[Stage 4: Generate]
    
    subgraph Generate [Sample Generation]
        G1[Quality Filter] -->|Pass| G2[Question Gen]
        G2 --> G3{Has Image?}
        G3 -->|Yes| G4[VLM Describe]
        G3 -->|No| G5[Answer Gen]
        G4 --> G5
        G5 --> G6[Sample]
    end
    
    G6 --> Dedup[Stage 5: Deduplicate]
    Dedup --> Split[Stage 6: Split]
    
    subgraph Split [Stratified Split]
        S1[Train 80%]
        S2[Val 10%]
        S3[Test 10%]
    end
    
    Split --> Export[Stage 7: Export]
    Export --> Final([HuggingFace Dataset])
    
    style Parse fill:#e1f5ff
    style Generate fill:#ffe1f5
    style Split fill:#f5ffe1
```

### Data Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Ingestion
    participant Parser
    participant Chunker
    participant CategoryMgr
    participant Pipeline
    participant LLM
    participant VLM
    participant Exporter
    
    User->>CLI: synthetic-data generate
    CLI->>Ingestion: ingest_source()
    Ingestion->>Parser: parse(file)
    Parser-->>Ingestion: Document
    Ingestion-->>CLI: List[Document]
    
    CLI->>Chunker: chunk_document()
    Chunker-->>CLI: List[Chunk]
    
    CLI->>CategoryMgr: organize_by_category()
    CategoryMgr->>LLM: classify(chunk)
    LLM-->>CategoryMgr: category
    CategoryMgr-->>CLI: Dict[category, chunks]
    
    CLI->>Pipeline: generate_samples()
    
    loop For each chunk
        Pipeline->>LLM: generate question
        LLM-->>Pipeline: question
        
        alt Has Image
            Pipeline->>VLM: describe image
            VLM-->>Pipeline: description
        end
        
        Pipeline->>LLM: generate answer
        LLM-->>Pipeline: answer
        Pipeline->>Pipeline: create Sample
    end
    
    Pipeline-->>CLI: List[Sample]
    
    CLI->>Exporter: export()
    Exporter-->>CLI: DatasetDict
    Exporter->>Exporter: save_to_disk()
    Exporter-->>User: Dataset ready
```

## Configuration Schema

### Complete Structure

```yaml
sources:                    # Input sources
  - path: string           # Local path or GitHub URL
    type: directory|github
    folders: [...]          # For GitHub repos
    include_patterns: [...] # File patterns to include
    exclude_patterns: [...] # File patterns to exclude

categories:                 # Knowledge domains
  - name: string
    description: string     # Used for LLM classification
    weight: float          # Relative sample allocation

models:
  endpoints:
    - name: string
      base_url: string
      api_key: string      # Supports ${ENV_VAR}
      model_name: string
      max_tokens: int
      temperature: float

prompts:                    # All prompts customizable
  question_generation: string
  answer_generation: string
  summary_generation: string
  caption_generation: string
  code_generation: string
  content_quality_check: string
  image_quality_check: string
  category_classification: string

generation:
  target_samples: int
  question_model: string
  vision_model: string
  answer_model: string
  
  batch_size: int
  multimodal_ratio: float
  
  question_types: [qa, code, caption, summary]
  question_type_weights: {qa: 1.0, ...}
  
  difficulty_levels: [easy, medium, hard]
  difficulty_weights: {easy: 1.0, ...}
  
  max_context_length: int
  chunk_overlap: int
  
  enable_content_filtering: bool
  enable_deduplication: bool
  similarity_threshold: float

dataset:
  name: string
  description: string
  parsed_dir: Path
  generated_dir: Path
  final_dir: Path
  images_dir: Path
  train_split: float
  val_split: float
  test_split: float
  license: string

seed: int
```

## Sample Structure

```json
{
  "question": "What is quantum superposition?",
  "answer": "Quantum superposition is...",
  "category": "quantum_fundamentals",
  "question_type": "qa",
  "difficulty": "easy",
  "image": null,
  "has_image": false,
  "code_context": "",
  "source_path": "data/intro.ipynb",
  "metadata": {}
}
```

## Extension Points

### Add New Parser

```python
from synthetic_data.parsers.base import DocumentParser, Document

class CustomParser(DocumentParser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix == ".custom"
    
    def parse(self, path: Path) -> Document:
        # Implementation
        pass

# Register in ingestion.py
```

### Add New Question Type

```python
# 1. Update enum in config/schema.py
class QuestionType(str, Enum):
    MY_TYPE = "my_type"

# 2. Add prompt in config.yaml
prompts:
  my_type_generation: |
    Your prompt template
    {context}

# 3. Update PromptSet in generators/prompts.py
```

## Caching System

The pipeline includes an intelligent caching system for incremental processing:

### Features
- **Per-source caching** - each data source cached independently
- **Incremental saving** - cache saved after each source completes
- **Interrupt-safe** - Ctrl+C preserves already-processed sources
- **Automatic cache invalidation** when files or config change
- **Manual cache control** via CLI commands

### Cache Commands

```bash
# Show cache information
synthetic-data cache-info --config config.yaml

# Clear all cache
synthetic-data cache-clear --config config.yaml

# Clear specific stage cache
synthetic-data cache-clear --config config.yaml --stage parse

# Run without using cache
synthetic-data parse --config config.yaml --no-cache

# Clear cache before running
synthetic-data parse --config config.yaml --clear-cache
```

### Cache Location
Cache is stored in `outputs/.cache/` with manifest tracking:
- **Per-source cache files** - Each data source gets its own cache file
- **File modification times** - Automatic invalidation on file changes
- **Configuration parameters** - Invalidation on config changes
- **Processing timestamps** - Track when each source was processed

### How It Works
1. Each source path gets a unique cache identifier (MD5 hash)
2. Cache saved immediately after each source completes
3. On Ctrl+C, already-processed sources remain cached
4. Next run loads cached sources, only processes uncached ones
5. File changes automatically invalidate affected source caches

## Progress Tracking

All pipeline stages now show detailed progress:
- **Accurate progress bars** tracking files processed (not documents)
- **File and document counts** - Shows "X docs from Y files"
- **Running totals** - "[42/1135] Processed..." shows cumulative progress
- **Time elapsed** and **time remaining** estimates
- **Separate tracking** for cached vs newly-processed sources

## Testing

Run integration tests:

```bash
python tests/test_integration.py
```

Tests:
- Endpoint connectivity
- Document parsing (all formats)
- Content chunking
- Full ingestion pipeline
- Configuration validation
- Cache functionality

