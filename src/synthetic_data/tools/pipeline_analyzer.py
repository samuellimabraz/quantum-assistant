"""Pipeline Analyzer - Comprehensive analysis for each pipeline stage.

Provides detailed statistics and analysis for:
- Source files (documents, types, sources)
- Image transcription and filtering
- Chunk creation and quality
- Allocation analysis
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict


@dataclass
class SourceStatistics:
    """Statistics about the source documents."""

    total_files: int = 0
    total_code_blocks: int = 0
    total_images: int = 0

    # By file type (.ipynb, .mdx, .pdf)
    files_by_type: dict[str, int] = field(default_factory=dict)
    code_by_type: dict[str, int] = field(default_factory=dict)
    images_by_type: dict[str, int] = field(default_factory=dict)

    # By source directory
    files_by_source: dict[str, int] = field(default_factory=dict)
    code_by_source: dict[str, int] = field(default_factory=dict)
    images_by_source: dict[str, int] = field(default_factory=dict)

    # Combined: source -> type -> count
    files_by_source_and_type: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class ImageStatistics:
    """Statistics about image transcription and filtering."""

    total_images: int = 0
    transcribed_images: int = 0
    classified_images: int = 0

    # After filtering
    images_after_filter: int = 0
    images_removed: int = 0

    # By image type (circuit, chart, bloch_sphere, etc.)
    by_type: dict[str, int] = field(default_factory=dict)
    by_type_after_filter: dict[str, int] = field(default_factory=dict)

    # Filter decisions
    filter_passed: int = 0
    filter_rejected: int = 0


@dataclass
class ChunkStatistics:
    """Comprehensive statistics about chunks."""

    total_chunks: int = 0
    chunks_with_code: int = 0
    chunks_with_images: int = 0
    multimodal_chunks: int = 0
    text_only_chunks: int = 0

    # Size distribution
    chunk_sizes: list[int] = field(default_factory=list)
    avg_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0

    # Code distribution
    code_block_counts: list[int] = field(default_factory=list)
    total_code_blocks: int = 0

    # Image distribution
    image_counts: list[int] = field(default_factory=list)
    unique_images: int = 0
    total_image_refs: int = 0

    # Image type distribution in chunks
    image_types_in_chunks: dict[str, int] = field(default_factory=dict)

    # Quality distribution
    quality_high: int = 0
    quality_medium: int = 0
    quality_low: int = 0

    # By source type
    chunks_by_source_type: dict[str, int] = field(default_factory=dict)
    multimodal_by_source_type: dict[str, int] = field(default_factory=dict)
    code_by_source_type: dict[str, int] = field(default_factory=dict)

    # After filtering
    chunks_after_filter: int = 0
    chunks_removed: int = 0
    images_after_filter: int = 0
    multimodal_after_filter: int = 0


@dataclass
class PipelineStatistics:
    """Complete pipeline statistics."""

    source: SourceStatistics = field(default_factory=SourceStatistics)
    images: ImageStatistics = field(default_factory=ImageStatistics)
    chunks: ChunkStatistics = field(default_factory=ChunkStatistics)


class PipelineAnalyzer:
    """Analyzes all pipeline stages for comprehensive statistics."""

    def __init__(self, base_dir: Path):
        """Initialize analyzer with base output directory.

        Args:
            base_dir: Path to outputs directory (e.g., outputs/)
        """
        self.base_dir = Path(base_dir)
        self._stats: PipelineStatistics | None = None

    def analyze(self) -> PipelineStatistics:
        """Run complete pipeline analysis."""
        stats = PipelineStatistics()

        # Analyze source documents
        parsed_dir = self.base_dir / "parsed"
        if (parsed_dir / "documents.pkl").exists():
            stats.source = self._analyze_sources(parsed_dir / "documents.pkl")

        # Analyze transcribed images
        transcribed_dir = self.base_dir / "transcribed"
        if (transcribed_dir / "documents.pkl").exists():
            stats.images = self._analyze_images(
                transcribed_dir / "documents.pkl",
                self.base_dir / "filtered_images" / "documents.pkl",
            )

        # Analyze chunks
        chunks_dir = self.base_dir / "chunks"
        filtered_dir = self.base_dir / "filtered"
        if (chunks_dir / "chunks.pkl").exists():
            stats.chunks = self._analyze_chunks(
                chunks_dir / "chunks.pkl",
                filtered_dir / "chunks.pkl" if (filtered_dir / "chunks.pkl").exists() else None,
            )

        self._stats = stats
        return stats

    def _analyze_sources(self, documents_path: Path) -> SourceStatistics:
        """Analyze source documents."""
        stats = SourceStatistics()

        with open(documents_path, "rb") as f:
            documents = pickle.load(f)

        stats.total_files = len(documents)

        # Initialize nested dicts
        files_by_source_and_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for doc in documents:
            file_type = doc.source_path.suffix.lower()
            source_dir = self._get_source_name(doc.source_path)

            # By type
            stats.files_by_type[file_type] = stats.files_by_type.get(file_type, 0) + 1
            stats.code_by_type[file_type] = stats.code_by_type.get(file_type, 0) + len(doc.code_blocks)
            stats.images_by_type[file_type] = stats.images_by_type.get(file_type, 0) + len(doc.images)

            # By source
            stats.files_by_source[source_dir] = stats.files_by_source.get(source_dir, 0) + 1
            stats.code_by_source[source_dir] = stats.code_by_source.get(source_dir, 0) + len(doc.code_blocks)
            stats.images_by_source[source_dir] = stats.images_by_source.get(source_dir, 0) + len(doc.images)

            # Combined
            files_by_source_and_type[source_dir][file_type] += 1

            # Totals
            stats.total_code_blocks += len(doc.code_blocks)
            stats.total_images += len(doc.images)

        stats.files_by_source_and_type = {k: dict(v) for k, v in files_by_source_and_type.items()}

        return stats

    def _analyze_images(
        self,
        transcribed_path: Path,
        filtered_path: Path | None,
    ) -> ImageStatistics:
        """Analyze image transcription and filtering."""
        stats = ImageStatistics()

        # Load transcribed documents
        with open(transcribed_path, "rb") as f:
            transcribed_docs = pickle.load(f)

        all_images = [img for doc in transcribed_docs for img in doc.images]
        stats.total_images = len(all_images)

        for img in all_images:
            if img.transcription:
                stats.transcribed_images += 1
            if img.image_type:
                stats.classified_images += 1
                type_name = img.image_type.value if hasattr(img.image_type, "value") else str(img.image_type)
                stats.by_type[type_name] = stats.by_type.get(type_name, 0) + 1

        # Analyze filtered images if available
        if filtered_path and filtered_path.exists():
            with open(filtered_path, "rb") as f:
                filtered_docs = pickle.load(f)

            filtered_images = [img for doc in filtered_docs for img in doc.images]
            stats.images_after_filter = len(filtered_images)
            stats.images_removed = stats.total_images - stats.images_after_filter

            for img in filtered_images:
                if img.image_type:
                    type_name = img.image_type.value if hasattr(img.image_type, "value") else str(img.image_type)
                    stats.by_type_after_filter[type_name] = stats.by_type_after_filter.get(type_name, 0) + 1

        return stats

    def _analyze_chunks(
        self,
        chunks_path: Path,
        filtered_path: Path | None,
    ) -> ChunkStatistics:
        """Analyze chunks comprehensively."""
        stats = ChunkStatistics()

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        stats.total_chunks = len(chunks)

        unique_image_ids = set()

        for chunk in chunks:
            chunk_size = len(chunk.text)
            stats.chunk_sizes.append(chunk_size)

            # Code statistics
            code_count = len(chunk.code_blocks)
            stats.code_block_counts.append(code_count)
            stats.total_code_blocks += code_count
            if code_count > 0:
                stats.chunks_with_code += 1

            # Image statistics
            image_count = len(chunk.images)
            stats.image_counts.append(image_count)
            stats.total_image_refs += image_count
            if image_count > 0:
                stats.chunks_with_images += 1

            for img in chunk.images:
                if img.image_id:
                    unique_image_ids.add(img.image_id)
                if img.image_type:
                    type_name = img.image_type.value if hasattr(img.image_type, "value") else str(img.image_type)
                    stats.image_types_in_chunks[type_name] = stats.image_types_in_chunks.get(type_name, 0) + 1

            # Multimodal check
            if chunk.is_multimodal:
                stats.multimodal_chunks += 1
            else:
                stats.text_only_chunks += 1

            # Quality
            quality = chunk.quality.value if hasattr(chunk.quality, "value") else str(chunk.quality)
            if quality == "high":
                stats.quality_high += 1
            elif quality == "medium":
                stats.quality_medium += 1
            else:
                stats.quality_low += 1

            # By source type
            source_type = chunk.source_path.suffix.lower()
            stats.chunks_by_source_type[source_type] = stats.chunks_by_source_type.get(source_type, 0) + 1
            if code_count > 0:
                stats.code_by_source_type[source_type] = stats.code_by_source_type.get(source_type, 0) + 1
            if chunk.is_multimodal:
                stats.multimodal_by_source_type[source_type] = stats.multimodal_by_source_type.get(source_type, 0) + 1

        stats.unique_images = len(unique_image_ids)

        # Size statistics
        if stats.chunk_sizes:
            stats.avg_chunk_size = sum(stats.chunk_sizes) / len(stats.chunk_sizes)
            stats.min_chunk_size = min(stats.chunk_sizes)
            stats.max_chunk_size = max(stats.chunk_sizes)

        # Analyze filtered chunks if available
        if filtered_path and filtered_path.exists():
            with open(filtered_path, "rb") as f:
                filtered_chunks = pickle.load(f)

            stats.chunks_after_filter = len(filtered_chunks)
            stats.chunks_removed = stats.total_chunks - stats.chunks_after_filter

            filtered_image_ids = set()
            stats.multimodal_after_filter = 0
            for chunk in filtered_chunks:
                if chunk.is_multimodal:
                    stats.multimodal_after_filter += 1
                for img in chunk.images:
                    if img.image_id:
                        filtered_image_ids.add(img.image_id)
            stats.images_after_filter = len(filtered_image_ids)

        return stats

    def _get_source_name(self, path: Path) -> str:
        """Extract meaningful source name from path."""
        parts = path.parts
        # Find 'data' directory and get the next part
        for i, part in enumerate(parts):
            if part == "data" and i + 1 < len(parts):
                return parts[i + 1]
        # Fallback: use parent directory name
        return path.parent.name

    def get_statistics(self) -> PipelineStatistics:
        """Get computed statistics, analyzing if needed."""
        if self._stats is None:
            return self.analyze()
        return self._stats


def load_pipeline_statistics(base_dir: Path) -> PipelineStatistics:
    """Convenience function to load pipeline statistics."""
    analyzer = PipelineAnalyzer(base_dir)
    return analyzer.analyze()





