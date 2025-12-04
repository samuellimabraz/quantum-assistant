"""Content-aware allocation for synthetic data generation.

Allocates chunks to sample generation tasks with:
1. Over-allocation to reduce retry attempts
2. Diversity-aware selection across chunks and images
3. Per-input-type multimodal ratios
4. Comprehensive utilization metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from synthetic_data.config import QuestionType
from synthetic_data.parsers.base import ImageReference, ImageType

if TYPE_CHECKING:
    from synthetic_data.extractors.chunker import Chunk


@dataclass
class TypeAllocationConfig:
    """Configuration for a single question type."""

    ratio: float  # Portion of total samples (e.g., 0.30 for 30%)
    multimodal_ratio: float  # Portion of this type that should be multimodal


@dataclass
class AllocationConfig:
    """Configuration for sample allocation."""

    target_samples: int
    type_configs: dict[QuestionType, TypeAllocationConfig]
    over_allocation_factor: float = 1.8  # Generate 80% more to account for failures

    # Scoring thresholds
    min_code_score: float = 0.05
    min_qa_score: float = 0.05

    def get_type_target(self, question_type: QuestionType) -> int:
        """Get target sample count for a question type."""
        config = self.type_configs.get(question_type)
        if not config:
            return 0
        return int(self.target_samples * config.ratio)

    def get_multimodal_target(self, question_type: QuestionType) -> int:
        """Get target multimodal count for a question type."""
        config = self.type_configs.get(question_type)
        if not config:
            return 0
        type_total = self.get_type_target(question_type)
        return int(type_total * config.multimodal_ratio)

    def get_text_only_target(self, question_type: QuestionType) -> int:
        """Get target text-only count for a question type."""
        return self.get_type_target(question_type) - self.get_multimodal_target(question_type)

    def get_over_allocated_target(self, question_type: QuestionType) -> int:
        """Get over-allocated target for a question type."""
        base_target = self.get_type_target(question_type)
        return int(base_target * self.over_allocation_factor)

    def get_over_allocated_multimodal(self, question_type: QuestionType) -> int:
        """Get over-allocated multimodal target for a question type."""
        base_target = self.get_multimodal_target(question_type)
        return int(base_target * self.over_allocation_factor)


@dataclass
class SampleTask:
    """A task representing one sample to generate."""

    chunk: Chunk
    question_type: QuestionType
    target_image: ImageReference | None = None
    score: float = 0.0

    @property
    def is_multimodal(self) -> bool:
        """Check if this is a multimodal task."""
        return self.target_image is not None

    @property
    def allocation_key(self) -> tuple:
        """Unique key for this allocation (source_path, chunk_id, image_id, type)."""
        image_key = self.target_image.image_id if self.target_image else "text"
        return (str(self.chunk.source_path), self.chunk.chunk_id, image_key, self.question_type)

    @property
    def chunk_key(self) -> tuple:
        """Key identifying just the chunk (for diversity tracking)."""
        return (str(self.chunk.source_path), self.chunk.chunk_id)

    @property
    def image_key(self) -> str | None:
        """Key identifying just the image (for diversity tracking)."""
        return self.target_image.image_id if self.target_image else None


@dataclass
class AllocationMetrics:
    """Metrics about allocation diversity and utilization."""

    total_chunks: int = 0
    chunks_used: int = 0
    chunks_multimodal: int = 0

    total_images: int = 0
    images_used: int = 0

    tasks_by_type: dict[QuestionType, int] = field(default_factory=dict)
    multimodal_by_type: dict[QuestionType, int] = field(default_factory=dict)

    # Diversity metrics
    avg_chunk_usage: float = 0.0  # How many times each chunk is used on average
    chunk_coverage: float = 0.0  # Percentage of chunks used at least once
    image_coverage: float = 0.0  # Percentage of images used at least once

    def __str__(self) -> str:
        """Human-readable metrics summary."""
        lines = [
            "Allocation Metrics:",
            f"  Chunks: {self.chunks_used}/{self.total_chunks} used ({self.chunk_coverage:.1%} coverage)",
            f"  Images: {self.images_used}/{self.total_images} used ({self.image_coverage:.1%} coverage)",
            f"  Avg chunk usage: {self.avg_chunk_usage:.2f}x",
            "",
            "  By Question Type:",
        ]
        for qt in QuestionType:
            total = self.tasks_by_type.get(qt, 0)
            mm = self.multimodal_by_type.get(qt, 0)
            if total > 0:
                lines.append(f"    {qt.value}: {total} (mm: {mm})")
        return "\n".join(lines)


@dataclass
class AllocationResult:
    """Result of allocation with statistics and metrics."""

    tasks: list[SampleTask]
    metrics: AllocationMetrics = field(default_factory=AllocationMetrics)

    @property
    def total_samples(self) -> int:
        return len(self.tasks)

    @property
    def multimodal_samples(self) -> int:
        return sum(1 for t in self.tasks if t.is_multimodal)

    @property
    def text_only_samples(self) -> int:
        return self.total_samples - self.multimodal_samples

    def samples_by_type(self) -> dict[QuestionType, int]:
        """Get sample counts by question type."""
        counts = {qt: 0 for qt in QuestionType}
        for task in self.tasks:
            counts[task.question_type] += 1
        return counts

    def multimodal_by_type(self) -> dict[QuestionType, int]:
        """Get multimodal counts by question type."""
        counts = {qt: 0 for qt in QuestionType}
        for task in self.tasks:
            if task.is_multimodal:
                counts[task.question_type] += 1
        return counts

    def unique_chunks_used(self) -> int:
        """Count unique chunks used."""
        return len(set(t.chunk_key for t in self.tasks))

    def unique_images_used(self) -> int:
        """Count unique images used."""
        return len(set(t.target_image.image_id for t in self.tasks if t.target_image))


class ChunkScorer:
    """Scores chunks for suitability to each question type."""

    CODE_IMAGE_TYPES = {ImageType.CIRCUIT}

    QA_IMAGE_TYPES = {
        ImageType.CIRCUIT,
        ImageType.CHART,
        ImageType.BLOCH_SPHERE,
        ImageType.DIAGRAM,
        ImageType.TABLE,
        ImageType.CODE_OUTPUT,
        ImageType.FORMULA,
    }

    def compute_code_score(self, chunk: Chunk, target_image: ImageReference | None = None) -> float:
        """Compute suitability score for code generation tasks."""
        score = 0.0

        if chunk.code_blocks:
            score += 0.5
            code_len = sum(len(cb) for cb in chunk.code_blocks)
            score += min(code_len / 400, 0.3)
        else:
            if len(chunk.text) >= 200:
                score += 0.2

        if target_image and target_image.image_type in self.CODE_IMAGE_TYPES:
            score += 0.3

        return min(score, 1.0)

    def compute_qa_score(self, chunk: Chunk, target_image: ImageReference | None = None) -> float:
        """Compute suitability score for QA tasks."""
        score = 0.0

        text_len = len(chunk.text)
        if text_len >= 50:
            score += 0.3
            length_ratio = min(text_len / 1000, 0.3)
            score += length_ratio

        if target_image:
            if target_image.image_type in self.QA_IMAGE_TYPES:
                score += 0.3
            else:
                score += 0.1

        if chunk.previous_chunk_text or chunk.next_chunk_text:
            score += 0.1

        return min(score, 1.0)

    def get_score(
        self, chunk: Chunk, question_type: QuestionType, target_image: ImageReference | None = None
    ) -> float:
        """Get suitability score for a chunk and question type."""
        if question_type in (QuestionType.FUNCTION_COMPLETION, QuestionType.CODE_GENERATION):
            return self.compute_code_score(chunk, target_image)
        return self.compute_qa_score(chunk, target_image)


class DiversityTracker:
    """Tracks chunk and image usage for diversity-aware selection."""

    def __init__(self):
        self.chunk_usage: dict[tuple, int] = {}  # chunk_key -> usage count
        self.image_usage: dict[str, int] = {}  # image_id -> usage count
        self.type_chunk_usage: dict[tuple[QuestionType, tuple], int] = (
            {}
        )  # (type, chunk_key) -> count

    def record_usage(self, task: SampleTask) -> None:
        """Record that a task was selected."""
        chunk_key = task.chunk_key
        self.chunk_usage[chunk_key] = self.chunk_usage.get(chunk_key, 0) + 1

        type_key = (task.question_type, chunk_key)
        self.type_chunk_usage[type_key] = self.type_chunk_usage.get(type_key, 0) + 1

        if task.image_key:
            self.image_usage[task.image_key] = self.image_usage.get(task.image_key, 0) + 1

    def get_diversity_penalty(self, task: SampleTask) -> float:
        """Get a penalty factor based on prior usage (0 = fresh, higher = more used)."""
        chunk_key = task.chunk_key
        chunk_uses = self.chunk_usage.get(chunk_key, 0)

        type_key = (task.question_type, chunk_key)
        type_uses = self.type_chunk_usage.get(type_key, 0)

        image_uses = 0
        if task.image_key:
            image_uses = self.image_usage.get(task.image_key, 0)

        # Penalty formula: penalize reuse, especially within same type
        penalty = (type_uses * 0.5) + (chunk_uses * 0.3) + (image_uses * 0.2)
        return penalty

    def get_adjusted_score(self, task: SampleTask, diversity_weight: float = 0.4) -> float:
        """Get score adjusted for diversity (higher is better)."""
        penalty = self.get_diversity_penalty(task)
        # Reduce score based on penalty, but don't go below 10% of original
        adjustment = max(0.1, 1.0 - (penalty * diversity_weight))
        return task.score * adjustment


class Allocator:
    """Allocates chunks to sample generation tasks with diversity awareness.

    Key features:
    1. Over-allocation to reduce retry attempts
    2. Diversity tracking to maximize chunk/image utilization
    3. Per-type multimodal ratios
    4. Comprehensive metrics
    """

    def __init__(
        self,
        config: AllocationConfig,
        scorer: ChunkScorer | None = None,
        diversity_weight: float = 0.4,
    ):
        """Initialize allocator.

        Args:
            config: Allocation configuration
            scorer: Optional scorer (creates default if not provided)
            diversity_weight: How much to weight diversity vs score (0-1)
        """
        self.config = config
        self.scorer = scorer or ChunkScorer()
        self.diversity_weight = diversity_weight
        self.tracker = DiversityTracker()

    def allocate(self, chunks: list[Chunk], use_over_allocation: bool = True) -> AllocationResult:
        """Allocate chunks to sample tasks with diversity awareness.

        Args:
            chunks: Available chunks
            use_over_allocation: Whether to apply over-allocation factor

        Returns:
            AllocationResult with tasks, statistics, and metrics
        """
        candidates = self._build_candidates(chunks)
        targets = self._calculate_targets(use_over_allocation)
        tasks = self._select_tasks_with_diversity(candidates, targets)
        metrics = self._compute_metrics(chunks, tasks)

        return AllocationResult(tasks=tasks, metrics=metrics)

    def _build_candidates(self, chunks: list[Chunk]) -> list[SampleTask]:
        """Build all possible candidate tasks from chunks."""
        candidates = []

        for chunk in chunks:
            transcribed_images = chunk.transcribed_images

            for qt in QuestionType:
                # Multimodal candidates: one per image
                for img in transcribed_images:
                    score = self.scorer.get_score(chunk, qt, img)
                    candidates.append(
                        SampleTask(
                            chunk=chunk,
                            question_type=qt,
                            target_image=img,
                            score=score,
                        )
                    )

                # Text-only candidate
                score = self.scorer.get_score(chunk, qt, None)
                candidates.append(
                    SampleTask(
                        chunk=chunk,
                        question_type=qt,
                        target_image=None,
                        score=score,
                    )
                )

        return candidates

    def _calculate_targets(self, use_over_allocation: bool) -> dict[tuple[QuestionType, bool], int]:
        """Calculate target counts for each (type, multimodal) combination."""
        targets = {}

        for qt in QuestionType:
            if use_over_allocation:
                mm_target = self.config.get_over_allocated_multimodal(qt)
                total_target = self.config.get_over_allocated_target(qt)
            else:
                mm_target = self.config.get_multimodal_target(qt)
                total_target = self.config.get_type_target(qt)

            text_target = total_target - mm_target
            targets[(qt, True)] = mm_target
            targets[(qt, False)] = text_target

        return targets

    def _select_tasks_with_diversity(
        self,
        candidates: list[SampleTask],
        targets: dict[tuple[QuestionType, bool], int],
    ) -> list[SampleTask]:
        """Select tasks prioritizing both quality and diversity."""
        selected: list[SampleTask] = []
        used_keys: set[tuple] = set()

        min_code = self.config.min_code_score
        min_qa = self.config.min_qa_score

        # Pre-filter and organize pools
        pools = {}
        for (qt, is_mm), target in targets.items():
            if target <= 0:
                continue

            min_score = min_code if qt != QuestionType.QA else min_qa
            pool = [
                c
                for c in candidates
                if c.question_type == qt and c.is_multimodal == is_mm and c.score >= min_score
            ]
            pools[(qt, is_mm)] = pool

        # Sort by scarcity (process scarce combinations first)
        sorted_targets = sorted(
            targets.items(),
            key=lambda x: len(pools.get(x[0], [])) / max(x[1], 1),
        )

        for (qt, is_mm), target in sorted_targets:
            if target <= 0:
                continue

            pool = pools.get((qt, is_mm), [])
            available = [c for c in pool if c.allocation_key not in used_keys]

            if not available:
                continue

            # Score with diversity adjustment
            scored = [
                (c, self.tracker.get_adjusted_score(c, self.diversity_weight)) for c in available
            ]

            # Sort by adjusted score (descending)
            scored.sort(key=lambda x: x[1], reverse=True)

            # Select up to target
            count = 0
            for candidate, _ in scored:
                if count >= target:
                    break

                selected.append(candidate)
                used_keys.add(candidate.allocation_key)
                self.tracker.record_usage(candidate)
                count += 1

        return selected

    def _compute_metrics(self, chunks: list[Chunk], tasks: list[SampleTask]) -> AllocationMetrics:
        """Compute allocation metrics."""
        metrics = AllocationMetrics()

        # Chunk statistics
        all_chunk_keys = set((str(c.source_path), c.chunk_id) for c in chunks)
        used_chunk_keys = set(t.chunk_key for t in tasks)
        multimodal_chunk_keys = set(t.chunk_key for t in tasks if t.is_multimodal)

        metrics.total_chunks = len(all_chunk_keys)
        metrics.chunks_used = len(used_chunk_keys)
        metrics.chunks_multimodal = len(multimodal_chunk_keys)

        # Image statistics
        all_image_ids = set()
        for chunk in chunks:
            for img in chunk.transcribed_images:
                all_image_ids.add(img.image_id)

        used_image_ids = set(t.image_key for t in tasks if t.image_key)

        metrics.total_images = len(all_image_ids)
        metrics.images_used = len(used_image_ids)

        # Task breakdown
        for qt in QuestionType:
            metrics.tasks_by_type[qt] = sum(1 for t in tasks if t.question_type == qt)
            metrics.multimodal_by_type[qt] = sum(
                1 for t in tasks if t.question_type == qt and t.is_multimodal
            )

        # Diversity metrics
        if metrics.chunks_used > 0 and len(tasks) > 0:
            metrics.avg_chunk_usage = len(tasks) / metrics.chunks_used
        metrics.chunk_coverage = (
            metrics.chunks_used / metrics.total_chunks if metrics.total_chunks > 0 else 0.0
        )
        metrics.image_coverage = (
            metrics.images_used / metrics.total_images if metrics.total_images > 0 else 0.0
        )

        return metrics


def create_default_config(
    target_samples: int = 8000,
    qa_ratio: float = 0.30,
    qa_multimodal_ratio: float = 0.70,
    code_gen_ratio: float = 0.35,
    code_gen_multimodal_ratio: float = 0.30,
    func_comp_ratio: float = 0.35,
    func_comp_multimodal_ratio: float = 0.30,
    over_allocation_factor: float = 1.8,
) -> AllocationConfig:
    """Create allocation config with default ratios.

    Args:
        target_samples: Total samples to generate
        qa_ratio: Portion for QA type
        qa_multimodal_ratio: Multimodal portion within QA
        code_gen_ratio: Portion for code_generation type
        code_gen_multimodal_ratio: Multimodal portion within code_generation
        func_comp_ratio: Portion for function_completion type
        func_comp_multimodal_ratio: Multimodal portion within function_completion
        over_allocation_factor: Factor to over-allocate (1.8 = 80% more)
    """
    return AllocationConfig(
        target_samples=target_samples,
        type_configs={
            QuestionType.QA: TypeAllocationConfig(
                ratio=qa_ratio,
                multimodal_ratio=qa_multimodal_ratio,
            ),
            QuestionType.CODE_GENERATION: TypeAllocationConfig(
                ratio=code_gen_ratio,
                multimodal_ratio=code_gen_multimodal_ratio,
            ),
            QuestionType.FUNCTION_COMPLETION: TypeAllocationConfig(
                ratio=func_comp_ratio,
                multimodal_ratio=func_comp_multimodal_ratio,
            ),
        },
        over_allocation_factor=over_allocation_factor,
    )
