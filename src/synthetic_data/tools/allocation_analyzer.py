"""Allocation Analyzer Tool.

Analyzes filtered chunks and simulates allocation strategies to find optimal
generation configurations for the diversity-aware allocator.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from synthetic_data.config import QuestionType
from synthetic_data.generators.allocation import (
    AllocationConfig,
    Allocator,
    ChunkScorer,
    TypeAllocationConfig,
)


@dataclass
class ChunkAnalysis:
    """Analysis of chunk characteristics."""

    total_chunks: int = 0
    multimodal_chunks: int = 0
    text_only_chunks: int = 0
    total_images: int = 0
    unique_images: int = 0
    chunks_with_code: int = 0

    code_score_avg: float = 0.0
    code_score_high: int = 0
    code_score_medium: int = 0
    code_score_low: int = 0

    qa_score_avg: float = 0.0
    qa_score_high: int = 0
    qa_score_medium: int = 0
    qa_score_low: int = 0

    image_types: dict[str, int] = field(default_factory=dict)


@dataclass
class AllocationSimulation:
    """Results of a single allocation simulation."""

    config_name: str
    target_samples: int
    over_allocation_factor: float
    diversity_weight: float

    total_allocated: int = 0
    multimodal_allocated: int = 0
    text_only_allocated: int = 0

    by_type: dict[QuestionType, int] = field(default_factory=dict)
    multimodal_by_type: dict[QuestionType, int] = field(default_factory=dict)

    chunk_coverage: float = 0.0
    image_coverage: float = 0.0
    avg_chunk_usage: float = 0.0
    unique_chunks_used: int = 0
    unique_images_used: int = 0


class QiskitColors:
    """Qiskit brand color palette."""

    PURPLE = "#6929C4"
    BLUE = "#0043CE"
    WHITE = "#FFFFFF"
    BLACK = "#000000"

    PURPLE_LIGHT = "#8A3FFC"
    PURPLE_DARK = "#491D8B"
    BLUE_LIGHT = "#4589FF"
    BLUE_DARK = "#002D9C"

    GRAY_10 = "#F4F4F4"
    GRAY_20 = "#E0E0E0"
    GRAY_50 = "#8D8D8D"
    GRAY_70 = "#525252"
    GRAY_90 = "#262626"

    SUCCESS = "#24A148"
    WARNING = "#F1C21B"
    ERROR = "#DA1E28"

    @classmethod
    def type_palette(cls) -> dict[str, str]:
        """Colors for question types."""
        return {
            "qa": cls.PURPLE,
            "code_generation": cls.BLUE,
            "function_completion": cls.PURPLE_LIGHT,
        }

    @classmethod
    def modality_palette(cls) -> dict[str, str]:
        """Colors for modality."""
        return {
            "multimodal": cls.PURPLE,
            "text_only": cls.BLUE_LIGHT,
        }

    @classmethod
    def gradient(cls, n: int) -> list[str]:
        """Generate n colors from purple to blue gradient."""
        colors = [
            cls.PURPLE,
            cls.PURPLE_LIGHT,
            cls.BLUE,
            cls.BLUE_LIGHT,
            cls.GRAY_50,
            cls.GRAY_70,
        ]
        return colors[:n] if n <= len(colors) else colors * (n // len(colors) + 1)


class ChunkAnalyzer:
    """Analyzes chunk characteristics for capacity planning."""

    def __init__(self, chunks: list):
        self.chunks = chunks
        self._scorer = ChunkScorer()

    def analyze(self) -> ChunkAnalysis:
        """Compute chunk analysis."""
        analysis = ChunkAnalysis()
        analysis.total_chunks = len(self.chunks)

        unique_image_ids = set()
        code_scores = []
        qa_scores = []

        for chunk in self.chunks:
            if chunk.transcribed_images:
                analysis.multimodal_chunks += 1
            else:
                analysis.text_only_chunks += 1

            for img in chunk.transcribed_images:
                analysis.total_images += 1
                if img.image_id:
                    unique_image_ids.add(img.image_id)
                img_type = img.image_type.value if img.image_type else "unknown"
                analysis.image_types[img_type] = analysis.image_types.get(img_type, 0) + 1

            if chunk.code_blocks:
                analysis.chunks_with_code += 1

            code_score = self._scorer.compute_code_score(chunk)
            qa_score = self._scorer.compute_qa_score(chunk)
            code_scores.append(code_score)
            qa_scores.append(qa_score)

            if code_score >= 0.5:
                analysis.code_score_high += 1
            elif code_score >= 0.2:
                analysis.code_score_medium += 1
            else:
                analysis.code_score_low += 1

            if qa_score >= 0.5:
                analysis.qa_score_high += 1
            elif qa_score >= 0.2:
                analysis.qa_score_medium += 1
            else:
                analysis.qa_score_low += 1

        analysis.unique_images = len(unique_image_ids)
        analysis.code_score_avg = sum(code_scores) / len(code_scores) if code_scores else 0
        analysis.qa_score_avg = sum(qa_scores) / len(qa_scores) if qa_scores else 0

        return analysis


class AllocationSimulator:
    """Simulates allocation strategies with different parameters."""

    def __init__(self, chunks: list):
        self.chunks = chunks

    def simulate(
        self,
        target_samples: int,
        type_configs: dict[QuestionType, TypeAllocationConfig],
        over_allocation_factor: float = 1.8,
        diversity_weight: float = 0.4,
        config_name: str = "default",
    ) -> AllocationSimulation:
        """Run a single allocation simulation."""
        config = AllocationConfig(
            target_samples=target_samples,
            type_configs=type_configs,
            over_allocation_factor=over_allocation_factor,
        )

        allocator = Allocator(config, diversity_weight=diversity_weight)
        result = allocator.allocate(self.chunks, use_over_allocation=True)

        return AllocationSimulation(
            config_name=config_name,
            target_samples=target_samples,
            over_allocation_factor=over_allocation_factor,
            diversity_weight=diversity_weight,
            total_allocated=result.total_samples,
            multimodal_allocated=result.multimodal_samples,
            text_only_allocated=result.text_only_samples,
            by_type=result.samples_by_type(),
            multimodal_by_type=result.multimodal_by_type(),
            chunk_coverage=result.metrics.chunk_coverage,
            image_coverage=result.metrics.image_coverage,
            avg_chunk_usage=result.metrics.avg_chunk_usage,
            unique_chunks_used=result.unique_chunks_used(),
            unique_images_used=result.unique_images_used(),
        )

    def sweep_diversity_weight(
        self,
        target_samples: int,
        type_configs: dict[QuestionType, TypeAllocationConfig],
        weights: Optional[list[float]] = None,
    ) -> list[AllocationSimulation]:
        """Sweep diversity weight parameter."""
        if weights is None:
            weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        return [
            self.simulate(
                target_samples,
                type_configs,
                diversity_weight=w,
                config_name=f"div_{w:.1f}",
            )
            for w in weights
        ]

    def sweep_over_allocation(
        self,
        target_samples: int,
        type_configs: dict[QuestionType, TypeAllocationConfig],
        factors: Optional[list[float]] = None,
    ) -> list[AllocationSimulation]:
        """Sweep over-allocation factor."""
        if factors is None:
            factors = [1.0, 1.4, 1.8, 2.0, 2.5]

        return [
            self.simulate(
                target_samples,
                type_configs,
                over_allocation_factor=f,
                config_name=f"alloc_{f:.1f}x",
            )
            for f in factors
        ]

    def sweep_target_samples(
        self,
        type_configs: dict[QuestionType, TypeAllocationConfig],
        targets: Optional[list[int]] = None,
    ) -> list[AllocationSimulation]:
        """Sweep target sample count."""
        if targets is None:
            targets = [2000, 4000, 6000, 8000, 10000, 12000]

        return [
            self.simulate(
                t,
                type_configs,
                config_name=f"target_{t}",
            )
            for t in targets
        ]


class AllocationPlotter:
    """Generates professional visualizations for allocation analysis.

    Focuses on diversity and target sweep plots that demonstrate
    the effectiveness of the diversity-aware allocation algorithm.
    """

    def __init__(self, output_dir: Path, width: int = 1100, height: int = 550):
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _base_layout(self, title: str, **kwargs) -> dict:
        """Create base layout with Qiskit styling."""
        return {
            "title": {"text": title, "font": {"size": 18, "color": QiskitColors.BLACK}},
            "plot_bgcolor": QiskitColors.WHITE,
            "paper_bgcolor": QiskitColors.WHITE,
            "font": {
                "family": "IBM Plex Sans, sans-serif",
                "size": 12,
                "color": QiskitColors.BLACK,
            },
            "margin": {"l": 70, "r": 40, "t": 90, "b": 70},
            "width": kwargs.get("width", self.width),
            "height": kwargs.get("height", self.height),
            **kwargs,
        }

    def plot_diversity_sweep(self, simulations: list[AllocationSimulation]) -> Path:
        """Plot diversity weight sweep results with point values.

        Shows how diversity weight affects chunk/image coverage and reuse factor.
        Values are displayed at each data point for clarity.
        """
        weights = [s.diversity_weight for s in simulations]
        chunk_cov = [s.chunk_coverage * 100 for s in simulations]
        image_cov = [s.image_coverage * 100 for s in simulations]
        avg_usage = [s.avg_chunk_usage for s in simulations]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Coverage vs Diversity Weight", "Chunk Reuse vs Diversity Weight"],
            horizontal_spacing=0.12,
        )

        # Panel 1: Coverage lines with point values
        fig.add_trace(
            go.Scatter(
                x=weights,
                y=chunk_cov,
                mode="lines+markers+text",
                name="Chunk Coverage",
                line={"color": QiskitColors.PURPLE, "width": 3},
                marker={"size": 12, "symbol": "circle"},
                text=[f"{v:.1f}%" for v in chunk_cov],
                textposition="top center",
                textfont={"size": 10, "color": QiskitColors.PURPLE},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=weights,
                y=image_cov,
                mode="lines+markers+text",
                name="Image Coverage",
                line={"color": QiskitColors.BLUE, "width": 3},
                marker={"size": 12, "symbol": "diamond"},
                text=[f"{v:.1f}%" for v in image_cov],
                textposition="bottom center",
                textfont={"size": 10, "color": QiskitColors.BLUE},
            ),
            row=1,
            col=1,
        )

        # Panel 2: Reuse factor with point values
        fig.add_trace(
            go.Scatter(
                x=weights,
                y=avg_usage,
                mode="lines+markers+text",
                name="Avg Chunk Reuse",
                line={"color": QiskitColors.PURPLE_LIGHT, "width": 3},
                marker={"size": 12, "symbol": "square"},
                text=[f"{v:.2f}x" for v in avg_usage],
                textposition="top center",
                textfont={"size": 10, "color": QiskitColors.PURPLE_DARK},
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Diversity Weight", row=1, col=1)
        fig.update_xaxes(title_text="Diversity Weight", row=1, col=2)
        fig.update_yaxes(title_text="Coverage (%)", row=1, col=1, range=[0, 105])
        fig.update_yaxes(title_text="Reuse Factor (×)", row=1, col=2)

        fig.update_layout(
            **self._base_layout(
                "Diversity Weight Impact on Resource Utilization",
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.05,
                    "xanchor": "center",
                    "x": 0.3,
                },
            )
        )

        output_path = self.output_dir / "diversity_sweep.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_target_sweep(self, simulations: list[AllocationSimulation]) -> Path:
        """Plot target samples sweep results with point values.

        Shows how target sample count affects coverage and chunk reuse.
        Values are displayed at each data point for interpretation.
        """
        targets = [s.target_samples for s in simulations]
        chunk_cov = [s.chunk_coverage * 100 for s in simulations]
        image_cov = [s.image_coverage * 100 for s in simulations]
        avg_usage = [s.avg_chunk_usage for s in simulations]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Coverage by Target Samples", "Chunk Reuse by Target Samples"],
            horizontal_spacing=0.12,
        )

        # Panel 1: Coverage lines with point values
        fig.add_trace(
            go.Scatter(
                x=targets,
                y=chunk_cov,
                mode="lines+markers+text",
                name="Chunk Coverage",
                line={"color": QiskitColors.PURPLE, "width": 3},
                marker={"size": 12, "symbol": "circle"},
                text=[f"{v:.1f}%" for v in chunk_cov],
                textposition="top center",
                textfont={"size": 10, "color": QiskitColors.PURPLE},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=targets,
                y=image_cov,
                mode="lines+markers+text",
                name="Image Coverage",
                line={"color": QiskitColors.BLUE, "width": 3},
                marker={"size": 12, "symbol": "diamond"},
                text=[f"{v:.1f}%" for v in image_cov],
                textposition="bottom center",
                textfont={"size": 10, "color": QiskitColors.BLUE},
            ),
            row=1,
            col=1,
        )

        # Panel 2: Reuse factor with point values
        fig.add_trace(
            go.Scatter(
                x=targets,
                y=avg_usage,
                mode="lines+markers+text",
                name="Avg Chunk Reuse",
                line={"color": QiskitColors.PURPLE_LIGHT, "width": 3},
                marker={"size": 12, "symbol": "square"},
                text=[f"{v:.2f}x" for v in avg_usage],
                textposition="top center",
                textfont={"size": 10, "color": QiskitColors.PURPLE_DARK},
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Format x-axis with K notation
        fig.update_xaxes(
            title_text="Target Samples",
            tickvals=targets,
            ticktext=[f"{t//1000}K" if t >= 1000 else str(t) for t in targets],
            row=1,
            col=1,
        )
        fig.update_xaxes(
            title_text="Target Samples",
            tickvals=targets,
            ticktext=[f"{t//1000}K" if t >= 1000 else str(t) for t in targets],
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Coverage (%)", row=1, col=1, range=[0, 105])
        fig.update_yaxes(title_text="Reuse Factor (×)", row=1, col=2)

        fig.update_layout(
            **self._base_layout(
                "Target Sample Count Impact on Resource Utilization",
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.05,
                    "xanchor": "center",
                    "x": 0.3,
                },
            )
        )

        output_path = self.output_dir / "target_sweep.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_all(
        self,
        _analysis: ChunkAnalysis,
        diversity_sweep: list[AllocationSimulation],
        target_sweep: list[AllocationSimulation],
    ) -> list[Path]:
        """Generate allocation sweep plots.

        Args:
            _analysis: Chunk analysis (unused, kept for API compatibility)
            diversity_sweep: Diversity weight sweep simulations
            target_sweep: Target samples sweep simulations

        Returns:
            List of generated plot paths
        """
        return [
            self.plot_diversity_sweep(diversity_sweep),
            self.plot_target_sweep(target_sweep),
        ]


def load_chunks(chunks_path: Path) -> list:
    """Load filtered chunks from pickle file."""
    with open(chunks_path, "rb") as f:
        return pickle.load(f)


def get_default_type_configs() -> dict[QuestionType, TypeAllocationConfig]:
    """Get default type allocation configs."""
    return {
        QuestionType.QA: TypeAllocationConfig(ratio=0.30, multimodal_ratio=0.70),
        QuestionType.CODE_GENERATION: TypeAllocationConfig(ratio=0.35, multimodal_ratio=0.30),
        QuestionType.FUNCTION_COMPLETION: TypeAllocationConfig(ratio=0.35, multimodal_ratio=0.30),
    }


def run_analysis(
    chunks_path: Path,
    output_dir: Path,
    target_samples: int = 8000,
    type_configs: Optional[dict[QuestionType, TypeAllocationConfig]] = None,
) -> tuple[ChunkAnalysis, AllocationSimulation, list[Path]]:
    """Run complete allocation analysis and generate plots.

    Returns:
        Tuple of (chunk_analysis, current_simulation, plot_paths)
    """
    if type_configs is None:
        type_configs = get_default_type_configs()

    chunks = load_chunks(chunks_path)

    # Analyze chunks
    analyzer = ChunkAnalyzer(chunks)
    analysis = analyzer.analyze()

    # Run simulations
    simulator = AllocationSimulator(chunks)
    diversity_sweep = simulator.sweep_diversity_weight(target_samples, type_configs)
    target_sweep = simulator.sweep_target_samples(type_configs)
    current_sim = simulator.simulate(target_samples, type_configs, config_name="current")

    # Generate plots (diversity and target sweeps only)
    plotter = AllocationPlotter(output_dir)
    plot_paths = plotter.plot_all(analysis, diversity_sweep, target_sweep)

    return analysis, current_sim, plot_paths


def main():
    """Run allocation analysis from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze allocation strategies")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("outputs/filtered/chunks.pkl"),
        help="Path to filtered chunks pickle file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=8000,
        help="Target number of samples",
    )

    args = parser.parse_args()

    if not args.chunks_path.is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        chunks_path = project_root / args.chunks_path
        output_dir = project_root / args.output_dir
    else:
        chunks_path = args.chunks_path
        output_dir = args.output_dir

    if not chunks_path.exists():
        print(f"Error: Chunks file not found at {chunks_path}")
        return 1

    print(f"Loading chunks from: {chunks_path}")
    analysis, simulation, plot_paths = run_analysis(chunks_path, output_dir, args.target)

    print("\nChunk Analysis:")
    print(f"  Total: {analysis.total_chunks:,}")
    print(
        f"  Multimodal: {analysis.multimodal_chunks:,} ({analysis.multimodal_chunks/analysis.total_chunks*100:.1f}%)"
    )
    print(f"  Images: {analysis.unique_images:,}")

    print("\nAllocation Simulation:")
    print(f"  Target: {simulation.target_samples:,}")
    print(f"  Allocated: {simulation.total_allocated:,}")
    print(f"  Chunk coverage: {simulation.chunk_coverage*100:.1f}%")
    print(f"  Image coverage: {simulation.image_coverage*100:.1f}%")

    print(f"\nGenerated {len(plot_paths)} plots:")
    for path in plot_paths:
        print(f"  - {path.name}")

    return 0


if __name__ == "__main__":
    exit(main())
