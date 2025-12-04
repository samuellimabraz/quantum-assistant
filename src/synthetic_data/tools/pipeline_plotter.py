"""Pipeline Plotter - Professional visualizations for pipeline stages.

Creates publication-quality plots for:
- Source analysis (files, types, sources)
- Image transcription and filtering
- Chunk analysis (sizes, code, images, quality)
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from synthetic_data.tools.pipeline_analyzer import PipelineStatistics


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

    TEAL = "#009D9A"
    CYAN = "#1192E8"
    MAGENTA = "#9F1853"

    @classmethod
    def palette(cls, n: int) -> list[str]:
        """Get n colors from the palette."""
        colors = [
            cls.PURPLE,
            cls.BLUE,
            cls.PURPLE_LIGHT,
            cls.BLUE_LIGHT,
            cls.TEAL,
            cls.CYAN,
            cls.MAGENTA,
            cls.PURPLE_DARK,
            cls.BLUE_DARK,
            cls.GRAY_70,
            cls.GRAY_50,
        ]
        if n <= len(colors):
            return colors[:n]
        return (colors * ((n // len(colors)) + 1))[:n]


class PipelinePlotter:
    """Generate professional visualizations for pipeline analysis."""

    def __init__(
        self,
        statistics: PipelineStatistics,
        output_dir: Path,
        width: int = 1200,
        height: int = 700,
    ):
        """Initialize plotter.

        Args:
            statistics: Pipeline statistics to visualize
            output_dir: Directory to save plots
            width: Default plot width
            height: Default plot height
        """
        self.stats = statistics
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _base_layout(
        self,
        title: str,
        xaxis_title: str = "",
        yaxis_title: str = "",
        showlegend: bool = True,
        height: int | None = None,
        width: int | None = None,
    ) -> dict:
        """Create base layout with Qiskit styling."""
        return {
            "title": {"text": title, "font": {"size": 18, "color": QiskitColors.BLACK}},
            "xaxis": {"title": xaxis_title, "tickfont": {"size": 11}},
            "yaxis": {"title": yaxis_title, "tickfont": {"size": 11}},
            "showlegend": showlegend,
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
            "plot_bgcolor": QiskitColors.WHITE,
            "paper_bgcolor": QiskitColors.WHITE,
            "font": {
                "family": "IBM Plex Sans, sans-serif",
                "size": 12,
                "color": QiskitColors.BLACK,
            },
            "margin": {"l": 60, "r": 40, "t": 80, "b": 60},
            "height": height or self.height,
            "width": width or self.width,
        }

    def plot_all(self) -> list[Path]:
        """Generate all pipeline plots."""
        plots = []

        if self.stats.source.total_files > 0:
            plots.append(self.plot_source_analysis())

        if self.stats.images.total_images > 0:
            plots.append(self.plot_image_analysis())

        if self.stats.chunks.total_chunks > 0:
            plots.append(self.plot_chunk_distribution())
            plots.append(self.plot_chunk_quality())

        return [p for p in plots if p is not None]

    def plot_source_analysis(self) -> Path:
        """Plot comprehensive source analysis with clean, professional layout.

        Shows files by type stacked by source directory, plus code and image distribution.
        """
        source = self.stats.source

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Files by Type (Stacked by Source)",
                "Code Blocks Distribution",
                "Images Distribution",
            ],
            horizontal_spacing=0.12,
        )

        # Get all unique sources and types
        all_sources = sorted(source.files_by_source.keys())
        all_types = sorted(source.files_by_type.keys())
        colors = QiskitColors.palette(len(all_sources))

        # Panel 1: Files by type, stacked by source
        for i, src in enumerate(all_sources):
            type_counts = source.files_by_source_and_type.get(src, {})
            counts = [type_counts.get(t, 0) for t in all_types]

            # Only show text for significant values
            texts = []
            for c in counts:
                if c > 0:
                    # Show number only if visible (>5% of total for that type)
                    total_for_type = source.files_by_type.get(all_types[counts.index(c)], 1)
                    if c / total_for_type >= 0.08:
                        texts.append(str(c))
                    else:
                        texts.append("")
                else:
                    texts.append("")

            fig.add_trace(
                go.Bar(
                    name=src,
                    x=all_types,
                    y=counts,
                    marker_color=colors[i],
                    text=texts,
                    textposition="inside",
                    textfont={"size": 10, "color": QiskitColors.WHITE},
                    legendgroup=src,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # Panel 2: Code blocks by source (clean bars with numbers inside)
        sorted_code = sorted(source.code_by_source.items(), key=lambda x: x[1], reverse=True)
        src_names = [s[0] for s in sorted_code]
        code_counts = [s[1] for s in sorted_code]
        src_colors = [colors[all_sources.index(s)] for s in src_names]

        fig.add_trace(
            go.Bar(
                x=src_names,
                y=code_counts,
                marker_color=src_colors,
                text=[f"{c:,}" for c in code_counts],
                textposition="outside",
                textfont={"size": 11, "color": QiskitColors.BLACK},
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Panel 3: Images by source (clean bars with numbers inside)
        sorted_images = sorted(source.images_by_source.items(), key=lambda x: x[1], reverse=True)
        img_src_names = [s[0] for s in sorted_images]
        img_counts = [s[1] for s in sorted_images]
        img_colors = [colors[all_sources.index(s)] for s in img_src_names]

        fig.add_trace(
            go.Bar(
                x=img_src_names,
                y=img_counts,
                marker_color=img_colors,
                text=[f"{c:,}" for c in img_counts],
                textposition="outside",
                textfont={"size": 11, "color": QiskitColors.BLACK},
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        # Update layout
        fig.update_layout(
            barmode="stack",
            title={
                "text": f"Source Analysis ({source.total_files:,} files, {source.total_code_blocks:,} code blocks, {source.total_images:,} images)",
                "font": {"size": 18, "color": QiskitColors.BLACK},
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.15,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 10},
            },
            plot_bgcolor=QiskitColors.WHITE,
            paper_bgcolor=QiskitColors.WHITE,
            font={"family": "IBM Plex Sans, sans-serif", "size": 12},
            height=500,
            width=self.width + 300,
            margin={"l": 80, "r": 80, "t": 100, "b": 100},
        )

        # Rotate x-axis labels for better readability
        fig.update_xaxes(title_text="File Type", tickangle=0, row=1, col=1)
        fig.update_yaxes(title_text="Number of Files", row=1, col=1)
        fig.update_xaxes(title_text="Source Directory", tickangle=20, row=1, col=2)
        fig.update_yaxes(title_text="Code Blocks", row=1, col=2)
        fig.update_xaxes(title_text="Source Directory", tickangle=20, row=1, col=3)
        fig.update_yaxes(title_text="Images", row=1, col=3)

        output_path = self.output_dir / "source_analysis.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_image_analysis(self) -> Path:
        """Plot image transcription and filtering analysis with clear visualizations."""
        images = self.stats.images

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Transcription Summary",
                "Image Type Distribution",
                "Filtering Impact by Type",
            ],
            horizontal_spacing=0.12,
        )

        # Panel 1: Transcription summary (vertical bars)
        categories = ["Total", "Transcribed", "Classified", "After Filter"]
        values = [
            images.total_images,
            images.transcribed_images,
            images.classified_images,
            images.images_after_filter if images.images_after_filter > 0 else images.total_images,
        ]
        bar_colors = [
            QiskitColors.GRAY_50,
            QiskitColors.BLUE,
            QiskitColors.PURPLE,
            QiskitColors.SUCCESS,
        ]

        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                marker_color=bar_colors,
                marker_line_color=QiskitColors.WHITE,
                marker_line_width=1,
                text=[f"{v:,}" for v in values],
                textposition="outside",
                textfont={"size": 12, "color": QiskitColors.BLACK},
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Panel 2: Image type distribution (horizontal bar)
        sorted_types = sorted(images.by_type.items(), key=lambda x: x[1], reverse=True)
        type_names = [t[0] for t in sorted_types]
        type_counts = [t[1] for t in sorted_types]
        type_colors = QiskitColors.palette(len(type_names))

        fig.add_trace(
            go.Bar(
                y=type_names,
                x=type_counts,
                orientation="h",
                marker_color=type_colors,
                marker_line_color=QiskitColors.WHITE,
                marker_line_width=1,
                text=[f"{c:,}" for c in type_counts],
                textposition="outside",
                textfont={"size": 11, "color": QiskitColors.BLACK},
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Panel 3: Filtering impact with clear grouped bars
        if images.by_type_after_filter:
            # Get only types that changed (more interesting)
            before_list = []
            after_list = []
            type_list = []

            for type_name, before_count in sorted_types:
                after_count = images.by_type_after_filter.get(type_name, 0)
                before_list.append(before_count)
                after_list.append(after_count)
                type_list.append(type_name)

            fig.add_trace(
                go.Bar(
                    name="Before Filter",
                    y=type_list,
                    x=before_list,
                    orientation="h",
                    marker_color=QiskitColors.GRAY_50,
                    marker_line_color=QiskitColors.WHITE,
                    marker_line_width=1,
                    text=[f"{c:,}" for c in before_list],
                    textposition="inside",
                    textfont={"size": 10, "color": QiskitColors.WHITE},
                    showlegend=True,
                ),
                row=1,
                col=3,
            )

            fig.add_trace(
                go.Bar(
                    name="After Filter",
                    y=type_list,
                    x=after_list,
                    orientation="h",
                    marker_color=QiskitColors.SUCCESS,
                    marker_line_color=QiskitColors.WHITE,
                    marker_line_width=1,
                    text=[f"{c:,}" for c in after_list],
                    textposition="inside",
                    textfont={"size": 10, "color": QiskitColors.WHITE},
                    showlegend=True,
                ),
                row=1,
                col=3,
            )

        fig.update_layout(
            barmode="group",
            title={
                "text": f"Image Transcription & Filtering ({images.total_images:,} total, {images.images_removed:,} removed)",
                "font": {"size": 18, "color": QiskitColors.BLACK},
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.08,
                "xanchor": "center",
                "x": 0.85,
                "font": {"size": 11},
            },
            plot_bgcolor=QiskitColors.WHITE,
            paper_bgcolor=QiskitColors.WHITE,
            font={"family": "IBM Plex Sans, sans-serif", "size": 12},
            height=500,
            width=self.width + 300,
            margin={"l": 110, "r": 80, "t": 110, "b": 60},
        )

        fig.update_xaxes(title_text="Images", row=1, col=1)
        fig.update_yaxes(title_text="Stage", row=1, col=1)
        fig.update_xaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(autorange="reversed", row=1, col=2)
        fig.update_xaxes(title_text="Count", row=1, col=3)
        fig.update_yaxes(autorange="reversed", row=1, col=3)

        output_path = self.output_dir / "image_analysis.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_chunk_distribution(self) -> Path:
        """Plot comprehensive chunk distribution analysis.

        Shows: size histogram (without outliers), code blocks distribution,
        image distribution, multimodal breakdown.
        """
        chunks = self.stats.chunks

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Chunk Size Distribution",
                "Code Blocks per Chunk",
                "Images per Chunk",
                "Multimodal vs Text-only by Source Type",
            ],
            vertical_spacing=0.18,
            horizontal_spacing=0.12,
        )

        # Panel 1: Chunk size histogram (clip outliers at 99th percentile)
        if chunks.chunk_sizes:
            # Remove outliers for better visualization
            p99 = np.percentile(chunks.chunk_sizes, 99)
            sizes_clipped = [min(s, p99) for s in chunks.chunk_sizes]

            fig.add_trace(
                go.Histogram(
                    x=sizes_clipped,
                    nbinsx=25,
                    marker_color=QiskitColors.PURPLE,
                    marker_line_color=QiskitColors.WHITE,
                    marker_line_width=1,
                    name="Chunk Size",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # Add statistical annotations
            mean_size = chunks.avg_chunk_size
            median_size = np.median(chunks.chunk_sizes)

            fig.add_vline(
                x=mean_size,
                line_dash="dash",
                line_color=QiskitColors.ERROR,
                line_width=2,
                annotation_text=f"Mean: {mean_size:.0f}",
                annotation_position="top",
                row=1,
                col=1,
            )

            fig.add_vline(
                x=median_size,
                line_dash="dot",
                line_color=QiskitColors.BLUE,
                line_width=2,
                annotation_text=f"Median: {median_size:.0f}",
                annotation_position="bottom right",
                row=1,
                col=1,
            )

        # Panel 2: Code blocks distribution
        if chunks.code_block_counts:
            code_dist = {}
            for c in chunks.code_block_counts:
                code_dist[c] = code_dist.get(c, 0) + 1

            x_vals = sorted(code_dist.keys())
            y_vals = [code_dist[x] for x in x_vals]

            fig.add_trace(
                go.Bar(
                    x=x_vals,
                    y=y_vals,
                    marker_color=QiskitColors.BLUE,
                    marker_line_color=QiskitColors.WHITE,
                    marker_line_width=1,
                    text=[f"{v:,}" for v in y_vals],
                    textposition="outside",
                    textfont={"size": 10, "color": QiskitColors.BLACK},
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        # Panel 3: Images per chunk distribution
        if chunks.image_counts:
            img_dist = {}
            for c in chunks.image_counts:
                img_dist[c] = img_dist.get(c, 0) + 1

            x_vals = sorted(img_dist.keys())
            y_vals = [img_dist[x] for x in x_vals]

            fig.add_trace(
                go.Bar(
                    x=x_vals,
                    y=y_vals,
                    marker_color=QiskitColors.PURPLE_LIGHT,
                    marker_line_color=QiskitColors.WHITE,
                    marker_line_width=1,
                    text=[f"{v:,}" for v in y_vals],
                    textposition="outside",
                    textfont={"size": 10, "color": QiskitColors.BLACK},
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        # Panel 4: Multimodal vs Text-only by source type (stacked bars)
        source_types = sorted(chunks.chunks_by_source_type.keys())

        multimodal_counts = [chunks.multimodal_by_source_type.get(st, 0) for st in source_types]
        text_only_counts = [
            chunks.chunks_by_source_type.get(st, 0) - chunks.multimodal_by_source_type.get(st, 0)
            for st in source_types
        ]

        fig.add_trace(
            go.Bar(
                name="Multimodal",
                x=source_types,
                y=multimodal_counts,
                marker_color=QiskitColors.PURPLE,
                marker_line_color=QiskitColors.WHITE,
                marker_line_width=1,
                text=[f"{c:,}" for c in multimodal_counts],
                textposition="inside",
                textfont={"size": 11, "color": QiskitColors.WHITE},
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                name="Text-only",
                x=source_types,
                y=text_only_counts,
                marker_color=QiskitColors.BLUE_LIGHT,
                marker_line_color=QiskitColors.WHITE,
                marker_line_width=1,
                text=[f"{c:,}" for c in text_only_counts],
                textposition="inside",
                textfont={"size": 11, "color": QiskitColors.WHITE},
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            barmode="stack",
            title={
                "text": (
                    f"Chunk Distribution Analysis ({chunks.total_chunks:,} chunks, "
                    f"{chunks.multimodal_chunks:,} multimodal, {chunks.unique_images:,} unique images)"
                ),
                "font": {"size": 18, "color": QiskitColors.BLACK},
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.05,
                "xanchor": "right",
                "x": 1,
            },
            plot_bgcolor=QiskitColors.WHITE,
            paper_bgcolor=QiskitColors.WHITE,
            font={"family": "IBM Plex Sans, sans-serif", "size": 12},
            height=750,
            width=self.width + 100,
            margin={"l": 70, "r": 70, "t": 100, "b": 70},
        )

        fig.update_xaxes(title_text="Characters", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Code Blocks per Chunk", row=1, col=2)
        fig.update_yaxes(title_text="Number of Chunks", row=1, col=2)
        fig.update_xaxes(title_text="Images per Chunk", row=2, col=1)
        fig.update_yaxes(title_text="Number of Chunks", row=2, col=1)
        fig.update_xaxes(title_text="Source Type", row=2, col=2)
        fig.update_yaxes(title_text="Number of Chunks", row=2, col=2)

        output_path = self.output_dir / "chunk_distribution.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_chunk_quality(self) -> Path:
        """Plot chunk quality analysis with professional styling.

        Shows: quality distribution, image types in chunks, filtering impact.
        """
        chunks = self.stats.chunks

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Chunk Quality Distribution",
                "Image Types in Chunks",
                "Chunk Filtering Impact",
            ],
            horizontal_spacing=0.14,
        )

        # Panel 1: Quality distribution (horizontal bar)
        quality_labels = ["High", "Medium", "Low"]
        quality_values = [chunks.quality_high, chunks.quality_medium, chunks.quality_low]
        quality_colors = [QiskitColors.SUCCESS, QiskitColors.WARNING, QiskitColors.ERROR]

        total = sum(quality_values)
        quality_texts = [
            f"{v:,} ({v/total*100:.1f}%)" if total > 0 else f"{v:,}" for v in quality_values
        ]

        fig.add_trace(
            go.Bar(
                y=quality_labels,
                x=quality_values,
                orientation="h",
                marker_color=quality_colors,
                marker_line_color=QiskitColors.WHITE,
                marker_line_width=1,
                text=quality_texts,
                textposition="outside",
                textfont={"size": 12, "color": QiskitColors.BLACK},
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Panel 2: Image types in chunks (horizontal bar)
        sorted_types = sorted(
            chunks.image_types_in_chunks.items(), key=lambda x: x[1], reverse=True
        )
        type_names = [t[0] for t in sorted_types]
        type_counts = [t[1] for t in sorted_types]
        type_colors = QiskitColors.palette(len(type_names))

        fig.add_trace(
            go.Bar(
                y=type_names,
                x=type_counts,
                orientation="h",
                marker_color=type_colors,
                marker_line_color=QiskitColors.WHITE,
                marker_line_width=1,
                text=[f"{c:,}" for c in type_counts],
                textposition="outside",
                textfont={"size": 11, "color": QiskitColors.BLACK},
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Panel 3: Filtering impact (clean grouped bars)
        filter_categories = ["Chunks", "Multimodal\nChunks", "Unique\nImages"]
        before_values = [chunks.total_chunks, chunks.multimodal_chunks, chunks.unique_images]
        after_values = [
            chunks.chunks_after_filter if chunks.chunks_after_filter > 0 else chunks.total_chunks,
            (
                chunks.multimodal_after_filter
                if chunks.multimodal_after_filter > 0
                else chunks.multimodal_chunks
            ),
            chunks.images_after_filter if chunks.images_after_filter > 0 else chunks.unique_images,
        ]

        fig.add_trace(
            go.Bar(
                name="Before",
                x=filter_categories,
                y=before_values,
                marker_color=QiskitColors.GRAY_50,
                marker_line_color=QiskitColors.WHITE,
                marker_line_width=1,
                text=[f"{v:,}" for v in before_values],
                textposition="outside",
                textfont={"size": 11, "color": QiskitColors.BLACK},
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Bar(
                name="After",
                x=filter_categories,
                y=after_values,
                marker_color=QiskitColors.SUCCESS,
                marker_line_color=QiskitColors.WHITE,
                marker_line_width=1,
                text=[f"{v:,}" for v in after_values],
                textposition="outside",
                textfont={"size": 11, "color": QiskitColors.BLACK},
            ),
            row=1,
            col=3,
        )

        removed_values = [b - a for b, a in zip(before_values, after_values)]
        removed_text = f"Removed: {removed_values[0]:,} chunks, {removed_values[2]:,} images"

        fig.update_layout(
            barmode="group",
            title={
                "text": f"Chunk Quality & Filtering ({removed_text})",
                "font": {"size": 18, "color": QiskitColors.BLACK},
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.08,
                "xanchor": "center",
                "x": 0.85,
                "font": {"size": 11},
            },
            plot_bgcolor=QiskitColors.WHITE,
            paper_bgcolor=QiskitColors.WHITE,
            font={"family": "IBM Plex Sans, sans-serif", "size": 12},
            height=500,
            width=self.width + 400,
            margin={"l": 110, "r": 120, "t": 110, "b": 60},
        )

        fig.update_xaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Quality Level", autorange="reversed", row=1, col=1)
        fig.update_xaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Image Type", autorange="reversed", row=1, col=2)
        fig.update_xaxes(title_text="Category", row=1, col=3)
        fig.update_yaxes(title_text="Count", row=1, col=3)

        output_path = self.output_dir / "chunk_quality.png"
        fig.write_image(str(output_path), scale=2)
        return output_path
