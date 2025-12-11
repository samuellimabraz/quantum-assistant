from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from synthetic_data.tools.pipeline_analyzer import PipelineStatistics
from synthetic_data.utils.colors import AcademicPalette as Colors, PlotStyle as Style


class PipelinePlotter:

    def __init__(
        self,
        statistics: PipelineStatistics,
        output_dir: Path,
        width: int = 1000,
        height: int = 500,
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
        """Create professional base layout."""
        return Style.base_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            showlegend=showlegend,
            height=height or self.height,
            width=width or self.width,
        )

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
        source = self.stats.source

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Files by Type",
                "Code Blocks by Source",
                "Images by Source",
            ],
            horizontal_spacing=0.12,
        )

        
        all_sources = sorted(source.files_by_source.keys())
        all_types = sorted(source.files_by_type.keys())
        colors = Colors.categorical(len(all_sources))

        for i, src in enumerate(all_sources):
            type_counts = source.files_by_source_and_type.get(src, {})
            counts = [type_counts.get(t, 0) for t in all_types]

            texts = []
            for j, c in enumerate(counts):
                if c > 0:
                    total_for_type = source.files_by_type.get(all_types[j], 1)
                    texts.append(str(c) if c / total_for_type >= 0.10 else "")
                else:
                    texts.append("")

            fig.add_trace(
                go.Bar(
                    name=src[:12] + "..." if len(src) > 12 else src,
                    x=all_types,
                    y=counts,
                    marker_color=colors[i],
                    marker_line_color=Colors.WHITE,
                    marker_line_width=0.5,
                    text=texts,
                    textposition="inside",
                    textfont={"size": 9, "color": Colors.WHITE},
                    legendgroup=src,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        sorted_code = sorted(source.code_by_source.items(), key=lambda x: x[1], reverse=True)
        src_names = [s[0][:12] for s in sorted_code]
        code_counts = [s[1] for s in sorted_code]
        src_colors = [colors[all_sources.index(s[0])] for s in sorted_code]

        fig.add_trace(
            go.Bar(
                x=src_names,
                y=code_counts,
                marker_color=src_colors,
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{c:,}" for c in code_counts],
                textposition="outside",
                textfont={"size": Style.ANNOTATION_SIZE, "color": Colors.TEXT_SECONDARY},
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        sorted_images = sorted(source.images_by_source.items(), key=lambda x: x[1], reverse=True)
        img_src_names = [s[0][:12] for s in sorted_images]
        img_counts = [s[1] for s in sorted_images]
        img_colors = [colors[all_sources.index(s[0])] for s in sorted_images]

        fig.add_trace(
            go.Bar(
                x=img_src_names,
                y=img_counts,
                marker_color=img_colors,
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{c:,}" for c in img_counts],
                textposition="outside",
                textfont={"size": Style.ANNOTATION_SIZE, "color": Colors.TEXT_SECONDARY},
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        fig.update_layout(
            barmode="stack",
            title={
                "text": f"Source Analysis ({source.total_files:,} files, {source.total_code_blocks:,} code, {source.total_images:,} images)",
                "font": {"size": Style.TITLE_SIZE, "color": Colors.TEXT_PRIMARY},
                "x": 0.5,
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.18,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 9, "color": Colors.TEXT_SECONDARY},
            },
            plot_bgcolor=Colors.WHITE,
            paper_bgcolor=Colors.WHITE,
            font={"family": Style.FONT_FAMILY, "size": Style.TICK_SIZE},
            height=480,
            width=self.width + 200,
            margin={"l": 70, "r": 50, "t": 70, "b": 100},
        )

        fig.update_xaxes(title_text="File Type", tickangle=0, row=1, col=1)
        fig.update_yaxes(title_text="Files", row=1, col=1)
        fig.update_xaxes(title_text="Source", tickangle=20, row=1, col=2)
        fig.update_yaxes(title_text="Code Blocks", row=1, col=2)
        fig.update_xaxes(title_text="Source", tickangle=20, row=1, col=3)
        fig.update_yaxes(title_text="Images", row=1, col=3)

        output_path = self.output_dir / "source_analysis.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_image_analysis(self) -> Path:
        images = self.stats.images

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Processing Summary",
                "Image Types",
                "Filter Impact by Type",
            ],
            horizontal_spacing=0.12,
        )

        categories = ["Total", "Transcribed", "Classified", "Retained"]
        values = [
            images.total_images,
            images.transcribed_images,
            images.classified_images,
            images.images_after_filter if images.images_after_filter > 0 else images.total_images,
        ]
        bar_colors = [
            Colors.LIGHT,
            Colors.SECONDARY,
            Colors.PRIMARY,
            Colors.SUCCESS,
        ]

        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                marker_color=bar_colors,
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{v:,}" for v in values],
                textposition="outside",
                textfont={"size": Style.ANNOTATION_SIZE, "color": Colors.TEXT_SECONDARY},
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        sorted_types = sorted(images.by_type.items(), key=lambda x: x[1], reverse=True)
        type_names = [t[0] for t in sorted_types]
        type_counts = [t[1] for t in sorted_types]
        type_colors = Colors.categorical(len(type_names))

        fig.add_trace(
            go.Bar(
                y=type_names,
                x=type_counts,
                orientation="h",
                marker_color=type_colors,
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{c:,}" for c in type_counts],
                textposition="outside",
                textfont={"size": Style.ANNOTATION_SIZE, "color": Colors.TEXT_SECONDARY},
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        if images.by_type_after_filter:
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
                    name="Before",
                    y=type_list,
                    x=before_list,
                    orientation="h",
                    marker_color=Colors.LIGHT,
                    marker_line_color=Colors.WHITE,
                    marker_line_width=0.5,
                    text=[f"{c:,}" for c in before_list],
                    textposition="inside",
                    textfont={"size": 9, "color": Colors.TEXT_PRIMARY},
                    showlegend=True,
                ),
                row=1,
                col=3,
            )

            fig.add_trace(
                go.Bar(
                    name="After",
                    y=type_list,
                    x=after_list,
                    orientation="h",
                    marker_color=Colors.SUCCESS,
                    marker_line_color=Colors.WHITE,
                    marker_line_width=0.5,
                    text=[f"{c:,}" for c in after_list],
                    textposition="inside",
                    textfont={"size": 9, "color": Colors.WHITE},
                    showlegend=True,
                ),
                row=1,
                col=3,
            )

        removed_pct = (
            (images.images_removed / images.total_images * 100) if images.total_images > 0 else 0
        )

        fig.update_layout(
            barmode="group",
            title={
                "text": f"Image Analysis ({images.total_images:,} total, {images.images_removed:,} filtered, {removed_pct:.1f}% removed)",
                "font": {"size": Style.TITLE_SIZE, "color": Colors.TEXT_PRIMARY},
                "x": 0.5,
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.06,
                "xanchor": "center",
                "x": 0.85,
                "font": {"size": Style.TICK_SIZE, "color": Colors.TEXT_SECONDARY},
            },
            plot_bgcolor=Colors.WHITE,
            paper_bgcolor=Colors.WHITE,
            font={"family": Style.FONT_FAMILY, "size": Style.TICK_SIZE},
            height=480,
            width=self.width + 200,
            margin={"l": 100, "r": 60, "t": 90, "b": 60},
        )

        fig.update_xaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(autorange="reversed", row=1, col=2)
        fig.update_xaxes(title_text="Count", row=1, col=3)
        fig.update_yaxes(autorange="reversed", row=1, col=3)

        output_path = self.output_dir / "image_analysis.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_chunk_distribution(self) -> Path:
        chunks = self.stats.chunks

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Chunk Size Distribution",
                "Code Blocks per Chunk",
                "Images per Chunk",
                "Composition by Source Type",
            ],
            vertical_spacing=0.18,
            horizontal_spacing=0.12,
        )

        if chunks.chunk_sizes:
            p99 = np.percentile(chunks.chunk_sizes, 99)
            sizes_clipped = [min(s, p99) for s in chunks.chunk_sizes]

            fig.add_trace(
                go.Histogram(
                    x=sizes_clipped,
                    nbinsx=25,
                    marker_color=Colors.PRIMARY,
                    marker_line_color=Colors.WHITE,
                    marker_line_width=0.5,
                    name="Size",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            mean_size = chunks.avg_chunk_size
            median_size = float(np.median(chunks.chunk_sizes))

            fig.add_vline(
                x=mean_size,
                line_dash="dash",
                line_color=Colors.ACCENT_1,
                line_width=1.5,
                annotation_text=f"μ={mean_size:.0f}",
                annotation_position="top",
                annotation_font_size=9,
                row=1,
                col=1,
            )

            fig.add_vline(
                x=median_size,
                line_dash="dot",
                line_color=Colors.ACCENT_2,
                line_width=1.5,
                annotation_text=f"med={median_size:.0f}",
                annotation_position="bottom right",
                annotation_font_size=9,
                row=1,
                col=1,
            )

        # Panel 2: Code blocks distribution
        if chunks.code_block_counts:
            code_dist: dict[int, int] = {}
            for c in chunks.code_block_counts:
                code_dist[c] = code_dist.get(c, 0) + 1

            x_vals = sorted(code_dist.keys())
            y_vals = [code_dist[x] for x in x_vals]

            fig.add_trace(
                go.Bar(
                    x=x_vals,
                    y=y_vals,
                    marker_color=Colors.SECONDARY,
                    marker_line_color=Colors.WHITE,
                    marker_line_width=0.5,
                    text=[f"{v:,}" for v in y_vals],
                    textposition="outside",
                    textfont={"size": 9, "color": Colors.TEXT_SECONDARY},
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        # Panel 3: Images per chunk
        if chunks.image_counts:
            img_dist: dict[int, int] = {}
            for c in chunks.image_counts:
                img_dist[c] = img_dist.get(c, 0) + 1

            x_vals = sorted(img_dist.keys())
            y_vals = [img_dist[x] for x in x_vals]

            fig.add_trace(
                go.Bar(
                    x=x_vals,
                    y=y_vals,
                    marker_color=Colors.TERTIARY,
                    marker_line_color=Colors.WHITE,
                    marker_line_width=0.5,
                    text=[f"{v:,}" for v in y_vals],
                    textposition="outside",
                    textfont={"size": 9, "color": Colors.TEXT_SECONDARY},
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        # Panel 4: Multimodal vs Text-only by source type
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
                marker_color=Colors.PRIMARY,
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{c:,}" for c in multimodal_counts],
                textposition="inside",
                textfont={"size": 10, "color": Colors.WHITE},
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                name="Text-only",
                x=source_types,
                y=text_only_counts,
                marker_color=Colors.LIGHT,
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{c:,}" for c in text_only_counts],
                textposition="inside",
                textfont={"size": 10, "color": Colors.TEXT_PRIMARY},
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            barmode="stack",
            title={
                "text": f"Chunk Distribution ({chunks.total_chunks:,} chunks, {chunks.multimodal_chunks:,} multimodal)",
                "font": {"size": Style.TITLE_SIZE, "color": Colors.TEXT_PRIMARY},
                "x": 0.5,
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.04,
                "xanchor": "right",
                "x": 1,
                "font": {"size": Style.TICK_SIZE, "color": Colors.TEXT_SECONDARY},
            },
            plot_bgcolor=Colors.WHITE,
            paper_bgcolor=Colors.WHITE,
            font={"family": Style.FONT_FAMILY, "size": Style.TICK_SIZE},
            height=650,
            width=self.width + 50,
            margin={"l": 70, "r": 50, "t": 80, "b": 60},
        )

        fig.update_xaxes(title_text="Characters", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Code Blocks", row=1, col=2)
        fig.update_yaxes(title_text="Chunks", row=1, col=2)
        fig.update_xaxes(title_text="Images", row=2, col=1)
        fig.update_yaxes(title_text="Chunks", row=2, col=1)
        fig.update_xaxes(title_text="Source Type", row=2, col=2)
        fig.update_yaxes(title_text="Chunks", row=2, col=2)

        output_path = self.output_dir / "chunk_distribution.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_chunk_quality(self) -> Path:
        """Plot chunk quality analysis."""
        chunks = self.stats.chunks

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Quality Distribution",
                "Image Types in Chunks",
                "Filter Impact",
            ],
            horizontal_spacing=0.14,
        )

        # Panel 1: Quality distribution
        quality_labels = ["High", "Medium", "Low"]
        quality_values = [chunks.quality_high, chunks.quality_medium, chunks.quality_low]
        quality_colors = [Colors.SUCCESS, Colors.WARNING, Colors.ERROR]

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
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=quality_texts,
                textposition="outside",
                textfont={"size": Style.ANNOTATION_SIZE, "color": Colors.TEXT_SECONDARY},
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Panel 2: Image types in chunks
        sorted_types = sorted(
            chunks.image_types_in_chunks.items(), key=lambda x: x[1], reverse=True
        )
        type_names = [t[0] for t in sorted_types]
        type_counts = [t[1] for t in sorted_types]
        type_colors = Colors.categorical(len(type_names))

        fig.add_trace(
            go.Bar(
                y=type_names,
                x=type_counts,
                orientation="h",
                marker_color=type_colors,
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{c:,}" for c in type_counts],
                textposition="outside",
                textfont={"size": Style.ANNOTATION_SIZE, "color": Colors.TEXT_SECONDARY},
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Panel 3: Filtering impact
        filter_categories = ["Chunks", "Multimodal", "Images"]
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
                marker_color=Colors.LIGHT,
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{v:,}" for v in before_values],
                textposition="outside",
                textfont={"size": Style.ANNOTATION_SIZE, "color": Colors.TEXT_SECONDARY},
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Bar(
                name="After",
                x=filter_categories,
                y=after_values,
                marker_color=Colors.SUCCESS,
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{v:,}" for v in after_values],
                textposition="outside",
                textfont={"size": Style.ANNOTATION_SIZE, "color": Colors.TEXT_SECONDARY},
            ),
            row=1,
            col=3,
        )

        removed_values = [b - a for b, a in zip(before_values, after_values)]
        removed_text = f"Removed: {removed_values[0]:,} chunks"

        fig.update_layout(
            barmode="group",
            title={
                "text": f"Chunk Quality & Filtering ({removed_text})",
                "font": {"size": Style.TITLE_SIZE, "color": Colors.TEXT_PRIMARY},
                "x": 0.5,
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.06,
                "xanchor": "center",
                "x": 0.85,
                "font": {"size": Style.TICK_SIZE, "color": Colors.TEXT_SECONDARY},
            },
            plot_bgcolor=Colors.WHITE,
            paper_bgcolor=Colors.WHITE,
            font={"family": Style.FONT_FAMILY, "size": Style.TICK_SIZE},
            height=450,
            width=self.width + 300,
            margin={"l": 90, "r": 100, "t": 90, "b": 60},
        )

        fig.update_xaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Quality", autorange="reversed", row=1, col=1)
        fig.update_xaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Type", autorange="reversed", row=1, col=2)
        fig.update_xaxes(title_text="Category", row=1, col=3)
        fig.update_yaxes(title_text="Count", row=1, col=3)

        output_path = self.output_dir / "chunk_quality.png"
        fig.write_image(str(output_path), scale=2)
        return output_path
