"""Professional dataset visualization with Plotly for scientific publications.

Uses academic color palette optimized for journals and accessibility.
"""

import base64
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from synthetic_data.dataset.analyzer import DatasetStatistics
from synthetic_data.models import Sample
from synthetic_data.utils.colors import AcademicPalette as Colors, PlotStyle as Style


class DatasetPlotter:
    """Generate publication-quality visualizations for dataset analysis."""

    SPLIT_COLORS = {
        "train": Colors.PRIMARY,
        "validation": Colors.SECONDARY,
        "test": Colors.TERTIARY,
    }

    TYPE_COLORS = {
        "function_completion": Colors.PRIMARY,
        "code_generation": Colors.ACCENT_1,
        "qa": Colors.ACCENT_2,
    }

    MODALITY_COLORS = {
        "multimodal": Colors.PRIMARY,
        "text_only": Colors.LIGHT,
    }

    def __init__(
        self,
        statistics: DatasetStatistics,
        output_dir: Optional[Path] = None,
        width: int = 900,
        height: int = 500,
    ):
        """Initialize plotter.

        Args:
            statistics: Dataset statistics to visualize
            output_dir: Directory to save plots
            width: Default plot width
            height: Default plot height
        """
        self.statistics = statistics
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.width = width
        self.height = height
        self._samples: Optional[dict[str, list[Sample]]] = None

    def set_samples(self, samples: dict[str, list[Sample]]) -> None:
        """Set samples for showcase plot."""
        self._samples = samples

    def plot_all(self) -> list[Path]:
        """Generate all standard plots."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plots = [
            self.plot_split_distribution(),
            self.plot_type_distribution(),
            self.plot_category_distribution(),
            self.plot_type_category_by_split(),
            self.plot_modality_distribution(),
            self.plot_multimodal_breakdown(),
            self.plot_overview_dashboard(),
        ]

        if self._samples:
            plots.append(self.plot_sample_showcase())

        return [p for p in plots if p is not None]

    def _base_layout(
        self,
        title: str,
        xaxis_title: str = "",
        yaxis_title: str = "",
        showlegend: bool = True,
        height: Optional[int] = None,
        width: Optional[int] = None,
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

    def plot_split_distribution(self) -> Path:
        """Plot sample distribution across splits."""
        splits = list(self.statistics.splits.keys())
        counts = [self.statistics.splits[s].total for s in splits]
        colors = [self.SPLIT_COLORS.get(s, Colors.LIGHT) for s in splits]

        total = self.statistics.total_samples
        texts = [f"{c:,} ({c/total*100:.1f}%)" for c in counts]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=splits,
                    y=counts,
                    marker_color=colors,
                    marker_line_color=Colors.WHITE,
                    marker_line_width=0.5,
                    text=texts,
                    textposition="outside",
                    textfont={"size": Style.ANNOTATION_SIZE, "color": Colors.TEXT_SECONDARY},
                )
            ]
        )

        fig.update_layout(
            **self._base_layout(
                f"Dataset Split Distribution (n={total:,})",
                xaxis_title="Split",
                yaxis_title="Samples",
                showlegend=False,
            )
        )

        output_path = self.output_dir / "split_distribution.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_type_distribution(self) -> Path:
        """Plot question type distribution."""
        aggregated = self.statistics.to_dict()["aggregated"]["by_type"]
        types = list(aggregated.keys())
        counts = list(aggregated.values())
        colors = [self.TYPE_COLORS.get(t, Colors.LIGHT) for t in types]

        total = sum(counts)
        texts = [f"{c:,} ({c/total*100:.1f}%)" for c in counts]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=types,
                    y=counts,
                    marker_color=colors,
                    marker_line_color=Colors.WHITE,
                    marker_line_width=0.5,
                    text=texts,
                    textposition="outside",
                    textfont={"size": Style.ANNOTATION_SIZE, "color": Colors.TEXT_SECONDARY},
                )
            ]
        )

        fig.update_layout(
            **self._base_layout(
                "Question Type Distribution",
                xaxis_title="Type",
                yaxis_title="Samples",
                showlegend=False,
            )
        )

        output_path = self.output_dir / "type_distribution.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_category_distribution(self) -> Path:
        """Plot category distribution as horizontal bar chart."""
        aggregated = self.statistics.to_dict()["aggregated"]["by_category"]
        sorted_cats = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
        categories = [c[0] for c in sorted_cats]
        counts = [c[1] for c in sorted_cats]

        total = sum(counts)
        cat_colors = Colors.categorical(len(categories))

        fig = go.Figure(
            data=[
                go.Bar(
                    y=categories,
                    x=counts,
                    orientation="h",
                    marker_color=cat_colors,
                    marker_line_color=Colors.WHITE,
                    marker_line_width=0.5,
                    text=[f"{c:,} ({c/total*100:.1f}%)" for c in counts],
                    textposition="outside",
                    textfont={"size": Style.ANNOTATION_SIZE, "color": Colors.TEXT_SECONDARY},
                )
            ]
        )

        fig.update_layout(
            **self._base_layout(
                "Category Distribution",
                xaxis_title="Samples",
                showlegend=False,
                height=max(400, len(categories) * 45),
                width=self.width + 100,
            )
        )
        fig.update_yaxes(autorange="reversed")

        output_path = self.output_dir / "category_distribution.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_type_category_by_split(self) -> Path:
        """Plot question types by split with categories stacked."""
        splits = list(self.statistics.splits.keys())
        types = self.statistics.all_types
        categories = self.statistics.all_categories

        fig = make_subplots(
            rows=1,
            cols=len(types),
            subplot_titles=[t.replace("_", " ").title() for t in types],
            shared_yaxes=True,
            horizontal_spacing=0.06,
        )

        cat_colors = Colors.categorical(len(categories))

        for col, qtype in enumerate(types, 1):
            for i, category in enumerate(categories):
                counts = []
                for split in splits:
                    split_stats = self.statistics.splits[split]
                    type_total = split_stats.by_type.get(qtype, 0)
                    cat_count = split_stats.by_category.get(category, 0)

                    if type_total > 0 and split_stats.total > 0:
                        cat_ratio = cat_count / split_stats.total
                        estimated = int(type_total * cat_ratio)
                    else:
                        estimated = 0

                    counts.append(estimated)

                fig.add_trace(
                    go.Bar(
                        name=category.replace("_", " ").title(),
                        x=splits,
                        y=counts,
                        marker_color=cat_colors[i],
                        marker_line_color=Colors.WHITE,
                        marker_line_width=0.3,
                        text=[f"{c}" if c > 30 else "" for c in counts],
                        textposition="inside",
                        textfont={"size": 8, "color": Colors.WHITE},
                        legendgroup=category,
                        showlegend=(col == 1),
                    ),
                    row=1,
                    col=col,
                )

        fig.update_layout(
            barmode="stack",
            title={
                "text": "Question Types by Split (Stacked by Category)",
                "font": {"size": Style.TITLE_SIZE, "color": Colors.TEXT_PRIMARY},
                "x": 0.5,
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.25,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 9, "color": Colors.TEXT_SECONDARY},
            },
            plot_bgcolor=Colors.WHITE,
            paper_bgcolor=Colors.WHITE,
            font={"family": Style.FONT_FAMILY, "size": Style.TICK_SIZE},
            height=500,
            width=self.width + 100,
            margin={"l": 60, "r": 40, "t": 70, "b": 130},
        )

        output_path = self.output_dir / "type_category_by_split.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_modality_distribution(self) -> Path:
        """Plot multimodal vs text-only distribution across splits."""
        splits = list(self.statistics.splits.keys())

        mm_counts = [self.statistics.splits[s].multimodal for s in splits]
        text_counts = [self.statistics.splits[s].text_only for s in splits]
        totals = [mm + txt for mm, txt in zip(mm_counts, text_counts)]

        mm_texts = [f"{c:,} ({c/t*100:.0f}%)" if t > 0 else "" for c, t in zip(mm_counts, totals)]
        text_texts = [
            f"{c:,} ({c/t*100:.0f}%)" if t > 0 else "" for c, t in zip(text_counts, totals)
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="Multimodal",
                x=splits,
                y=mm_counts,
                marker_color=self.MODALITY_COLORS["multimodal"],
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=mm_texts,
                textposition="inside",
                textfont={"size": Style.ANNOTATION_SIZE, "color": Colors.WHITE},
            )
        )

        fig.add_trace(
            go.Bar(
                name="Text-only",
                x=splits,
                y=text_counts,
                marker_color=self.MODALITY_COLORS["text_only"],
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=text_texts,
                textposition="inside",
                textfont={"size": Style.ANNOTATION_SIZE, "color": Colors.TEXT_PRIMARY},
            )
        )

        fig.update_layout(
            barmode="stack",
            **self._base_layout(
                "Modality Distribution by Split",
                xaxis_title="Split",
                yaxis_title="Samples",
            ),
        )

        output_path = self.output_dir / "modality_distribution.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_multimodal_breakdown(self) -> Path:
        """Plot multimodal distribution breakdown."""
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=["By Split", "By Question Type", "By Category (Top 7)"],
            horizontal_spacing=0.10,
        )

        splits = list(self.statistics.splits.keys())

        # By Split
        mm_counts = [self.statistics.splits[s].multimodal for s in splits]
        text_counts = [self.statistics.splits[s].text_only for s in splits]
        totals = [mm + txt for mm, txt in zip(mm_counts, text_counts)]
        mm_pcts = [mm / t * 100 if t > 0 else 0 for mm, t in zip(mm_counts, totals)]

        fig.add_trace(
            go.Bar(
                x=splits,
                y=mm_counts,
                name="Multimodal",
                marker_color=self.MODALITY_COLORS["multimodal"],
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{c:,} ({p:.0f}%)" for c, p in zip(mm_counts, mm_pcts)],
                textposition="inside",
                textfont={"size": 9, "color": Colors.WHITE},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=splits,
                y=text_counts,
                name="Text-only",
                marker_color=self.MODALITY_COLORS["text_only"],
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # By Type
        types = self.statistics.all_types
        type_mm = []
        type_text = []

        for qtype in types:
            mm_total = sum(
                s.multimodal_by_type.get(qtype, 0) for s in self.statistics.splits.values()
            )
            text_total = (
                sum(s.by_type.get(qtype, 0) for s in self.statistics.splits.values()) - mm_total
            )
            type_mm.append(mm_total)
            type_text.append(text_total)

        type_totals = [mm + txt for mm, txt in zip(type_mm, type_text)]
        type_pcts = [mm / t * 100 if t > 0 else 0 for mm, t in zip(type_mm, type_totals)]

        fig.add_trace(
            go.Bar(
                x=types,
                y=type_mm,
                marker_color=self.MODALITY_COLORS["multimodal"],
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{c:,} ({p:.0f}%)" for c, p in zip(type_mm, type_pcts)],
                textposition="inside",
                textfont={"size": 9, "color": Colors.WHITE},
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=types,
                y=type_text,
                marker_color=self.MODALITY_COLORS["text_only"],
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # By Category
        categories = self.statistics.all_categories[:7]
        cat_mm = []
        cat_text = []

        for category in categories:
            mm_total = sum(
                s.multimodal_by_category.get(category, 0) for s in self.statistics.splits.values()
            )
            text_total = (
                sum(s.by_category.get(category, 0) for s in self.statistics.splits.values())
                - mm_total
            )
            cat_mm.append(mm_total)
            cat_text.append(text_total)

        cat_totals = [mm + txt for mm, txt in zip(cat_mm, cat_text)]
        cat_pcts = [mm / t * 100 if t > 0 else 0 for mm, t in zip(cat_mm, cat_totals)]

        # Shorten category names for display
        short_cats = [c.replace("_and_", "/").replace("_", " ")[:15] for c in categories]

        fig.add_trace(
            go.Bar(
                x=short_cats,
                y=cat_mm,
                marker_color=self.MODALITY_COLORS["multimodal"],
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{p:.0f}%" for p in cat_pcts],
                textposition="inside",
                textfont={"size": 8, "color": Colors.WHITE},
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Bar(
                x=short_cats,
                y=cat_text,
                marker_color=self.MODALITY_COLORS["text_only"],
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        fig.update_layout(
            barmode="stack",
            title={
                "text": "Multimodal Distribution Breakdown",
                "font": {"size": Style.TITLE_SIZE, "color": Colors.TEXT_PRIMARY},
                "x": 0.5,
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.06,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": Style.TICK_SIZE, "color": Colors.TEXT_SECONDARY},
            },
            plot_bgcolor=Colors.WHITE,
            paper_bgcolor=Colors.WHITE,
            font={"family": Style.FONT_FAMILY, "size": Style.TICK_SIZE},
            height=450,
            width=self.width + 200,
            margin={"l": 60, "r": 40, "t": 80, "b": 70},
        )

        fig.update_xaxes(tickangle=25, row=1, col=3)

        output_path = self.output_dir / "multimodal_breakdown.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_overview_dashboard(self) -> Path:
        """Generate comprehensive dashboard with key metrics."""
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[
                "Split Distribution",
                "Question Types",
                "Modality by Split",
                "Categories by Split",
                "Test Coverage",
                "Top Categories",
            ],
            vertical_spacing=0.18,
            horizontal_spacing=0.10,
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "bar", "colspan": 2}, None, {"type": "bar"}],
            ],
        )

        splits = list(self.statistics.splits.keys())
        total = self.statistics.total_samples
        cat_colors = Colors.categorical(len(self.statistics.all_categories))

        # Split Distribution
        counts = [self.statistics.splits[s].total for s in splits]
        split_colors = [self.SPLIT_COLORS.get(s, Colors.LIGHT) for s in splits]
        texts = [f"{c:,} ({c/total*100:.0f}%)" for c in counts]

        fig.add_trace(
            go.Bar(
                x=splits,
                y=counts,
                marker_color=split_colors,
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=texts,
                textposition="outside",
                textfont={"size": 9, "color": Colors.TEXT_SECONDARY},
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Question Types
        aggregated = self.statistics.to_dict()["aggregated"]["by_type"]
        types = list(aggregated.keys())
        type_counts = list(aggregated.values())
        type_colors = [self.TYPE_COLORS.get(t, Colors.LIGHT) for t in types]
        type_texts = [f"{c:,} ({c/total*100:.0f}%)" for c in type_counts]

        fig.add_trace(
            go.Bar(
                x=[t.replace("_", " ")[:12] for t in types],
                y=type_counts,
                marker_color=type_colors,
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=type_texts,
                textposition="outside",
                textfont={"size": 9, "color": Colors.TEXT_SECONDARY},
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Modality Distribution
        mm_counts = [self.statistics.splits[s].multimodal for s in splits]
        text_counts = [self.statistics.splits[s].text_only for s in splits]

        fig.add_trace(
            go.Bar(
                x=splits,
                y=mm_counts,
                name="Multimodal",
                marker_color=self.MODALITY_COLORS["multimodal"],
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{c:,}" for c in mm_counts],
                textposition="inside",
                textfont={"color": Colors.WHITE, "size": 9},
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Bar(
                x=splits,
                y=text_counts,
                name="Text-only",
                marker_color=self.MODALITY_COLORS["text_only"],
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{c:,}" for c in text_counts],
                textposition="inside",
                textfont={"color": Colors.TEXT_PRIMARY, "size": 9},
            ),
            row=1,
            col=3,
        )

        # Categories by Split
        categories = self.statistics.all_categories
        for i, category in enumerate(categories):
            cat_counts = [self.statistics.splits[s].by_category.get(category, 0) for s in splits]
            short_name = category.replace("_and_", "/").replace("_", " ")[:18]
            fig.add_trace(
                go.Bar(
                    name=short_name,
                    x=splits,
                    y=cat_counts,
                    marker_color=cat_colors[i],
                    marker_line_color=Colors.WHITE,
                    marker_line_width=0.3,
                    text=[f"{c}" if c > 50 else "" for c in cat_counts],
                    textposition="inside",
                    textfont={"size": 7, "color": Colors.WHITE},
                    legendgroup=category,
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

        # Test Coverage
        coverages = [self.statistics.splits[s].test_coverage * 100 for s in splits]

        fig.add_trace(
            go.Bar(
                x=splits,
                y=coverages,
                marker_color=split_colors,
                marker_line_color=Colors.WHITE,
                marker_line_width=0.5,
                text=[f"{c:.0f}%" for c in coverages],
                textposition="outside",
                textfont={"size": 9, "color": Colors.TEXT_SECONDARY},
                showlegend=False,
            ),
            row=2,
            col=3,
        )

        fig.update_layout(
            barmode="stack",
            title={
                "text": f"Dataset Analysis Dashboard (n={total:,})",
                "font": {"size": 16, "color": Colors.TEXT_PRIMARY},
                "x": 0.5,
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.12,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 8, "color": Colors.TEXT_SECONDARY},
            },
            plot_bgcolor=Colors.WHITE,
            paper_bgcolor=Colors.WHITE,
            font={"family": Style.FONT_FAMILY, "size": Style.TICK_SIZE},
            height=750,
            width=1200,
            margin={"l": 60, "r": 40, "t": 70, "b": 100},
        )

        output_path = self.output_dir / "overview_dashboard.png"
        fig.write_image(str(output_path), scale=2)
        return output_path

    def plot_sample_showcase(self) -> Optional[Path]:
        """Generate a showcase of multimodal samples."""
        if not self._samples:
            return None

        samples_by_type: dict[str, Sample] = {}
        for split_samples in self._samples.values():
            for sample in split_samples:
                if sample.image_path and sample.question_type not in samples_by_type:
                    samples_by_type[sample.question_type] = sample
                if len(samples_by_type) >= 3:
                    break
            if len(samples_by_type) >= 3:
                break

        if not samples_by_type:
            return None

        types = list(samples_by_type.keys())
        n_types = len(types)

        fig = make_subplots(
            rows=n_types,
            cols=2,
            column_widths=[0.6, 0.4],
            row_heights=[1.0 / n_types] * n_types,
            subplot_titles=[f"{t.replace('_', ' ').title()}" for t in types for _ in range(2)],
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            specs=[[{"type": "table"}, {"type": "image"}] for _ in range(n_types)],
        )

        type_colors = [self.TYPE_COLORS.get(t, Colors.PRIMARY) for t in types]

        for row, qtype in enumerate(types, 1):
            sample = samples_by_type[qtype]

            question_preview = (
                sample.question[:500] + "..." if len(sample.question) > 500 else sample.question
            )

            header_values = ["Field", "Content"]
            cell_values = [
                ["Type", "Category", "Question", "Has Test"],
                [
                    qtype.replace("_", " ").title(),
                    (sample.category or "N/A").replace("_", " ").title(),
                    question_preview.replace("\n", "<br>"),
                    "✓" if sample.test_code else "✗",
                ],
            ]

            fig.add_trace(
                go.Table(
                    header={
                        "values": header_values,
                        "fill_color": type_colors[row - 1],
                        "font": {"color": Colors.WHITE, "size": 10},
                        "align": "left",
                    },
                    cells={
                        "values": cell_values,
                        "fill_color": [[Colors.BACKGROUND, Colors.WHITE] * len(cell_values[0])],
                        "font": {"size": 9, "color": Colors.TEXT_PRIMARY},
                        "align": "left",
                        "height": 28,
                    },
                ),
                row=row,
                col=1,
            )

            if sample.image_path:
                image_path = Path(sample.image_path)
                if image_path.exists():
                    try:
                        with open(image_path, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode()

                        suffix = image_path.suffix.lower()
                        mime_type = {
                            ".png": "png",
                            ".jpg": "jpeg",
                            ".jpeg": "jpeg",
                            ".gif": "gif",
                        }.get(suffix, "png")

                        fig.add_layout_image(
                            {
                                "source": f"data:image/{mime_type};base64,{img_data}",
                                "xref": f"x{row * 2} domain",
                                "yref": f"y{row * 2} domain",
                                "x": 0,
                                "y": 1,
                                "sizex": 1,
                                "sizey": 1,
                                "xanchor": "left",
                                "yanchor": "top",
                                "layer": "above",
                            }
                        )
                    except (OSError, IOError):
                        pass

        fig.update_layout(
            title={
                "text": "Multimodal Sample Showcase",
                "font": {"size": Style.TITLE_SIZE, "color": Colors.TEXT_PRIMARY},
                "x": 0.5,
            },
            paper_bgcolor=Colors.WHITE,
            font={"family": Style.FONT_FAMILY, "size": Style.TICK_SIZE},
            height=350 * n_types,
            width=1200,
            margin={"l": 40, "r": 40, "t": 70, "b": 40},
            showlegend=False,
        )

        output_path = self.output_dir / "sample_showcase.png"
        fig.write_image(str(output_path), scale=2)
        return output_path
