from dataclasses import dataclass


@dataclass(frozen=True)
class AcademicPalette:

    PRIMARY = "#2E5A87"  
    SECONDARY = "#5B8FB9"  
    TERTIARY = "#8BB8D9"  

    ACCENT_1 = "#D4785C"  
    ACCENT_2 = "#7BA376"  
    ACCENT_3 = "#9B7BB8"  

    DARK = "#2D3142"  
    MEDIUM = "#4F5D75"  
    LIGHT = "#8D99AE"  
    LIGHTER = "#BFC0C0"  
    BACKGROUND = "#FAFBFC"  
    WHITE = "#FFFFFF"

    SUCCESS = "#6AAB6A"  
    WARNING = "#E5BA6C"  
    ERROR = "#C75F5F"  
    TEXT_PRIMARY = "#2D3142"
    TEXT_SECONDARY = "#4F5D75"
    TEXT_MUTED = "#8D99AE"  

    @classmethod
    def categorical(cls, n: int = 8) -> list[str]:
        colors = [
            cls.PRIMARY,  
            cls.ACCENT_1,  
            cls.ACCENT_2,  
            cls.ACCENT_3,  
            cls.SECONDARY,  
            "#B8A07E",  
            cls.LIGHT,  
            "#6B8E8E",  
            "#A17A74",  
            "#7B9EA8",  
        ]
        if n <= len(colors):
            return colors[:n]
        return (colors * ((n // len(colors)) + 1))[:n]

    @classmethod
    def sequential(cls, n: int = 5) -> list[str]:
        blues = [
            "#E8F1F8",  
            "#B8D4E9",
            "#7FB5D9",
            "#4A90C2",
            "#2E5A87",  
        ]
        if n <= len(blues):
            step = len(blues) // n
            return [blues[i * step] for i in range(n)]
        return blues

    @classmethod
    def diverging(cls, n: int = 5) -> list[str]:
        return [
            cls.LIGHT,  
            cls.SUCCESS,  
        ][:n]


class PlotStyle:
    FONT_FAMILY = "Inter, Helvetica Neue, Arial, sans-serif"
    TITLE_SIZE = 14
    LABEL_SIZE = 11
    TICK_SIZE = 10
    ANNOTATION_SIZE = 9

    LINE_WIDTH = 1.5
    MARKER_SIZE = 8
    BAR_LINE_WIDTH = 0.5

    MARGIN = {"l": 70, "r": 50, "t": 60, "b": 60}

    @classmethod
    def base_layout(
        cls,
        title: str = "",
        xaxis_title: str = "",
        yaxis_title: str = "",
        showlegend: bool = True,
        height: int = 500,
        width: int = 900,
        **kwargs,
    ) -> dict:
        colors = AcademicPalette

        layout = {
            "title": {
                "text": title,
                "font": {
                    "family": cls.FONT_FAMILY,
                    "size": cls.TITLE_SIZE,
                    "color": colors.TEXT_PRIMARY,
                },
                "x": 0.5,
                "xanchor": "center",
            },
            "xaxis": {
                "title": {
                    "text": xaxis_title,
                    "font": {"size": cls.LABEL_SIZE, "color": colors.TEXT_SECONDARY},
                },
                "tickfont": {"size": cls.TICK_SIZE, "color": colors.TEXT_SECONDARY},
                "gridcolor": colors.LIGHTER,
                "gridwidth": 0.5,
                "linecolor": colors.LIGHTER,
                "linewidth": 1,
                "showgrid": True,
                "zeroline": False,
            },
            "yaxis": {
                "title": {
                    "text": yaxis_title,
                    "font": {"size": cls.LABEL_SIZE, "color": colors.TEXT_SECONDARY},
                },
                "tickfont": {"size": cls.TICK_SIZE, "color": colors.TEXT_SECONDARY},
                "gridcolor": colors.LIGHTER,
                "gridwidth": 0.5,
                "linecolor": colors.LIGHTER,
                "linewidth": 1,
                "showgrid": True,
                "zeroline": False,
            },
            "showlegend": showlegend,
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": cls.TICK_SIZE, "color": colors.TEXT_SECONDARY},
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": colors.LIGHTER,
                "borderwidth": 0.5,
            },
            "plot_bgcolor": colors.WHITE,
            "paper_bgcolor": colors.WHITE,
            "font": {
                "family": cls.FONT_FAMILY,
                "size": cls.TICK_SIZE,
                "color": colors.TEXT_PRIMARY,
            },
            "margin": cls.MARGIN,
            "height": height,
            "width": width,
        }

        layout.update(kwargs)
        return layout

    @classmethod
    def bar_style(cls, color: str, show_text: bool = True) -> dict:
        return {
            "marker_color": color,
            "marker_line_color": AcademicPalette.WHITE,
            "marker_line_width": cls.BAR_LINE_WIDTH,
            "textposition": "outside" if show_text else "none",
            "textfont": {
                "size": cls.ANNOTATION_SIZE,
                "color": AcademicPalette.TEXT_SECONDARY,
            },
        }

    @classmethod
    def line_style(cls, color: str) -> dict:
        return {
            "line": {"color": color, "width": cls.LINE_WIDTH},
            "marker": {"size": cls.MARKER_SIZE, "color": color},
        }


Colors = AcademicPalette
Style = PlotStyle

