"""SlideCraft — AI-oriented PPT generation toolkit."""

try:
    from paperops.slides.build import Presentation
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extras
    if exc.name != "pptx":
        raise
    _presentation_import_error = exc

    class Presentation:  # type: ignore[override]
        def __init__(self, *_args, **_kwargs):
            raise ModuleNotFoundError(
                "Presentation requires the optional 'slides' extras; install python-pptx to use it."
            ) from _presentation_import_error
from paperops.slides.core.constants import Align, Direction
from paperops.slides.core.theme import Theme, themes
from paperops.slides.layout import Column, Constraints, Flex, Grid, GridItem, HStack, IntrinsicSize, LayoutIssue, Padding, Row, VStack, auto, fixed, fr
from paperops.slides.components.shapes import Arrow, Badge, Box, Circle, Line, RoundedBox
from paperops.slides.components.text import BulletList, TextBlock
from paperops.slides.components.table import Table
from paperops.slides.components.image import Image, SvgImage
from paperops.slides.components.composite import Callout, Flow
from paperops.slides.components.charts import (
    BarChart,
    Flowchart,
    HorizontalBarChart,
    LineChart,
    PieChart,
    RadarChart,
)
from paperops.slides.components.svg_canvas import SvgCanvas

__all__ = [
    "Align",
    "Arrow",
    "Badge",
    "BarChart",
    "Box",
    "BulletList",
    "Callout",
    "Circle",
    "Column",
    "Constraints",
    "Direction",
    "Flex",
    "Flow",
    "Flowchart",
    "Grid",
    "GridItem",
    "HStack",
    "HorizontalBarChart",
    "Image",
    "IntrinsicSize",
    "LayoutIssue",
    "Line",
    "LineChart",
    "Padding",
    "PieChart",
    "Presentation",
    "RadarChart",
    "RoundedBox",
    "Row",
    "SvgCanvas",
    "SvgImage",
    "Table",
    "TextBlock",
    "Theme",
    "VStack",
    "auto",
    "fixed",
    "fr",
    "themes",
]
