"""SlideCraft — AI-oriented PPT generation toolkit."""

from paperops.slides.build import Presentation
from paperops.slides.core.constants import Direction, Align
from paperops.slides.core.theme import Theme, themes
from paperops.slides.layout.containers import HStack, VStack, Grid, Padding
from paperops.slides.components.shapes import Box, RoundedBox, Circle, Badge, Arrow, Line
from paperops.slides.components.text import TextBlock, BulletList
from paperops.slides.components.table import Table
from paperops.slides.components.image import Image, SvgImage
from paperops.slides.components.composite import Callout, Flow
from paperops.slides.components.charts import (
    BarChart, RadarChart, Flowchart, LineChart, PieChart, HorizontalBarChart,
)
from paperops.slides.components.svg_canvas import SvgCanvas

__all__ = [
    "Presentation", "Direction", "Align", "Theme", "themes",
    "HStack", "VStack", "Grid", "Padding",
    "Box", "RoundedBox", "Circle", "Badge", "Arrow", "Line",
    "TextBlock", "BulletList", "Table", "Image", "SvgImage",
    "Callout", "Flow",
    "BarChart", "RadarChart", "Flowchart",
    "LineChart", "PieChart", "HorizontalBarChart",
    "SvgCanvas",
]
