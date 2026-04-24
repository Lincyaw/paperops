"""Core slide nodes: shapes, text, images, tables, and SVG."""

from paperops.slides.components import charts  # noqa: F401
from paperops.slides.components import semantic  # noqa: F401
from paperops.slides.components.shapes import Box, RoundedBox, Circle, Badge, Arrow, Line
from paperops.slides.components.text import TextBlock, BulletList
from paperops.slides.components.table import Table
from paperops.slides.components.image import Image, SvgImage
from paperops.slides.components.charts.chart import Chart

__all__ = [
    "Arrow",
    "Badge",
    "Box",
    "BulletList",
    "Chart",
    "Circle",
    "Image",
    "Line",
    "RoundedBox",
    "SvgImage",
    "Table",
    "TextBlock",
]
