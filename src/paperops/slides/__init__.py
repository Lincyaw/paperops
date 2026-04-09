"""SlideCraft — AI-oriented PPT generation toolkit."""

try:
    from paperops.slides.build import Presentation
    from paperops.slides.slides.base import SlideBuilder as Slide
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extras
    if exc.name != "pptx":
        raise
    _presentation_import_error = exc

    class Presentation:  # type: ignore[override]
        def __init__(self, *_args, **_kwargs):
            raise ModuleNotFoundError(
                "Presentation requires the optional 'slides' extras; install python-pptx to use it."
            ) from _presentation_import_error

    class Slide:  # type: ignore[override]
        def __init__(self, *_args, **_kwargs):
            raise ModuleNotFoundError(
                "Slide requires the optional 'slides' extras; install python-pptx to use it."
            ) from _presentation_import_error
from paperops.slides.core.constants import Align, Direction
from paperops.slides.core.theme import Theme, themes
from paperops.slides.layout import Absolute, AbsoluteItem, Column, Constraints, Flex, Grid, GridItem, HStack, IntrinsicSize, Layer, LayoutIssue, Padding, Row, Spacer, VStack, auto, fixed, fr
from paperops.slides.components.shapes import Arrow, Badge, Box, Circle, Line, RoundedBox
from paperops.slides.components.text import BulletList, TextBlock
from paperops.slides.components.table import Table
from paperops.slides.components.image import Image, SvgImage

Text = TextBlock
Rect = Box
RoundedRect = RoundedBox
Svg = SvgImage

__all__ = [
    "Align",
    "Absolute",
    "AbsoluteItem",
    "Arrow",
    "Badge",
    "Box",
    "BulletList",
    "Circle",
    "Column",
    "Constraints",
    "Direction",
    "Flex",
    "Grid",
    "GridItem",
    "HStack",
    "Image",
    "IntrinsicSize",
    "Layer",
    "LayoutIssue",
    "Line",
    "Padding",
    "Presentation",
    "Rect",
    "RoundedBox",
    "RoundedRect",
    "Row",
    "Slide",
    "Spacer",
    "Svg",
    "SvgImage",
    "Table",
    "Text",
    "TextBlock",
    "Theme",
    "VStack",
    "auto",
    "fixed",
    "fr",
    "themes",
]
