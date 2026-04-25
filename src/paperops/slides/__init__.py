"""SlideCraft public API."""

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
from paperops.slides.dsl import (
    Deck,
    Document,
    Flex as IRFlex,
    Grid as IRGrid,
    Heading as IRHeading,
    HStack as IRHStack,
    KPI as IRKPI,
    Layer as IRLayer,
    Padding as IRPadding,
    Slide as IRSlide,
    Stack as IRStack,
    Subtitle as IRSubtitle,
    Text as IRText,
    Title as IRTitle,
    VStack as IRVStack,
    load_json_document,
    load_markdown_document,
    load_mdx_document,
    parse_markdown_text,
    parse_mdx_text,
    render_json,
)
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
    "Deck",
    "Direction",
    "Document",
    "Flex",
    "Grid",
    "GridItem",
    "HStack",
    "IRFlex",
    "IRGrid",
    "IRHeading",
    "IRHStack",
    "IRKPI",
    "IRLayer",
    "IRPadding",
    "IRSlide",
    "IRStack",
    "IRSubtitle",
    "IRText",
    "IRTitle",
    "IRVStack",
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
    "load_json_document",
    "load_markdown_document",
    "load_mdx_document",
    "parse_markdown_text",
    "parse_mdx_text",
    "render_json",
    "themes",
]
