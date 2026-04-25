"""DSL entrypoints for slide IR parsing and construction."""

from paperops.slides.dsl.json_loader import Document, DEFAULT_SCHEMA, load_json_document
from paperops.slides.dsl.markdown_parser import (
    MarkdownParseError,
    load_markdown_document,
    parse_markdown_fragment,
    parse_markdown_text,
)
from paperops.slides.dsl.mdx_parser import (
    MDXParseError,
    load_mdx_document,
    parse_mdx_fragment,
    parse_mdx_text,
)
from paperops.slides.dsl.python_builder import (
    Deck,
    Flex,
    Grid,
    Heading,
    HStack,
    KPI,
    Layer,
    Padding,
    Slide,
    Stack,
    Subtitle,
    Text,
    Title,
    VStack,
    Box,
    render_json,
)

__all__ = [
    "DEFAULT_SCHEMA",
    "Deck",
    "Document",
    "Grid",
    "Flex",
    "Heading",
    "HStack",
    "KPI",
    "load_json_document",
    "load_markdown_document",
    "load_mdx_document",
    "Layer",
    "Padding",
    "Slide",
    "MarkdownParseError",
    "MDXParseError",
    "Stack",
    "parse_markdown_fragment",
    "parse_markdown_text",
    "parse_mdx_fragment",
    "parse_mdx_text",
    "Subtitle",
    "Text",
    "Title",
    "VStack",
    "Box",
    "render_json",
]
