"""DSL entrypoints for slide IR parsing and construction."""

from paperops.slides.dsl.json_loader import Document, DEFAULT_SCHEMA, load_json_document
from paperops.slides.dsl.python_builder import (
    Deck,
    Flex,
    Grid,
    Heading,
    KPI,
    Layer,
    Padding,
    Slide,
    Stack,
    Subtitle,
    Text,
    Title,
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
    "KPI",
    "load_json_document",
    "Layer",
    "Padding",
    "Slide",
    "Stack",
    "Subtitle",
    "Text",
    "Title",
    "Box",
    "render_json",
]
