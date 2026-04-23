"""Core layer: theme, types, constants."""

from paperops.slides.core.theme import Theme, themes
from paperops.slides.core.tokens import (
    UnknownTokenCategoryError,
    UnknownTokenError,
    resolve,
    resolve_token,
)
from paperops.slides.core.types import resolve_color, resolve_font_size, resolve_size
from paperops.slides.core.constants import SLIDE_WIDTH, SLIDE_HEIGHT, CONTENT_REGION

__all__ = [
    "UnknownTokenCategoryError",
    "UnknownTokenError",
    "Theme",
    "themes",
    "CONTENT_REGION",
    "SLIDE_HEIGHT",
    "SLIDE_WIDTH",
    "resolve",
    "resolve_color",
    "resolve_font_size",
    "resolve_size",
    "resolve_token",
]
