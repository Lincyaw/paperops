"""Built-in style sheets."""

from __future__ import annotations

from paperops.slides.style.sheet_registry import register_sheet
from .default import SHEET as default
from .keynote import SHEET as keynote
from .minimal import SHEET as minimal
from .academic import SHEET as academic
from .seminar import SHEET as seminar
from .whitepaper import SHEET as whitepaper
from .pitch import SHEET as pitch


# Register all built-ins eagerly so they are available by `sheet` name.
register_sheet("default", default)
register_sheet("minimal", minimal)
register_sheet("academic", academic)
register_sheet("seminar", seminar)
register_sheet("keynote", keynote)
register_sheet("whitepaper", whitepaper)
register_sheet("pitch", pitch)

__all__ = [
    "default",
    "minimal",
    "academic",
    "seminar",
    "keynote",
    "whitepaper",
    "pitch",
]
