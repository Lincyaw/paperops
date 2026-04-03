"""Text components — TextBlock, BulletList."""

from __future__ import annotations

from dataclasses import dataclass, field

from paperops.slides.core.constants import Align
from paperops.slides.layout.containers import LayoutNode
from paperops.slides.components.shapes import _resolve_font_pt
from paperops.slides.layout.auto_size import (
    estimate_min_text_width,
    measure_text,
    measure_wrapped_text_height,
)


@dataclass
class TextBlock(LayoutNode):
    """Block of text. Wraps within allocated width.

    For rich text, set ``runs`` to a list of (text, format_dict) tuples.
    format_dict keys: bold, italic, color, font_size.  When ``runs`` is
    set, the plain ``text`` field is ignored.
    """

    text: str = ""
    runs: list | None = None  # list of (text, format_dict) tuples
    font_size: str | float = "body"
    color: str = "text"
    align: Align | str = "left"
    bold: bool = False
    italic: bool = False

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        if self.width is not None and self.height is not None:
            return (self.width, self.height)

        pt = _resolve_font_pt(theme, self.font_size)
        font_family = getattr(theme, "font_family", "Calibri") if theme else "Calibri"

        if self.runs is not None:
            text = "".join(run_text for run_text, _fmt in self.runs)
        else:
            text = self.text

        if not text:
            return (self.width or available_width, self.height or pt * 0.025 + 0.1)

        wrap_width = self.width if self.width is not None else available_width
        # Match the default left/right margins applied by PowerPoint text boxes.
        usable_width = max(wrap_width - 0.10, 0.10)
        min_text_width = estimate_min_text_width(text, font_family, pt)
        if self.min_width is None:
            self.min_width = max(min_text_width + 0.10, 0.35)
        _text_w, text_h = measure_text(text, font_family, pt, max_width_inches=usable_width)
        h = text_h + 0.14

        w = self.width if self.width is not None else available_width
        h = self.height if self.height is not None else h
        return (w, h)


@dataclass
class BulletList(LayoutNode):
    """List of bullet points.

    Items can be plain strings or (text, indent_level) tuples.
    """

    items: list = field(default_factory=list)
    font_size: str | float = "body"
    color: str = "text"

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        if self.width is not None and self.height is not None:
            return (self.width, self.height)

        pt = _resolve_font_pt(theme, self.font_size)
        font_family = getattr(theme, "font_family", "Calibri") if theme else "Calibri"
        bullet_text = "\n".join(
            item if isinstance(item, str) else item[0]
            for item in self.items
        )
        usable_width = max((self.width or available_width) - 0.20, 0.10)
        h = measure_wrapped_text_height(
            bullet_text,
            font_family,
            pt,
            usable_width,
        ) + 0.12

        w = self.width if self.width is not None else available_width
        h = self.height if self.height is not None else h
        return (w, h)
