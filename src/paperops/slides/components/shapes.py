"""Basic shape components — Box, RoundedBox, Circle, Badge, Arrow, Line."""

from __future__ import annotations

from dataclasses import dataclass, field

from paperops.slides.core.constants import Align, Direction
from paperops.slides.layout.containers import LayoutNode
from paperops.slides.layout.auto_size import measure_text


def _resolve_font_pt(theme, size: str | float) -> float:
    """Resolve a semantic or numeric font size to points."""
    if theme is not None:
        return theme.resolve_font_size(size)
    return float(size) if isinstance(size, (int, float)) else 18.0


def _estimate_text_size(
    text: str,
    font_size_pt: float,
    explicit_w: float | None,
    explicit_h: float | None,
    margin_x: float = 0.3,
    margin_y: float = 0.2,
    font_family: str = "Calibri",
) -> tuple[float, float]:
    """Estimate text bounding box in inches, delegating to measure_text.

    Adds *margin_x* / *margin_y* padding around the measured content.
    """
    if not text:
        w = explicit_w or 0.5
        h = explicit_h or 0.5
        return (w, h)

    content_w, content_h = measure_text(text, font_family, font_size_pt)

    w = explicit_w if explicit_w is not None else content_w + margin_x
    h = explicit_h if explicit_h is not None else content_h + margin_y
    return (w, h)


@dataclass
class Box(LayoutNode):
    """Rectangle with optional text."""

    text: str = ""
    color: str = "bg_alt"
    border: str = "border"
    text_color: str = "text"
    font_size: str | float = "body"
    bold: bool = False
    align: Align | str = "center"
    border_width: float = 1.0

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        pt = _resolve_font_pt(theme, self.font_size)
        ff = getattr(theme, 'font_family', 'Calibri') if theme else 'Calibri'
        return _estimate_text_size(self.text, pt, self.width, self.height,
                                   font_family=ff)


@dataclass
class RoundedBox(Box):
    """Rounded rectangle — same as Box but rendered with rounded corners."""

    radius: float = 0.08


@dataclass
class Circle(LayoutNode):
    """Circle with optional centered text."""

    text: str = ""
    color: str = "primary"
    text_color: str = "white"
    font_size: str | float = "body"
    bold: bool = True
    radius: float | None = None

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        if self.width is not None and self.height is not None:
            return (self.width, self.height)

        if self.radius is not None:
            d = self.radius * 2
            return (self.width or d, self.height or d)

        pt = _resolve_font_pt(theme, self.font_size)
        ff = getattr(theme, 'font_family', 'Calibri') if theme else 'Calibri'
        w, h = _estimate_text_size(self.text, pt, None, None, margin_x=0.2,
                                   margin_y=0.2, font_family=ff)
        d = max(w, h)
        return (self.width or d, self.height or d)


@dataclass
class Badge(LayoutNode):
    """Small colored label, auto-width."""

    text: str = ""
    color: str = "primary"
    text_color: str = "white"
    font_size: str | float = "caption"
    bold: bool = True

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        pt = _resolve_font_pt(theme, self.font_size)
        ff = getattr(theme, 'font_family', 'Calibri') if theme else 'Calibri'
        return _estimate_text_size(
            self.text, pt, self.width, self.height,
            margin_x=0.2, margin_y=0.16, font_family=ff,
        )


@dataclass
class Arrow(LayoutNode):
    """Arrow connecting two components. Post-layout: rendered after positioning."""

    from_component: LayoutNode | None = None
    to_component: LayoutNode | None = None
    label: str | None = None
    color: str = "primary"
    width_pt: float = 1.5
    direction: Direction | str = "horizontal"  # "horizontal" or "vertical"

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        return (0, 0)


@dataclass
class Line(LayoutNode):
    """Line connecting two components. Post-layout."""

    from_component: LayoutNode | None = None
    to_component: LayoutNode | None = None
    color: str = "border"
    width_pt: float = 1.0
    dashed: bool = False

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        return (0, 0)
