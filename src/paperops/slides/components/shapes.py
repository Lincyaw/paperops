"""Basic shape components — Box, RoundedBox, Circle, Badge, Arrow, Line."""

from __future__ import annotations

from dataclasses import dataclass

from paperops.slides.core.constants import Align, Direction
from paperops.slides.layout.auto_size import TextStyle, build_intrinsic_size, measure_text_intrinsic
from paperops.slides.layout.containers import LayoutNode
from paperops.slides.layout.types import Constraints, IntrinsicSize


def _resolve_font_pt(theme, size: str | float) -> float:
    if theme is not None:
        return theme.resolve_font_size(size)
    return float(size) if isinstance(size, (int, float)) else 18.0


def _clamp_intrinsic(node: LayoutNode, intrinsic: IntrinsicSize, constraints: Constraints) -> IntrinsicSize:
    adjusted = intrinsic
    if node.width is not None:
        adjusted = IntrinsicSize(adjusted.min_width, node.width, adjusted.min_height, adjusted.preferred_height)
    if node.height is not None:
        adjusted = IntrinsicSize(adjusted.min_width, adjusted.preferred_width, adjusted.min_height, node.height)
    if node.min_width is not None:
        adjusted = IntrinsicSize(max(adjusted.min_width, node.min_width), adjusted.preferred_width, adjusted.min_height, adjusted.preferred_height)
    if node.min_height is not None:
        adjusted = IntrinsicSize(adjusted.min_width, adjusted.preferred_width, max(adjusted.min_height, node.min_height), adjusted.preferred_height)
    if node.height is not None:
        adjusted = IntrinsicSize(adjusted.min_width, adjusted.preferred_width, node.height, node.height)
    return adjusted.clamp(constraints)


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
    padding_x: float = 0.22
    padding_y: float = 0.14

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        pt = _resolve_font_pt(theme, self.font_size)
        ff = getattr(theme, "font_family", "Calibri") if theme else "Calibri"
        style = TextStyle(
            font_family=ff,
            font_size_pt=pt,
            bold=self.bold,
            margin_x=self.padding_x * 2,
            margin_y=self.padding_y * 2,
        )
        text_intrinsic = build_intrinsic_size(measure_text_intrinsic(self.text, style, max_width_inches=self.width or constraints.max_width))
        if not self.text:
            text_intrinsic = IntrinsicSize(0.5, 0.5, 0.35, 0.35)
        return _clamp_intrinsic(self, text_intrinsic, constraints)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        intrinsic = self.measure(Constraints(max_width=max(available_width, 0.0)), theme)
        return intrinsic.preferred_width, intrinsic.preferred_height


@dataclass
class RoundedBox(Box):
    radius: float = 0.08


@dataclass
class Circle(LayoutNode):
    """Circle with centered text."""

    text: str = ""
    color: str = "primary"
    text_color: str = "white"
    font_size: str | float = "body"
    bold: bool = True
    radius: float | None = None
    padding: float = 0.18

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        if self.radius is not None:
            diameter = self.radius * 2
            return IntrinsicSize(diameter, diameter, diameter, diameter).clamp(constraints)

        pt = _resolve_font_pt(theme, self.font_size)
        ff = getattr(theme, "font_family", "Calibri") if theme else "Calibri"
        style = TextStyle(font_family=ff, font_size_pt=pt, bold=self.bold)
        text_intrinsic = build_intrinsic_size(measure_text_intrinsic(self.text, style, max_width_inches=self.width or constraints.max_width))
        diameter = max(text_intrinsic.preferred_width, text_intrinsic.preferred_height) + self.padding * 2
        intrinsic = IntrinsicSize(diameter, diameter, diameter, diameter)
        return _clamp_intrinsic(self, intrinsic, constraints)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        intrinsic = self.measure(Constraints(max_width=max(available_width, 0.0)), theme)
        return intrinsic.preferred_width, intrinsic.preferred_height


@dataclass
class Badge(LayoutNode):
    """Compact label."""

    text: str = ""
    color: str = "primary"
    text_color: str = "white"
    font_size: str | float = "caption"
    bold: bool = True
    padding_x: float = 0.18
    padding_y: float = 0.08

    def __post_init__(self):
        if self.size_mode_x == "auto":
            self.size_mode_x = "fit"

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        pt = _resolve_font_pt(theme, self.font_size)
        ff = getattr(theme, "font_family", "Calibri") if theme else "Calibri"
        style = TextStyle(
            font_family=ff,
            font_size_pt=pt,
            bold=self.bold,
            margin_x=self.padding_x * 2,
            margin_y=self.padding_y * 2,
        )
        intrinsic = build_intrinsic_size(measure_text_intrinsic(self.text, style, max_width_inches=self.width or constraints.max_width))
        return _clamp_intrinsic(self, intrinsic, constraints)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        intrinsic = self.measure(Constraints(max_width=max(available_width, 0.0)), theme)
        return intrinsic.preferred_width, intrinsic.preferred_height


@dataclass
class Arrow(LayoutNode):
    """Connector placeholder resolved after layout."""

    from_component: LayoutNode | None = None
    to_component: LayoutNode | None = None
    label: str | None = None
    color: str = "primary"
    width_pt: float = 1.5
    direction: Direction | str = "horizontal"

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        return IntrinsicSize(0.0, 0.0, 0.0, 0.0)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        return (0.0, 0.0)


@dataclass
class Line(LayoutNode):
    """Line placeholder resolved after layout."""

    from_component: LayoutNode | None = None
    to_component: LayoutNode | None = None
    color: str = "border"
    width_pt: float = 1.0
    dashed: bool = False

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        return IntrinsicSize(0.0, 0.0, 0.0, 0.0)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        return (0.0, 0.0)
