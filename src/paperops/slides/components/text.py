"""Text components — TextBlock, BulletList."""

from __future__ import annotations

from dataclasses import dataclass, field

from paperops.slides.components.shapes import _resolve_font_pt
from paperops.slides.core.constants import Align
from paperops.slides.layout.auto_size import TextStyle, build_intrinsic_size, measure_text_intrinsic
from paperops.slides.layout.containers import LayoutNode
from paperops.slides.layout.types import Constraints, IntrinsicSize


@dataclass
class TextBlock(LayoutNode):
    """Block of wrapped text."""

    text: str = ""
    runs: list | None = None
    font_size: str | float = "body"
    color: str = "text"
    align: Align | str = "left"
    bold: bool = False
    italic: bool = False
    line_spacing: float = 1.25
    margin_x: float = 0.10
    margin_y: float = 0.14

    def _content_text(self) -> str:
        if self.runs is not None:
            return "".join(run_text for run_text, _fmt in self.runs)
        return self.text

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        if self.width is not None and self.height is not None:
            return IntrinsicSize(self.width, self.width, self.height, self.height).clamp(constraints)

        pt = _resolve_font_pt(theme, self.font_size)
        font_family = getattr(theme, "font_family", "Calibri") if theme else "Calibri"
        content = self._content_text()
        if not content:
            height = self.height if self.height is not None else (pt / 72.0) * self.line_spacing + self.margin_y
            width = self.width if self.width is not None else (constraints.max_width or 0.0)
            return IntrinsicSize(0.0, width, height, height).clamp(constraints)

        wrap_width = self.width if self.width is not None else constraints.max_width
        style = TextStyle(
            font_family=font_family,
            font_size_pt=pt,
            bold=self.bold,
            italic=self.italic,
            line_spacing=self.line_spacing,
            margin_x=self.margin_x,
            margin_y=self.margin_y,
        )
        intrinsic = build_intrinsic_size(measure_text_intrinsic(content, style, max_width_inches=wrap_width))
        if self.width is not None:
            intrinsic = IntrinsicSize(intrinsic.min_width, self.width, intrinsic.min_height, intrinsic.preferred_height)
        if self.height is not None:
            intrinsic = IntrinsicSize(intrinsic.min_width, intrinsic.preferred_width, self.height, self.height)
        if self.min_width is not None:
            intrinsic = IntrinsicSize(max(intrinsic.min_width, self.min_width), intrinsic.preferred_width, intrinsic.min_height, intrinsic.preferred_height)
        if self.min_height is not None:
            intrinsic = IntrinsicSize(intrinsic.min_width, intrinsic.preferred_width, max(intrinsic.min_height, self.min_height), intrinsic.preferred_height)
        return intrinsic.clamp(constraints)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        intrinsic = self.measure(Constraints(max_width=max(available_width, 0.0)), theme)
        return intrinsic.preferred_width, intrinsic.preferred_height


@dataclass
class BulletList(LayoutNode):
    """List of bullet points with indent-aware measurement."""

    items: list = field(default_factory=list)
    font_size: str | float = "body"
    color: str = "text"
    line_spacing: float = 1.25
    margin_x: float = 0.20
    margin_y: float = 0.12
    indent_step: float = 0.24

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        if self.width is not None and self.height is not None:
            return IntrinsicSize(self.width, self.width, self.height, self.height).clamp(constraints)

        pt = _resolve_font_pt(theme, self.font_size)
        font_family = getattr(theme, "font_family", "Calibri") if theme else "Calibri"
        wrap_width = self.width if self.width is not None else constraints.max_width
        base_width = wrap_width if wrap_width is not None else 6.0

        min_width = 0.0
        preferred_height = 0.0
        for item in self.items:
            if isinstance(item, tuple):
                text, indent = item
            else:
                text = item
                indent = 0
            indent_width = max(indent, 0) * self.indent_step
            usable_width = max((wrap_width if wrap_width is not None else base_width) - indent_width - self.margin_x, 0.4)
            style = TextStyle(
                font_family=font_family,
                font_size_pt=pt,
                line_spacing=self.line_spacing,
                margin_x=self.margin_x,
                margin_y=0.0,
            )
            item_intrinsic = measure_text_intrinsic(text, style, max_width_inches=usable_width)
            min_width = max(min_width, item_intrinsic.min_width + indent_width)
            preferred_height += item_intrinsic.preferred_height

        preferred_height += self.margin_y
        width = self.width if self.width is not None else (wrap_width if wrap_width is not None else min_width)
        intrinsic = IntrinsicSize(
            min_width=max(self.min_width or 0.0, min_width),
            preferred_width=max(width or min_width, min_width),
            min_height=self.height if self.height is not None else min((pt / 72.0) * self.line_spacing + self.margin_y, preferred_height),
            preferred_height=self.height if self.height is not None else preferred_height,
        )
        return intrinsic.clamp(constraints)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        intrinsic = self.measure(Constraints(max_width=max(available_width, 0.0)), theme)
        return intrinsic.preferred_width, intrinsic.preferred_height
