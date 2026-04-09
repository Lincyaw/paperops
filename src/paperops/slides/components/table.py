"""Table component."""

from __future__ import annotations

from dataclasses import dataclass, field

from paperops.slides.layout.containers import LayoutNode
from paperops.slides.layout.types import Constraints, IntrinsicSize


@dataclass
class Table(LayoutNode):
    """Data table."""

    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    header_color: str = "primary"
    header_text_color: str = "white"
    font_size: str | float = "caption"
    row_height: float = 0.45

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        width = self.width if self.width is not None else (constraints.max_width or 4.0)
        num_rows = len(self.rows) + (1 if self.headers else 0)
        height = self.height if self.height is not None else max(num_rows, 1) * self.row_height
        intrinsic = IntrinsicSize(
            min_width=self.min_width or min(width, 2.0),
            preferred_width=width,
            min_height=self.min_height or min(height, self.row_height),
            preferred_height=height,
        )
        return intrinsic.clamp(constraints)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        intrinsic = self.measure(Constraints(max_width=max(available_width, 0.0)), theme)
        return intrinsic.preferred_width, intrinsic.preferred_height
