"""Table component."""

from __future__ import annotations

from dataclasses import dataclass, field

from paperops.slides.layout.containers import LayoutNode


@dataclass
class Table(LayoutNode):
    """Data table."""

    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    header_color: str = "primary"
    header_text_color: str = "white"
    font_size: str | float = "caption"
    row_height: float = 0.45

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        w = self.width if self.width is not None else available_width
        num_rows = len(self.rows) + (1 if self.headers else 0)
        h = self.height if self.height is not None else num_rows * self.row_height
        return (w, h)
