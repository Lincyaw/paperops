"""Layout containers — HStack, VStack, Grid, Padding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from paperops.slides.core.constants import Region


@dataclass
class LayoutNode:
    """Base for all things that participate in layout."""

    _region: Region | None = field(default=None, init=False, repr=False)

    # Override in subclasses or set per-instance (keyword-only to avoid
    # positional arg conflicts with subclass fields like Box.text)
    width: float | None = field(default=None, kw_only=True)
    height: float | None = field(default=None, kw_only=True)
    min_width: float | None = field(default=None, kw_only=True)
    min_height: float | None = field(default=None, kw_only=True)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        """Return (width, height) this node would like.  Override in subclasses."""
        return (self.width or 1.0, self.height or 1.0)


@dataclass
class HStack(LayoutNode):
    """Lay children out left-to-right."""

    gap: float = 0.3
    children: list = field(default_factory=list)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        if not self.children:
            return (self.width or 0.0, self.height or 0.0)

        total_gap = self.gap * (len(self.children) - 1)
        remaining = available_width - total_gap

        # Fixed-width children consume their share first
        fixed_total = 0.0
        flex_count = 0
        for child in self.children:
            if hasattr(child, "width") and child.width is not None:
                fixed_total += child.width
            else:
                flex_count += 1

        if flex_count > 0:
            per_flex = max((remaining - fixed_total) / flex_count, 0.0)
        else:
            per_flex = 0.0

        max_h = 0.0
        total_w = 0.0
        for child in self.children:
            if hasattr(child, "width") and child.width is not None:
                cw = child.width
            else:
                cw = per_flex
            pw, ph = child.preferred_size(theme, cw)
            total_w += pw
            max_h = max(max_h, ph)

        total_w += total_gap
        w = self.width if self.width is not None else total_w
        h = self.height if self.height is not None else max_h
        return (w, h)

    def __iter__(self):
        return iter(self.children)


@dataclass
class VStack(LayoutNode):
    """Lay children out top-to-bottom."""

    gap: float = 0.3
    children: list = field(default_factory=list)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        if not self.children:
            return (self.width or 0.0, self.height or 0.0)

        total_gap = self.gap * (len(self.children) - 1)
        max_w = 0.0
        total_h = 0.0

        for child in self.children:
            pw, ph = child.preferred_size(theme, available_width)
            max_w = max(max_w, pw)
            total_h += ph

        total_h += total_gap
        w = self.width if self.width is not None else max_w
        h = self.height if self.height is not None else total_h
        return (w, h)

    def __iter__(self):
        return iter(self.children)


@dataclass
class Grid(LayoutNode):
    """Arrange children in a grid with *cols* columns and auto-calculated rows."""

    cols: int = 2
    gap: float = 0.3
    center_last_row: bool = False
    children: list = field(default_factory=list)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        if not self.children:
            return (self.width or 0.0, self.height or 0.0)

        n = len(self.children)
        rows = (n + self.cols - 1) // self.cols

        col_gap_total = self.gap * (self.cols - 1)
        row_gap_total = self.gap * (rows - 1)
        cell_w = (available_width - col_gap_total) / self.cols

        max_cell_h = 0.0
        for child in self.children:
            _, ph = child.preferred_size(theme, cell_w)
            max_cell_h = max(max_cell_h, ph)

        w = self.width if self.width is not None else available_width
        h = self.height if self.height is not None else (max_cell_h * rows + row_gap_total)
        return (w, h)

    def __iter__(self):
        return iter(self.children)


@dataclass
class Padding(LayoutNode):
    """Wrap a single child with padding."""

    child: Any = None
    all: float | None = None
    left: float | None = None
    right: float | None = None
    top: float | None = None
    bottom: float | None = None

    @property
    def _left(self) -> float:
        return self.left if self.left is not None else (self.all or 0.0)

    @property
    def _right(self) -> float:
        return self.right if self.right is not None else (self.all or 0.0)

    @property
    def _top(self) -> float:
        return self.top if self.top is not None else (self.all or 0.0)

    @property
    def _bottom(self) -> float:
        return self.bottom if self.bottom is not None else (self.all or 0.0)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        pad_h = self._left + self._right
        pad_v = self._top + self._bottom
        inner_w = max(available_width - pad_h, 0.0)

        if self.child is not None:
            cw, ch = self.child.preferred_size(theme, inner_w)
        else:
            cw, ch = (0.0, 0.0)

        w = self.width if self.width is not None else (cw + pad_h)
        h = self.height if self.height is not None else (ch + pad_v)
        return (w, h)
