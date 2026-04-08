"""Image components — static images and SVG."""

from __future__ import annotations

from dataclasses import dataclass

from paperops.slides.layout.containers import LayoutNode


@dataclass
class Image(LayoutNode):
    """Static image from file."""

    path: str = ""
    preserve_aspect: bool = True

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        # Images use explicitly set dimensions or fall back to defaults
        w = self.width if self.width is not None else available_width
        h = self.height if self.height is not None else w * 0.75  # 4:3 default aspect
        return (w, h)


@dataclass
class SvgImage(LayoutNode):
    """SVG rendered to PNG."""

    svg: str | object = ""  # SVG string or SvgCanvas instance
    scale: int = 3

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        w = self.width if self.width is not None else available_width
        h = self.height if self.height is not None else w * 0.75
        return (w, h)
