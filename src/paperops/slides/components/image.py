"""Image components — static images and SVG."""

from __future__ import annotations

from dataclasses import dataclass

from paperops.slides.layout.containers import LayoutNode
from paperops.slides.layout.types import Constraints, IntrinsicSize


@dataclass
class Image(LayoutNode):
    """Static image from file."""

    path: str = ""
    preserve_aspect: bool = True
    aspect_ratio: float | None = None

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        width = self.width if self.width is not None else constraints.max_width or 4.0
        ratio = self.aspect_ratio or (4 / 3)
        if self.height is not None:
            height = self.height
            if self.preserve_aspect and self.width is None:
                width = height * ratio
        else:
            height = width / ratio if self.preserve_aspect else width * 0.75
        intrinsic = IntrinsicSize(
            min_width=self.min_width or min(width, 1.0),
            preferred_width=width,
            min_height=self.min_height or min(height, 0.75),
            preferred_height=height,
        )
        return intrinsic.clamp(constraints)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        intrinsic = self.measure(Constraints(max_width=max(available_width, 0.0)), theme)
        return intrinsic.preferred_width, intrinsic.preferred_height


@dataclass
class SvgImage(LayoutNode):
    """SVG rendered to PNG."""

    svg: str | object = ""
    scale: int = 3
    preserve_aspect: bool = True
    aspect_ratio: float | None = None

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        width = self.width if self.width is not None else constraints.max_width or 4.0
        ratio = self.aspect_ratio or (4 / 3)
        if self.height is not None:
            height = self.height
            if self.preserve_aspect and self.width is None:
                width = height * ratio
        else:
            height = width / ratio if self.preserve_aspect else width * 0.75
        intrinsic = IntrinsicSize(
            min_width=self.min_width or min(width, 1.0),
            preferred_width=width,
            min_height=self.min_height or min(height, 0.75),
            preferred_height=height,
        )
        return intrinsic.clamp(constraints)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        intrinsic = self.measure(Constraints(max_width=max(available_width, 0.0)), theme)
        return intrinsic.preferred_width, intrinsic.preferred_height
