"""Slide dimensions and layout constants."""

from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    """Flow / arrow direction."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    RIGHT = "right"
    DOWN = "down"


class Align(str, Enum):
    """Text / box alignment."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass(frozen=True)
class Region:
    """A rectangular region on a slide, in inches."""
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height


# 16:9 widescreen
SLIDE_WIDTH = 13.333
SLIDE_HEIGHT = 7.5

# Standard margins
MARGIN_LEFT = 0.8
MARGIN_RIGHT = 0.8
MARGIN_TOP = 0.5
MARGIN_BOTTOM = 0.4

# Title region
TITLE_REGION = Region(
    left=MARGIN_LEFT,
    top=0.4,
    width=SLIDE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT,
    height=0.85,
)

# Accent line position (below title)
ACCENT_LINE_TOP = 1.25

# Content region (below title + accent line)
CONTENT_REGION = Region(
    left=MARGIN_LEFT,
    top=1.5,
    width=SLIDE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT,
    height=SLIDE_HEIGHT - 1.5 - MARGIN_BOTTOM,
)

# Reference region (bottom-right)
REFERENCE_REGION = Region(
    left=9.0,
    top=7.0,
    width=4.0,
    height=0.4,
)
