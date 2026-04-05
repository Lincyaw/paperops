"""Composite components — convenience wrappers that expand into layout trees."""

from __future__ import annotations

from dataclasses import dataclass, field

from paperops.slides.core.constants import Direction
from paperops.slides.layout.containers import LayoutNode, HStack, VStack
from paperops.slides.layout.auto_size import measure_text
from paperops.slides.components.shapes import Box, RoundedBox, Arrow
from paperops.slides.components.text import TextBlock


def _resolve_font_pt(theme, size: str | float) -> float:
    """Resolve a semantic or numeric font size to points."""
    if theme is not None:
        return theme.resolve_font_size(size)
    return float(size) if isinstance(size, (int, float)) else 18.0


def _estimate_full_text_width(text: str, font_size_pt: float, font_family: str = "Calibri") -> float:
    """Estimate the full width needed for text, accounting for margins.

    Unlike estimate_min_text_width which only measures the longest token,
    this measures the entire text string.
    """
    if not text:
        return 0.5
    content_w, _ = measure_text(text, font_family, font_size_pt)
    # Add horizontal margin padding (0.7 inches total, matching RoundedBox)
    # Increased from 0.3 to prevent text wrapping issues
    return content_w + 0.7


@dataclass
class Callout(LayoutNode):
    """Callout box with colored left accent bar, title, and body text."""

    title: str = ""
    body: str = ""
    color: str = "primary"

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        layout = self.to_layout()
        return layout.preferred_size(theme, available_width)

    def to_layout(self) -> LayoutNode:
        """Expand into a layout tree of basic components."""
        accent_bar = Box(
            width=0.06,
            color=self.color,
            border=self.color,
            size_mode_y="stretch",
        )

        children: list[LayoutNode] = []
        if self.title:
            children.append(TextBlock(
                text=self.title,
                bold=True,
                color=self.color,
                font_size="body",
            ))
        if self.body:
            children.append(TextBlock(
                text=self.body,
                font_size="caption",
                color="text",
            ))

        text_stack = VStack(gap=0.1, children=children)

        return HStack(
            gap=0.15,
            children=[accent_bar, text_stack],
            width=self.width,
            height=self.height,
            size_mode_x=self.size_mode_x,
            size_mode_y=self.size_mode_y,
            grow=self.grow,
            shrink=self.shrink,
            basis=self.basis,
            wrap=self.wrap,
        )


@dataclass
class Flow(LayoutNode):
    """Connected flow of boxes with arrows between them."""

    labels: list[str] = field(default_factory=list)
    direction: Direction | str = "horizontal"
    colors: list[str] | None = None
    arrow_color: str = "primary"

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        layout = self.to_layout()
        return layout.preferred_size(theme, available_width)

    def to_layout(self) -> LayoutNode:
        """Expand into HStack/VStack of RoundedBox + Arrow."""
        if not self.labels:
            return HStack(children=[])

        box_colors = self.colors or ["bg_alt"] * len(self.labels)
        # Extend colors list if shorter than labels
        while len(box_colors) < len(self.labels):
            box_colors.append("bg_alt")

        # Pre-calculate text widths to ensure boxes fit content
        # Use a reasonable estimate for caption font size (10pt)
        caption_pt = 10.0
        font_family = "Liberation Sans"

        boxes: list[RoundedBox] = []
        for label, color in zip(self.labels, box_colors):
            # Let RoundedBox calculate its own preferred size
            # The preferred_size method now includes safety margins
            boxes.append(
                RoundedBox(
                    text=label,
                    color=color,
                    font_size="caption",
                    height=0.82,
                    size_mode_x="fit",
                )
            )

        children: list[LayoutNode] = []
        for i, box in enumerate(boxes):
            children.append(box)
            if i < len(boxes) - 1:
                children.append(Arrow(
                    from_component=boxes[i],
                    to_component=boxes[i + 1],
                    color=self.arrow_color,
                    width=0.22,
                ))

        Container = HStack if self.direction == "horizontal" else VStack
        return Container(
            gap=0.15,
            children=children,
            width=self.width,
            height=self.height,
            size_mode_x=self.size_mode_x,
            size_mode_y=self.size_mode_y,
            grow=self.grow,
            shrink=self.shrink,
            basis=self.basis,
            wrap=self.wrap,
        )
