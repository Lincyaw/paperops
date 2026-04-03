"""Composite components — convenience wrappers that expand into layout trees."""

from __future__ import annotations

from dataclasses import dataclass, field

from paperops.slides.core.constants import Direction
from paperops.slides.layout.containers import LayoutNode, HStack, VStack
from paperops.slides.components.shapes import Box, RoundedBox, Arrow
from paperops.slides.components.text import TextBlock


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

        boxes: list[RoundedBox] = []
        for label, color in zip(self.labels, box_colors):
            boxes.append(
                RoundedBox(
                    text=label,
                    color=color,
                    font_size="caption",
                    height=0.82,
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
        )
