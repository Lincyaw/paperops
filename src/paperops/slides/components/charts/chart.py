"""Chart component declarations."""

from __future__ import annotations

from dataclasses import dataclass, field

from paperops.slides.components.registry import register_component
from paperops.slides.layout.containers import LayoutNode
from paperops.slides.layout.types import Constraints, IntrinsicSize


@register_component(
    "chart",
    props_schema={
        "properties": {
            "chart_type": {"type": "string", "enum": ["line", "bar", "pie", "area"], "required": True},
            "data": {"type": "object"},
            "title": {"type": "string"},
            "labels": {"type": "array"},
            "series": {"type": "array"},
            "height": {"type": ["number", "string"]},
            "width": {"type": ["number", "string"]},
        },
        "required": ["chart_type"],
    },
    default_classes=["chart"],
)
class _ChartDefinition:
    pass


@dataclass
class Chart(LayoutNode):
    """Minimal chart layout placeholder."""

    chart_type: str = "line"
    data: dict[str, object] = field(default_factory=dict)

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        width = self.width if self.width is not None else (constraints.max_width or 4.0)
        height = self.height if self.height is not None else (width * 0.62)
        return IntrinsicSize(
            min_width=self.min_width or min(width, 1.0),
            preferred_width=width,
            min_height=self.min_height or min(height, 0.5),
            preferred_height=height,
        ).clamp(constraints)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        intrinsic = self.measure(Constraints(max_width=max(available_width, 0.0)), theme)
        return intrinsic.preferred_width, intrinsic.preferred_height
