"""Horizontal bar chart component — SVG."""

from __future__ import annotations

from dataclasses import dataclass, field

from paperops.slides.components.svg_canvas import SvgCanvas
from paperops.slides.layout.containers import LayoutNode
from paperops.slides.components.charts._helpers import format_value

HBarItem = tuple[str, float, str]  # (label, value, color_name)


@dataclass
class HorizontalBarChart(LayoutNode):
    """Horizontal bar chart for rankings.

    items: list of (label, value, color_name)
    Example:
        HorizontalBarChart(items=[
            ("Python", 85, "primary"),
            ("JavaScript", 72, "secondary"),
            ("Rust", 58, "accent"),
        ])
    """

    items: list[HBarItem] = field(default_factory=list)
    x_label: str | None = None
    show_values: bool = True
    max_value: float | None = None

    def preferred_size(self, theme, available_width):
        return (self.width or 10.0, self.height or 4.0)

    def to_svg(self, theme) -> str:
        """Generate SVG string for this chart."""
        canvas_w, canvas_h = 1200, 500
        svg = SvgCanvas(width=canvas_w, height=canvas_h, theme=theme)

        if not self.items:
            return svg.render()

        all_values = [v for _, v, _ in self.items]
        max_val = self.max_value if self.max_value is not None else max(all_values)
        if max_val <= 0:
            max_val = 100

        # --- layout constants ---
        label_width = 200       # space for labels on the left
        right_margin = 80
        top_margin = 40
        bottom_margin = 60
        chart_left = label_width
        chart_right = canvas_w - right_margin
        chart_top = top_margin
        chart_bottom = canvas_h - bottom_margin
        chart_w = chart_right - chart_left
        chart_h = chart_bottom - chart_top

        n_items = len(self.items)
        bar_height = min(40, chart_h / n_items * 0.65)
        bar_gap = chart_h / n_items - bar_height

        # --- x-axis ticks & grid ---
        num_ticks = 5
        for i in range(num_ticks + 1):
            frac = i / num_ticks
            x = chart_left + frac * chart_w
            tick_val = frac * max_val

            svg.line(x, chart_top, x, chart_bottom, color="border", width=1, dashed=True)

            label = f"{tick_val:.0f}" if tick_val == int(tick_val) else f"{tick_val:.1f}"
            svg.text(x, chart_bottom + 20, label,
                     color="text_mid", size=13, anchor="middle")

        # x-axis label
        if self.x_label:
            svg.text((chart_left + chart_right) / 2, chart_bottom + 45, self.x_label,
                     color="text_mid", size=14, anchor="middle")

        # axes
        svg.line(chart_left, chart_top, chart_left, chart_bottom, color="border", width=2)
        svg.line(chart_left, chart_bottom, chart_right, chart_bottom, color="border", width=2)

        # --- draw bars ---
        for idx, (label, value, color_name) in enumerate(self.items):
            by = chart_top + idx * (bar_height + bar_gap) + bar_gap / 2
            bw = (value / max_val) * chart_w

            svg.rect(chart_left, by, bw, bar_height,
                     fill=color_name, stroke="none", rx=4, opacity=0.9)

            # label on the left
            svg.text(chart_left - 12, by + bar_height / 2, label,
                     color="text", size=14, anchor="end")

            # value at bar tip
            if self.show_values:
                val_str = format_value(value)
                svg.text(chart_left + bw + 8, by + bar_height / 2, val_str,
                         color="text", size=13, bold=True, anchor="start")

        return svg.render()
