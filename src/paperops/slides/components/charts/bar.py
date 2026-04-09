"""Bar chart component — grouped bar chart as SVG."""

from __future__ import annotations

from dataclasses import dataclass, field

from paperops.slides.components.svg_canvas import SvgCanvas
from paperops.slides.layout.containers import LayoutNode
from paperops.slides.layout.types import Constraints, IntrinsicSize
from paperops.slides.components.charts._helpers import format_value, render_legend

BarItem = tuple[str, float, str]  # (label, value, color_name)
BarGroup = tuple[str, list[BarItem]]  # (group_label, bars)


@dataclass
class BarChart(LayoutNode):
    """Grouped bar chart.

    groups: list of (group_label, bars) where bars is list of (bar_label, value, color_name)
    Example:
        BarChart(groups=[
            ("alphaNLI", [("LLM", 92, "primary"), ("Human", 91, "secondary")]),
            ("alphaNLG", [("LLM", 79, "warning"), ("Human", 96, "secondary")]),
        ])
    """

    groups: list[BarGroup] = field(default_factory=list)
    y_label: str | None = None
    show_values: bool = True
    max_value: float | None = None  # auto from data if None

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        width = self.width if self.width is not None else (constraints.max_width or 10.0)
        height = self.height if self.height is not None else 4.0
        return IntrinsicSize(
            min_width=self.min_width or min(width, 4.0),
            preferred_width=width,
            min_height=self.min_height or min(height, 2.0),
            preferred_height=height,
        ).clamp(constraints)

    def preferred_size(self, theme, available_width):
        intrinsic = self.measure(Constraints(max_width=max(available_width, 0.0)), theme)
        return intrinsic.preferred_width, intrinsic.preferred_height

    def to_svg(self, theme) -> str:
        """Generate SVG string for this chart."""
        canvas_w, canvas_h = 1200, 500

        svg = SvgCanvas(width=canvas_w, height=canvas_h, theme=theme)

        # --- resolve max value ---
        all_values = [
            v for _, bars in self.groups for _, v, _ in bars
        ]
        if not all_values:
            return svg.render()

        max_val = self.max_value if self.max_value is not None else max(all_values)
        if max_val <= 0:
            max_val = 100

        # --- layout constants ---
        left_margin = 120       # space for y-axis labels
        right_margin = 60
        top_margin = 40
        bottom_margin = 80      # space for group labels
        chart_left = left_margin
        chart_right = canvas_w - right_margin
        chart_top = top_margin
        chart_bottom = canvas_h - bottom_margin
        chart_w = chart_right - chart_left
        chart_h = chart_bottom - chart_top

        bar_width = 60
        bar_gap = 10            # within a group
        group_gap = 80          # between groups

        # --- y-axis ticks ---
        num_ticks = 5
        for i in range(num_ticks + 1):
            frac = i / num_ticks
            y = chart_bottom - frac * chart_h
            tick_val = frac * max_val

            # grid line
            svg.line(chart_left, y, chart_right, y, color="border", width=1, dashed=True)

            # tick label
            label = f"{tick_val:.0f}" if tick_val == int(tick_val) else f"{tick_val:.1f}"
            svg.text(chart_left - 12, y + 5, label, color="text_mid", size=13, anchor="end")

        # y-axis label (rotated — approximate with vertical text placement)
        if self.y_label:
            svg.text(20, (chart_top + chart_bottom) / 2, self.y_label,
                     color="text_mid", size=14, anchor="middle")

        # x-axis baseline
        svg.line(chart_left, chart_bottom, chart_right, chart_bottom, color="border", width=2)
        # y-axis line
        svg.line(chart_left, chart_top, chart_left, chart_bottom, color="border", width=2)

        # --- compute total width needed & centering offset ---
        n_groups = len(self.groups)
        if n_groups == 0:
            return svg.render()

        max_bars = max(len(bars) for _, bars in self.groups)
        group_width = max_bars * bar_width + (max_bars - 1) * bar_gap
        total_width = n_groups * group_width + (n_groups - 1) * group_gap
        x_offset = chart_left + (chart_w - total_width) / 2

        # --- draw bars ---
        for g_idx, (group_label, bars) in enumerate(self.groups):
            group_x = x_offset + g_idx * (group_width + group_gap)
            n_bars = len(bars)
            # center bars within the group slot
            bars_total_w = n_bars * bar_width + (n_bars - 1) * bar_gap
            bar_start_x = group_x + (group_width - bars_total_w) / 2

            for b_idx, (bar_label, value, color_name) in enumerate(bars):
                bx = bar_start_x + b_idx * (bar_width + bar_gap)
                bar_h = (value / max_val) * chart_h
                by = chart_bottom - bar_h

                svg.rect(bx, by, bar_width, bar_h,
                         fill=color_name, stroke="none", rx=4, opacity=0.9)

                # value label above bar
                if self.show_values:
                    val_str = format_value(value)
                    svg.text(bx + bar_width / 2, by - 8, val_str,
                             color="text", size=13, bold=True, anchor="middle")

            # group label below x-axis
            group_center_x = group_x + group_width / 2
            svg.text(group_center_x, chart_bottom + 25, group_label,
                     color="text", size=14, bold=False, anchor="middle")

        # --- legend (using first group's bar labels & colors) ---
        if self.groups:
            _, first_bars = self.groups[0]
            legend_items = [(bar_label, color_name) for bar_label, _, color_name in first_bars]
            render_legend(svg, legend_items, chart_right - 100, chart_top + 10)

        return svg.render()
