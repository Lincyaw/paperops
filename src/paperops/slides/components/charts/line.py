"""Line chart component — multi-series line chart as SVG."""

from __future__ import annotations

from dataclasses import dataclass, field

from paperops.slides.components.svg_canvas import SvgCanvas
from paperops.slides.layout.containers import LayoutNode
from paperops.slides.components.charts._helpers import format_value, render_legend

LineSeries = tuple[str, list[float], str]  # (name, values, color_name)


@dataclass
class LineChart(LayoutNode):
    """Line chart for trends.

    series: list of (name, values, color_name) where values is list of floats
    x_labels: list of x-axis labels
    Example:
        LineChart(
            series=[
                ("Revenue", [10, 25, 40, 55, 70], "primary"),
                ("Cost", [30, 28, 25, 22, 20], "secondary"),
            ],
            x_labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        )
    """

    series: list[LineSeries] = field(default_factory=list)
    x_labels: list[str] = field(default_factory=list)
    y_label: str | None = None
    show_values: bool = False
    show_dots: bool = True
    max_value: float | None = None

    def preferred_size(self, theme, available_width):
        return (self.width or 10.0, self.height or 4.0)

    def to_svg(self, theme) -> str:
        """Generate SVG string for this chart."""
        canvas_w, canvas_h = 1200, 500
        svg = SvgCanvas(width=canvas_w, height=canvas_h, theme=theme)

        # --- collect all values ---
        all_values = [v for _, values, _ in self.series for v in values]
        if not all_values:
            return svg.render()

        max_val = self.max_value if self.max_value is not None else max(all_values)
        if max_val <= 0:
            max_val = 100

        # --- layout constants ---
        left_margin = 120
        right_margin = 60
        top_margin = 40
        bottom_margin = 80
        chart_left = left_margin
        chart_right = canvas_w - right_margin
        chart_top = top_margin
        chart_bottom = canvas_h - bottom_margin
        chart_w = chart_right - chart_left
        chart_h = chart_bottom - chart_top

        # --- y-axis ticks & grid ---
        num_ticks = 5
        for i in range(num_ticks + 1):
            frac = i / num_ticks
            y = chart_bottom - frac * chart_h
            tick_val = frac * max_val

            svg.line(chart_left, y, chart_right, y, color="border", width=1, dashed=True)

            label = f"{tick_val:.0f}" if tick_val == int(tick_val) else f"{tick_val:.1f}"
            svg.text(chart_left - 12, y + 5, label,
                     color="text_mid", size=13, anchor="end")

        # y-axis label
        if self.y_label:
            svg.text(20, (chart_top + chart_bottom) / 2, self.y_label,
                     color="text_mid", size=14, anchor="middle")

        # axes
        svg.line(chart_left, chart_bottom, chart_right, chart_bottom, color="border", width=2)
        svg.line(chart_left, chart_top, chart_left, chart_bottom, color="border", width=2)

        # --- determine x positions ---
        n_points = max(len(values) for _, values, _ in self.series) if self.series else 0
        if n_points == 0:
            return svg.render()

        if n_points == 1:
            x_positions = [chart_left + chart_w / 2]
        else:
            x_positions = [
                chart_left + i * chart_w / (n_points - 1) for i in range(n_points)
            ]

        # --- x-axis labels ---
        for i, xp in enumerate(x_positions):
            if i < len(self.x_labels):
                svg.text(xp, chart_bottom + 25, self.x_labels[i],
                         color="text", size=14, anchor="middle")

        # --- draw series ---
        for name, values, color_name in self.series:
            points: list[tuple[float, float]] = []
            for i, val in enumerate(values):
                if i >= len(x_positions):
                    break
                xp = x_positions[i]
                yp = chart_bottom - (val / max_val) * chart_h
                points.append((xp, yp))

            # polyline path
            if len(points) >= 2:
                d_parts = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
                for px, py in points[1:]:
                    d_parts.append(f"L {px:.1f} {py:.1f}")
                svg.path(" ".join(d_parts), stroke=color_name, fill="none",
                         stroke_width=3)

            # dots and values
            for val_idx, (px, py) in enumerate(points):
                if self.show_dots:
                    svg.circle(px, py, 5, fill=color_name, text_color="white",
                               font_size=1, bold=False)
                if self.show_values:
                    val = values[val_idx]
                    val_str = format_value(val)
                    svg.text(px, py - 14, val_str,
                             color="text", size=12, bold=True, anchor="middle")

        # --- legend ---
        legend_items = [(name, color_name) for name, _, color_name in self.series]
        render_legend(svg, legend_items, chart_right - 100, chart_top + 10)

        return svg.render()
