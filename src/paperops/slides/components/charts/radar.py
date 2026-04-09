"""Radar (spider) chart component — SVG."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from paperops.slides.components.svg_canvas import SvgCanvas
from paperops.slides.layout.containers import LayoutNode
from paperops.slides.layout.types import Constraints, IntrinsicSize
from paperops.slides.components.charts._helpers import render_legend

RadarSeries = tuple[str, list[float], str]  # (name, values_0_to_1, color_name)


@dataclass
class RadarChart(LayoutNode):
    """Spider/radar chart comparing multiple series.

    dimensions: list of dimension names
    series: list of (name, values, color_name) where values is list of floats 0-1
    Example:
        RadarChart(
            dimensions=["Selection", "Generation", "Counterfactual", "Real-world", "IT Ops"],
            series=[
                ("Human", [0.91, 0.96, 0.95, 0.90, 0.85], "secondary"),
                ("LLM", [0.93, 0.79, 0.80, 0.50, 0.14], "primary"),
            ]
        )
    """

    dimensions: list[str] = field(default_factory=list)
    series: list[RadarSeries] = field(default_factory=list)

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        width = self.width if self.width is not None else (constraints.max_width or 8.0)
        height = self.height if self.height is not None else 5.0
        return IntrinsicSize(
            min_width=self.min_width or min(width, 4.0),
            preferred_width=width,
            min_height=self.min_height or min(height, 2.5),
            preferred_height=height,
        ).clamp(constraints)

    def preferred_size(self, theme, available_width):
        intrinsic = self.measure(Constraints(max_width=max(available_width, 0.0)), theme)
        return intrinsic.preferred_width, intrinsic.preferred_height

    def to_svg(self, theme) -> str:
        """Generate SVG string for this chart."""
        canvas_w, canvas_h = 900, 600
        svg = SvgCanvas(width=canvas_w, height=canvas_h, theme=theme)

        n = len(self.dimensions)
        if n < 3:
            return svg.render()

        cx, cy = 420, 300
        max_r = 200

        # angle for each dimension (start from top, i.e. -pi/2)
        angles = [(-math.pi / 2 + 2 * math.pi * i / n) for i in range(n)]

        def polar(angle, r):
            return cx + r * math.cos(angle), cy + r * math.sin(angle)

        # --- grid circles ---
        grid_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
        for level in grid_levels:
            r = max_r * level
            # draw polygon grid instead of circles for radar style
            points = [polar(a, r) for a in angles]
            points_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            svg.polygon(points_str, fill="none", stroke="border")
            # percentage label on first axis
            px, py = polar(angles[0], r)
            svg.text(px + 8, py - 4, f"{int(level * 100)}%",
                     color="text_light", size=10, anchor="start")

        # --- axis lines ---
        for i, angle in enumerate(angles):
            ex, ey = polar(angle, max_r)
            svg.line(cx, cy, ex, ey, color="border", width=1)

        # --- dimension labels ---
        label_margin = 28
        for i, dim in enumerate(self.dimensions):
            lx, ly = polar(angles[i], max_r + label_margin)
            # adjust anchor based on position
            if abs(math.cos(angles[i])) < 0.1:
                anchor = "middle"
            elif math.cos(angles[i]) > 0:
                anchor = "start"
            else:
                anchor = "end"
            # nudge y for top/bottom labels
            if math.sin(angles[i]) < -0.5:
                ly -= 6
            elif math.sin(angles[i]) > 0.5:
                ly += 12
            svg.text(lx, ly, dim, color="text", size=14, bold=False, anchor=anchor)

        # --- series polygons ---
        for name, values, color_name in self.series:
            color_hex = theme.resolve_color(color_name)
            pts = []
            for i, val in enumerate(values):
                val = max(0.0, min(1.0, val))
                px, py = polar(angles[i], max_r * val)
                pts.append((px, py))

            points_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            svg.polygon(points_str, fill=color_name, stroke=color_name)

            # dots on vertices
            for px, py in pts:
                svg.circle(px, py, 4, fill=color_name, text_color="white",
                           font_size=1, bold=False)

        # --- legend ---
        legend_items = [(name, color_name) for name, _, color_name in self.series]
        render_legend(svg, legend_items, 720, 30, row_height=28)

        return svg.render()
