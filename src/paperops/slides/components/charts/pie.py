"""Pie / donut chart component — SVG."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from paperops.slides.components.svg_canvas import SvgCanvas
from paperops.slides.layout.containers import LayoutNode
from paperops.slides.components.charts._helpers import render_legend

PieSlice = tuple[str, float, str]  # (label, value, color_name)


@dataclass
class PieChart(LayoutNode):
    """Pie/donut chart.

    slices: list of (label, value, color_name)
    Example:
        PieChart(slices=[
            ("Desktop", 60, "primary"),
            ("Mobile", 30, "secondary"),
            ("Tablet", 10, "accent"),
        ])
    """

    slices: list[PieSlice] = field(default_factory=list)
    donut: bool = False
    show_labels: bool = True
    show_percentages: bool = True

    def preferred_size(self, theme, available_width):
        return (self.width or 8.0, self.height or 5.0)

    def to_svg(self, theme) -> str:
        """Generate SVG string for this chart."""
        canvas_w, canvas_h = 900, 600
        svg = SvgCanvas(width=canvas_w, height=canvas_h, theme=theme)

        if not self.slices:
            return svg.render()

        total = sum(val for _, val, _ in self.slices)
        if total <= 0:
            return svg.render()

        cx, cy = 400, 300
        radius = 220
        inner_radius = radius * 0.55 if self.donut else 0

        # --- draw slices ---
        start_angle = -math.pi / 2  # start from top

        for label, value, color_name in self.slices:
            if value <= 0:
                continue
            fraction = value / total
            sweep = fraction * 2 * math.pi

            end_angle = start_angle + sweep

            # arc endpoints
            x1 = cx + radius * math.cos(start_angle)
            y1 = cy + radius * math.sin(start_angle)
            x2 = cx + radius * math.cos(end_angle)
            y2 = cy + radius * math.sin(end_angle)

            large_arc = 1 if sweep > math.pi else 0

            if self.donut:
                # outer arc, line to inner, inner arc (reverse), close
                ix1 = cx + inner_radius * math.cos(end_angle)
                iy1 = cy + inner_radius * math.sin(end_angle)
                ix2 = cx + inner_radius * math.cos(start_angle)
                iy2 = cy + inner_radius * math.sin(start_angle)

                d = (
                    f"M {x1:.2f} {y1:.2f} "
                    f"A {radius} {radius} 0 {large_arc} 1 {x2:.2f} {y2:.2f} "
                    f"L {ix1:.2f} {iy1:.2f} "
                    f"A {inner_radius} {inner_radius} 0 {large_arc} 0 {ix2:.2f} {iy2:.2f} "
                    f"Z"
                )
            else:
                d = (
                    f"M {cx} {cy} "
                    f"L {x1:.2f} {y1:.2f} "
                    f"A {radius} {radius} 0 {large_arc} 1 {x2:.2f} {y2:.2f} "
                    f"Z"
                )

            svg.path(d, stroke="white", fill=color_name, stroke_width=2)

            # --- label ---
            if self.show_labels or self.show_percentages:
                mid_angle = start_angle + sweep / 2
                label_r = radius + 30
                lx = cx + label_r * math.cos(mid_angle)
                ly = cy + label_r * math.sin(mid_angle)

                anchor = "start" if math.cos(mid_angle) >= 0 else "end"

                parts: list[str] = []
                if self.show_labels:
                    parts.append(label)
                if self.show_percentages:
                    parts.append(f"{fraction * 100:.1f}%")
                text = " ".join(parts)

                svg.text(lx, ly, text, color="text", size=13, anchor=anchor)

            start_angle = end_angle

        # donut center hole (white overlay)
        if self.donut:
            svg.circle(cx, cy, int(inner_radius), fill="white",
                       text_color="white", font_size=1, bold=False)

        # --- legend ---
        legend_items = [(label, color_name) for label, _, color_name in self.slices]
        render_legend(svg, legend_items, 720, 80, row_height=28)

        return svg.render()
