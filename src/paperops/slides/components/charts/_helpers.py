"""Shared helpers for chart components."""

from __future__ import annotations


def format_value(val: float) -> str:
    """Format a numeric value for display: integer if whole, else 1 decimal."""
    return f"{val:.0f}" if val == int(val) else f"{val:.1f}"


def render_legend(svg, items: list[tuple[str, str]], x: float, y: float,
                  row_height: float = 24, swatch_size: float = 16) -> None:
    """Draw a legend on an SvgCanvas.

    items: list of (label, color_name) pairs.
    """
    for i, (label, color_name) in enumerate(items):
        ly = y + i * row_height
        svg.rect(x, ly, swatch_size, swatch_size,
                 fill=color_name, stroke="none", rx=3)
        svg.text(x + swatch_size + 8, ly + swatch_size - 3, label,
                 color="text", size=13, anchor="start")
