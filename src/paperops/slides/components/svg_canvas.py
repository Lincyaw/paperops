"""Structured SVG builder — replaces raw SVG string concatenation."""

from __future__ import annotations

import html


_DEFAULT_FONT = "Calibri, Arial, sans-serif"


class _SvgGroup:
    """A <g> element that exposes the same draw API as SvgCanvas."""

    def __init__(self, canvas: SvgCanvas, transform: str | None = None,
                 opacity: float | None = None):
        self._canvas = canvas
        self._transform = transform
        self._opacity = opacity
        self._elements: list[str] = []

    # -- context manager ------------------------------------------------
    def __enter__(self) -> _SvgGroup:
        return self

    def __exit__(self, *exc):
        attrs = ""
        if self._transform:
            attrs += f' transform="{html.escape(self._transform)}"'
        if self._opacity is not None:
            attrs += f' opacity="{self._opacity}"'
        inner = "\n".join(self._elements)
        self._canvas._elements.append(f"<g{attrs}>\n{inner}\n</g>")

    # -- delegated draw helpers -----------------------------------------
    def _resolve_color(self, color) -> str:
        return self._canvas._resolve_color(color)

    @property
    def _font(self) -> str:
        return self._canvas._font

    # -- draw methods (mirror SvgCanvas, append to self._elements) ------

    def rounded_rect(self, x, y, w, h, text="", fill="bg_alt", stroke="border",
                     text_color="text", font_size=18, rx=12, stroke_width=2,
                     bold=False, opacity=1.0, filter_id=None) -> _SvgGroup:
        self._elements.append(
            _render_rounded_rect(self, x, y, w, h, text, fill, stroke,
                                 text_color, font_size, rx, stroke_width,
                                 bold, opacity, filter_id))
        return self

    def rect(self, x, y, w, h, fill="bg_alt", stroke="border",
             stroke_width=1, rx=0, opacity=1.0) -> _SvgGroup:
        self._elements.append(
            _render_rect(self, x, y, w, h, fill, stroke, stroke_width, rx, opacity))
        return self

    def circle(self, cx, cy, r, text="", fill="primary", text_color="white",
               font_size=16, bold=True, opacity=1.0) -> _SvgGroup:
        self._elements.append(
            _render_circle(self, cx, cy, r, text, fill, text_color,
                           font_size, bold, opacity))
        return self

    def text(self, x, y, text, color="text", size=16, bold=False,
             anchor="middle", font=None, italic=False) -> _SvgGroup:
        self._elements.append(
            _render_text(self, x, y, text, color, size, bold, anchor, font, italic))
        return self

    def line(self, x1, y1, x2, y2, color="border", width=1,
             dashed=False) -> _SvgGroup:
        self._elements.append(
            _render_line(self, x1, y1, x2, y2, color, width, dashed))
        return self

    def arrow(self, x1, y1, x2, y2, color="primary", width=2) -> _SvgGroup:
        self._canvas._ensure_arrow_marker(color)
        self._elements.append(
            _render_arrow(self, x1, y1, x2, y2, color, width))
        return self

    def path(self, d, stroke="primary", fill="none", stroke_width=2,
             dashed=False, marker_end=None) -> _SvgGroup:
        self._elements.append(
            _render_path(self, d, stroke, fill, stroke_width, dashed, marker_end))
        return self

    def polygon(self, points, fill="primary", stroke="none",
                stroke_width=0, opacity=1.0) -> _SvgGroup:
        self._elements.append(
            _render_polygon(self, points, fill, stroke, stroke_width, opacity))
        return self


class SvgCanvas:
    """Structured SVG builder with theme-aware color resolution."""

    def __init__(self, width: int = 1200, height: int = 600,
                 theme=None, bg: str | None = None):
        self.width = width
        self.height = height
        self._theme = theme
        self._bg = bg
        self._defs: list[str] = []
        self._elements: list[str] = []
        self._marker_colors: set[str] = set()  # hex colors with markers already added
        self._font = _DEFAULT_FONT

    # -- color resolution -----------------------------------------------

    def _resolve_color(self, color) -> str:
        """Resolve *color* to a hex string."""
        if color == "none":
            return "none"
        if color == "white":
            return "#FFFFFF"
        if color == "black":
            return "#000000"
        if isinstance(color, str) and color.startswith("url("):
            return color
        if isinstance(color, tuple):
            r, g, b = color
            return f"#{r:02X}{g:02X}{b:02X}"
        if isinstance(color, str) and color.startswith("#"):
            return color
        if self._theme is not None:
            return self._theme.resolve_color(color)
        # No theme and not a recognised literal — pass through (best effort)
        return color

    # -- draw methods ---------------------------------------------------

    def rounded_rect(self, x, y, w, h, text="", fill="bg_alt", stroke="border",
                     text_color="text", font_size=18, rx=12, stroke_width=2,
                     bold=False, opacity=1.0, filter_id=None) -> SvgCanvas:
        self._elements.append(
            _render_rounded_rect(self, x, y, w, h, text, fill, stroke,
                                 text_color, font_size, rx, stroke_width,
                                 bold, opacity, filter_id))
        return self

    def rect(self, x, y, w, h, fill="bg_alt", stroke="border",
             stroke_width=1, rx=0, opacity=1.0) -> SvgCanvas:
        self._elements.append(
            _render_rect(self, x, y, w, h, fill, stroke, stroke_width, rx, opacity))
        return self

    def circle(self, cx, cy, r, text="", fill="primary", text_color="white",
               font_size=16, bold=True, opacity=1.0) -> SvgCanvas:
        self._elements.append(
            _render_circle(self, cx, cy, r, text, fill, text_color,
                           font_size, bold, opacity))
        return self

    def text(self, x, y, text, color="text", size=16, bold=False,
             anchor="middle", font=None, italic=False) -> SvgCanvas:
        self._elements.append(
            _render_text(self, x, y, text, color, size, bold, anchor, font, italic))
        return self

    def line(self, x1, y1, x2, y2, color="border", width=1,
             dashed=False) -> SvgCanvas:
        self._elements.append(
            _render_line(self, x1, y1, x2, y2, color, width, dashed))
        return self

    def arrow(self, x1, y1, x2, y2, color="primary", width=2) -> SvgCanvas:
        self._ensure_arrow_marker(color)
        self._elements.append(
            _render_arrow(self, x1, y1, x2, y2, color, width))
        return self

    def path(self, d, stroke="primary", fill="none", stroke_width=2,
             dashed=False, marker_end=None) -> SvgCanvas:
        self._elements.append(
            _render_path(self, d, stroke, fill, stroke_width, dashed, marker_end))
        return self

    def polygon(self, points, fill="primary", stroke="none",
                stroke_width=0, opacity=1.0) -> SvgCanvas:
        self._elements.append(
            _render_polygon(self, points, fill, stroke, stroke_width, opacity))
        return self

    # -- defs helpers ---------------------------------------------------

    def shadow_filter(self, filter_id="shadow") -> SvgCanvas:
        """Add a drop-shadow filter definition."""
        fid = html.escape(filter_id)
        self._defs.append(
            f'<filter id="{fid}" x="-10%" y="-10%" width="130%" height="130%">'
            f'<feDropShadow dx="2" dy="3" stdDeviation="3" flood-opacity="0.18"/>'
            f'</filter>'
        )
        return self

    def _ensure_arrow_marker(self, color) -> None:
        """Create an arrowhead marker for *color* if one doesn't exist yet."""
        hex_color = self._resolve_color(color)
        if hex_color in self._marker_colors:
            return
        self._marker_colors.add(hex_color)
        mid = _marker_id(hex_color)
        self._defs.append(
            f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{hex_color}"/>'
            f'</marker>'
        )

    def arrow_markers(self, colors=None) -> SvgCanvas:
        """Pre-create arrowhead markers for a set of colors."""
        if colors is None:
            colors = ["primary", "secondary", "positive", "negative",
                      "accent", "text_mid"]
        for c in colors:
            self._ensure_arrow_marker(c)
        return self

    def gradient(self, grad_id: str, color1, color2,
                 direction: str = "vertical") -> SvgCanvas:
        """Add a linear gradient definition."""
        gid = html.escape(grad_id)
        c1 = self._resolve_color(color1)
        c2 = self._resolve_color(color2)
        if direction == "horizontal":
            x1, y1, x2, y2 = 0, 0, 1, 0
        else:  # vertical
            x1, y1, x2, y2 = 0, 0, 0, 1
        self._defs.append(
            f'<linearGradient id="{gid}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">'
            f'<stop offset="0%" stop-color="{c1}"/>'
            f'<stop offset="100%" stop-color="{c2}"/>'
            f'</linearGradient>'
        )
        return self

    # -- group ----------------------------------------------------------

    def group(self, transform=None, opacity=None) -> _SvgGroup:
        """Return a group context manager."""
        return _SvgGroup(self, transform, opacity)

    # -- render ---------------------------------------------------------

    def render(self) -> str:
        """Assemble and return the complete SVG string."""
        parts: list[str] = []
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">'
        )

        # defs
        if self._defs:
            parts.append("<defs>")
            parts.extend(self._defs)
            parts.append("</defs>")

        # background
        if self._bg is not None:
            bg_hex = self._resolve_color(self._bg)
            parts.append(
                f'<rect width="{self.width}" height="{self.height}" fill="{bg_hex}"/>'
            )

        # elements
        parts.extend(self._elements)

        parts.append("</svg>")
        return "\n".join(parts)


# ======================================================================
# Shared rendering helpers (used by both SvgCanvas and _SvgGroup)
# ======================================================================

def _marker_id(hex_color: str) -> str:
    """Deterministic marker id from hex color."""
    return "arrow_" + hex_color.lstrip("#")


def _render_rounded_rect(ctx, x, y, w, h, text, fill, stroke,
                         text_color, font_size, rx, stroke_width,
                         bold, opacity, filter_id) -> str:
    fill_hex = ctx._resolve_color(fill)
    stroke_hex = ctx._resolve_color(stroke)
    parts: list[str] = []
    filt = f' filter="url(#{html.escape(filter_id)})"' if filter_id else ""
    opa = f' opacity="{opacity}"' if opacity != 1.0 else ""
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill_hex}" stroke="{stroke_hex}" stroke-width="{stroke_width}"'
        f'{opa}{filt}/>'
    )
    if text:
        tc = ctx._resolve_color(text_color)
        fw = "bold" if bold else "normal"
        center_x = x + w / 2
        center_y = y + h / 2
        lines = str(text).split("\n")
        if len(lines) == 1:
            escaped = html.escape(lines[0])
            parts.append(
                f'<text x="{center_x}" y="{center_y}" text-anchor="middle" '
                f'dominant-baseline="central" fill="{tc}" '
                f'font-family="{ctx._font}" font-size="{font_size}" '
                f'font-weight="{fw}">{escaped}</text>'
            )
        else:
            line_height = font_size * 1.2
            block_height = line_height * (len(lines) - 1)
            start_y = center_y - block_height / 2
            inner = ""
            for i, line in enumerate(lines):
                escaped = html.escape(line)
                dy = "0" if i == 0 else f"{line_height:.1f}"
                inner += f'<tspan x="{center_x}" dy="{dy}">{escaped}</tspan>'
            parts.append(
                f'<text x="{center_x}" y="{start_y}" text-anchor="middle" '
                f'dominant-baseline="central" fill="{tc}" '
                f'font-family="{ctx._font}" font-size="{font_size}" '
                f'font-weight="{fw}">{inner}</text>'
            )
    return "\n".join(parts)


def _render_rect(ctx, x, y, w, h, fill, stroke, stroke_width, rx, opacity) -> str:
    fill_hex = ctx._resolve_color(fill)
    stroke_hex = ctx._resolve_color(stroke)
    opa = f' opacity="{opacity}"' if opacity != 1.0 else ""
    rx_attr = f' rx="{rx}"' if rx else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}"{rx_attr} '
        f'fill="{fill_hex}" stroke="{stroke_hex}" stroke-width="{stroke_width}"{opa}/>'
    )


def _render_circle(ctx, cx, cy, r, text, fill, text_color,
                   font_size, bold, opacity) -> str:
    fill_hex = ctx._resolve_color(fill)
    opa = f' opacity="{opacity}"' if opacity != 1.0 else ""
    parts = [
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill_hex}"{opa}/>'
    ]
    if text:
        tc = ctx._resolve_color(text_color)
        fw = "bold" if bold else "normal"
        escaped = html.escape(str(text))
        parts.append(
            f'<text x="{cx}" y="{cy}" text-anchor="middle" '
            f'dominant-baseline="central" fill="{tc}" '
            f'font-family="{ctx._font}" font-size="{font_size}" '
            f'font-weight="{fw}">{escaped}</text>'
        )
    return "\n".join(parts)


def _render_text(ctx, x, y, text, color, size, bold, anchor, font, italic) -> str:
    c = ctx._resolve_color(color)
    fw = "bold" if bold else "normal"
    fs = "italic" if italic else "normal"
    fam = font or ctx._font
    lines = str(text).split("\n")
    if len(lines) == 1:
        escaped = html.escape(lines[0])
        return (
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'dominant-baseline="central" fill="{c}" '
            f'font-family="{fam}" font-size="{size}" '
            f'font-weight="{fw}" font-style="{fs}">{escaped}</text>'
        )
    # multiline
    inner = ""
    for i, line in enumerate(lines):
        escaped = html.escape(line)
        dy = "0" if i == 0 else f"{size * 1.2:.1f}"
        inner += f'<tspan x="{x}" dy="{dy}">{escaped}</tspan>'
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'dominant-baseline="central" fill="{c}" '
        f'font-family="{fam}" font-size="{size}" '
        f'font-weight="{fw}" font-style="{fs}">{inner}</text>'
    )


def _render_line(ctx, x1, y1, x2, y2, color, width, dashed) -> str:
    c = ctx._resolve_color(color)
    dash = ' stroke-dasharray="6,4"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{c}" stroke-width="{width}"{dash}/>'
    )


def _render_arrow(ctx, x1, y1, x2, y2, color, width) -> str:
    c = ctx._resolve_color(color)
    mid = _marker_id(c)
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{c}" stroke-width="{width}" marker-end="url(#{mid})"/>'
    )


def _render_path(ctx, d, stroke, fill, stroke_width, dashed, marker_end) -> str:
    s = ctx._resolve_color(stroke)
    f = ctx._resolve_color(fill)
    dash = ' stroke-dasharray="6,4"' if dashed else ""
    me = f' marker-end="url(#{html.escape(marker_end)})"' if marker_end else ""
    return (
        f'<path d="{html.escape(d)}" stroke="{s}" fill="{f}" '
        f'stroke-width="{stroke_width}"{dash}{me}/>'
    )


def _render_polygon(ctx, points, fill, stroke, stroke_width, opacity) -> str:
    f = ctx._resolve_color(fill)
    s = ctx._resolve_color(stroke)
    if isinstance(points, list):
        pts_str = " ".join(f"{px},{py}" for px, py in points)
    else:
        pts_str = str(points)
    opa = f' opacity="{opacity}"' if opacity != 1.0 else ""
    return (
        f'<polygon points="{pts_str}" fill="{f}" stroke="{s}" '
        f'stroke-width="{stroke_width}"{opa}/>'
    )


# ======================================================================
# Quick self-test
# ======================================================================

if __name__ == "__main__":
    import sys, os, types
    # Register package stubs so we can import core.theme without triggering
    # the full slidecraft __init__.py (which may reference not-yet-created modules).
    _pkg_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, _pkg_root)
    for _stub in ("slidecraft", "slidecraft.core", "slidecraft.components"):
        if _stub not in sys.modules:
            _m = types.ModuleType(_stub)
            _m.__path__ = [os.path.join(_pkg_root, _stub.replace(".", "/"))]
            sys.modules[_stub] = _m
    from paperops.slides.core.theme import themes

    theme = themes.professional

    svg = (
        SvgCanvas(800, 400, theme=theme, bg="bg")
        .shadow_filter("shadow")
        .arrow_markers()
        .gradient("grad1", "primary", "secondary", direction="horizontal")
        .rounded_rect(50, 50, 250, 80, text="A & B < C",
                      fill="bg_alt", stroke="primary",
                      text_color="text", font_size=20, bold=True,
                      filter_id="shadow")
        .arrow(300, 90, 450, 90, color="primary", width=2)
        .circle(500, 90, 30, text="?", fill="accent")
        .text(400, 200, "Special chars: <tag> & \"quotes\"",
              color="text_mid", size=14, italic=True)
        .line(50, 250, 750, 250, color="border", dashed=True)
        .rect(50, 270, 700, 80, fill="url(#grad1)", stroke="none", rx=8)
        .polygon([(400, 370), (420, 395), (380, 395)], fill="negative")
    )

    print(svg.render())
