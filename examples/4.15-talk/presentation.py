from __future__ import annotations

import sys
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paperops.slides import (  # noqa: E402
    Arrow,
    Badge,
    Box,
    BulletList,
    Circle,
    Grid,
    HStack,
    Padding,
    Presentation,
    RoundedBox,
    Spacer,
    SvgImage,
    Table,
    TextBlock,
    VStack,
    themes,
)

OUTPUT_FILE = Path(__file__).with_name("talk_4_15.pptx")
SVG_ASSET_DIR = Path(__file__).with_name("talk_4_15_svg_assets")

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

THEME = themes.academic_seminar.override(
    colors={
        "primary": "#1C4E80",
        "secondary": "#5B7C99",
        "accent": "#C7772E",
        "positive": "#2D6A4F",
        "negative": "#B04A4A",
        "highlight": "#6A5AA8",
        "warning": "#B8892E",
        "bg": "#FBFCFE",
        "bg_alt": "#F2F5F8",
        "bg_accent": "#EAF0F6",
    },
    fonts={
        "title": 28,
        "subtitle": 18,
        "heading": 19,
        "body": 15,
        "caption": 11,
        "small": 9,
    },
)


# ---------------------------------------------------------------------------
# Local SVG builder — custom SVG string generation, no DeckSvg dependency
# ---------------------------------------------------------------------------


class DeckSvg:
    def __init__(self, width: int = 1200, height: int = 600, *, theme=THEME, bg: str | None = None):
        self.width = width
        self.height = height
        self.theme = theme
        self.bg = bg
        self._defs: list[str] = []
        self._elements: list[str] = []
        self._marker_colors: set[str] = set()

    def _color(self, color: str | tuple[int, int, int] | None) -> str:
        if color is None:
            return "none"
        if color == "none":
            return "none"
        if isinstance(color, tuple):
            r, g, b = color
            return f"rgb({r},{g},{b})"
        try:
            return self.theme.resolve_color(color)
        except ValueError:
            # Allow raw SVG/CSS color values for slide-local accents like white text.
            return color

    def _font_size(self, size: str | int | float) -> float:
        return self.theme.resolve_font_size(size)

    @staticmethod
    def _fmt(value: float | int) -> str:
        return f"{float(value):.2f}".rstrip("0").rstrip(".")

    def _attrs(self, **attrs: object) -> str:
        parts: list[str] = []
        for key, value in attrs.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    parts.append(f'{key}="true"')
                continue
            parts.append(f'{key}="{escape(str(value), quote=True)}"')
        return " ".join(parts)

    def _dasharray(self, dashed: bool) -> str | None:
        return "8 6" if dashed else None

    def render(self) -> str:
        bg_rect = ""
        if self.bg is not None:
            bg_rect = (
                f'<rect x="0" y="0" width="{self.width}" height="{self.height}" '
                f'fill="{self._color(self.bg)}" />'
            )
        defs = f"<defs>{''.join(self._defs)}</defs>" if self._defs else ""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">'
            f"{defs}{bg_rect}{''.join(self._elements)}</svg>"
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 1.0,
        rx: float = 0.0,
        opacity: float | None = None,
        text: str | None = None,
        text_color: str = "text",
        font_size: str | int | float = 12,
        bold: bool = False,
    ) -> None:
        self._elements.append(
            f'<rect {self._attrs(x=self._fmt(x), y=self._fmt(y), width=self._fmt(w), height=self._fmt(h), rx=self._fmt(rx) if rx else None, fill=self._color(fill), stroke=self._color(stroke), **{"stroke-width": self._fmt(stroke_width), "opacity": opacity})} />'
        )
        if text:
            self.text(
                x + w / 2,
                y + h / 2 + self._font_size(font_size) * 0.35,
                text,
                color=text_color,
                size=font_size,
                bold=bold,
            )

    def rounded_rect(self, x: float, y: float, w: float, h: float, **kwargs: object) -> None:
        kwargs.setdefault("rx", 14)
        self.rect(x, y, w, h, **kwargs)

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 1.0,
        opacity: float | None = None,
        text: str | None = None,
        color: str = "text",
        size: str | int | float = 12,
        bold: bool = False,
    ) -> None:
        self._elements.append(
            f'<circle {self._attrs(cx=self._fmt(cx), cy=self._fmt(cy), r=self._fmt(r), fill=self._color(fill), stroke=self._color(stroke), **{"stroke-width": self._fmt(stroke_width), "opacity": opacity})} />'
        )
        if text:
            self.text(cx, cy + self._font_size(size) * 0.35, text, color=color, size=size, bold=bold)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = "text",
        width: float = 1.0,
        opacity: float | None = None,
        dashed: bool = False,
        marker_end: str | None = None,
    ) -> None:
        self._elements.append(
            f'<line {self._attrs(x1=self._fmt(x1), y1=self._fmt(y1), x2=self._fmt(x2), y2=self._fmt(y2), stroke=self._color(color), **{"stroke-width": self._fmt(width), "opacity": opacity, "stroke-dasharray": self._dasharray(dashed), "marker-end": marker_end})} />'
        )

    def path(
        self,
        d: str,
        *,
        stroke: str = "text",
        fill: str = "none",
        stroke_width: float = 1.0,
        opacity: float | None = None,
        dashed: bool = False,
    ) -> None:
        self._elements.append(
            f'<path {self._attrs(d=d, stroke=self._color(stroke), fill=self._color(fill), **{"stroke-width": self._fmt(stroke_width), "opacity": opacity, "stroke-dasharray": self._dasharray(dashed)})} />'
        )

    def polygon(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 1.0,
        opacity: float | None = None,
    ) -> None:
        pts = " ".join(f"{self._fmt(x)},{self._fmt(y)}" for x, y in points)
        self._elements.append(
            f'<polygon {self._attrs(points=pts, fill=self._color(fill), stroke=self._color(stroke), **{"stroke-width": self._fmt(stroke_width), "opacity": opacity})} />'
        )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        color: str = "text",
        size: str | int | float = 12,
        bold: bool = False,
        italic: bool = False,
        anchor: str = "middle",
        opacity: float | None = None,
    ) -> None:
        anchor_map = {"start": "start", "middle": "middle", "end": "end"}
        self._elements.append(
            f'<text {self._attrs(x=self._fmt(x), y=self._fmt(y), fill=self._color(color), **{"font-size": self._fmt(self._font_size(size)), "font-family": self.theme.font_family, "font-weight": "700" if bold else "400", "font-style": "italic" if italic else None, "text-anchor": anchor_map.get(anchor, "middle"), "opacity": opacity})}>{escape(text)}</text>'
        )

    def arrow_markers(self, colors: list[str] | None = None) -> None:
        if not colors:
            return
        for color in colors:
            self._ensure_arrow_marker(color)

    def _ensure_arrow_marker(self, color: str) -> str:
        hex_color = self._color(color).replace("#", "")
        marker_id = f"arrow-{hex_color.lower()}"
        if marker_id not in self._marker_colors:
            self._marker_colors.add(marker_id)
            self._defs.append(
                f'<marker id="{marker_id}" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">'
                f'<path d="M0,0 L0,6 L9,3 z" fill="{self._color(color)}" /></marker>'
            )
        return f"url(#{marker_id})"

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = "accent",
        width: float = 2.0,
        opacity: float | None = None,
    ) -> None:
        self.line(
            x1,
            y1,
            x2,
            y2,
            color=color,
            width=width,
            opacity=opacity,
            marker_end=self._ensure_arrow_marker(color),
        )

def _draw_anomaly_burst(
    c: DeckSvg,
    *,
    cx: float,
    cy: float,
    tone: str = "warning",
    scale: float = 1.0,
) -> None:
    radii = [10, 16, 22]
    for idx, radius in enumerate(radii):
        c.circle(cx, cy, radius * scale, fill="none", text="", opacity=max(0.12, 0.28 - idx * 0.07))
        c.path(
            (
                f"M {cx - radius * scale} {cy} "
                f"A {radius * scale} {radius * scale} 0 1 0 {cx + radius * scale} {cy} "
                f"A {radius * scale} {radius * scale} 0 1 0 {cx - radius * scale} {cy}"
            ),
            stroke=tone,
            fill="none",
            stroke_width=max(1.5, 3 - idx * 0.6),
        )
    c.circle(cx, cy, 4.5 * scale, fill=tone, opacity=0.95)


def _draw_root_cause_mark(
    c: DeckSvg,
    *,
    cx: float,
    cy: float,
    tone: str = "negative",
    scale: float = 1.0,
) -> None:
    _draw_anomaly_burst(c, cx=cx, cy=cy, tone=tone, scale=scale)
    bolt = [
        (cx - 4 * scale, cy - 11 * scale),
        (cx + 1 * scale, cy - 11 * scale),
        (cx - 2 * scale, cy - 1 * scale),
        (cx + 6 * scale, cy - 1 * scale),
        (cx - 6 * scale, cy + 13 * scale),
        (cx - 1 * scale, cy + 2 * scale),
        (cx - 8 * scale, cy + 2 * scale),
    ]
    c.polygon(bolt, fill="white", stroke="none", opacity=0.98)


def _draw_service_module(
    c: DeckSvg,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    status: str = "normal",
    badge: str | None = None,
    style: str = "rich",
) -> None:
    palette = {
        "normal": {
            "fill": "bg_alt",
            "stroke": "secondary",
            "header": "secondary",
            "text": "text",
            "subtext": "text_mid",
            "accent": "secondary",
            "halo": "secondary",
        },
        "propagated": {
            "fill": "#FFF4E8",
            "stroke": "warning",
            "header": "warning",
            "text": "text",
            "subtext": "#805B1A",
            "accent": "warning",
            "halo": "warning",
        },
        "root": {
            "fill": "#FCECEC",
            "stroke": "negative",
            "header": "negative",
            "text": "text",
            "subtext": "#873B3B",
            "accent": "negative",
            "halo": "negative",
        },
    }[status]

    if style == "minimal":
        c.rect(x, y, w, h, fill=palette["fill"], stroke=palette["stroke"], stroke_width=2.0, rx=16)
        c.rect(x + 10, y + 10, 8, h - 20, fill=palette["header"], stroke="none", rx=4, opacity=0.92)
        c.circle(x + w - 18, y + 18, 5.5, fill=palette["accent"], opacity=0.95)
        c.text(x + 34, y + 33, title, color=palette["text"], size=15, bold=True, anchor="start")
        c.text(x + 34, y + 54, subtitle, color=palette["subtext"], size=11, anchor="start")
        c.line(x + 34, y + h - 21, x + w - 20, y + h - 21, color="border", width=1.4)
    else:
        if status != "normal":
            c.rect(x - 7, y - 7, w + 14, h + 14, fill="none", stroke=palette["halo"], stroke_width=2, rx=22, opacity=0.18)

        c.rect(x, y, w, h, fill=palette["fill"], stroke=palette["stroke"], stroke_width=2.4, rx=18)
        c.rect(x + 12, y + 11, w - 24, 13, fill=palette["header"], stroke="none", rx=6, opacity=0.90)

        port_x = x + 18
        for py in [y + 28, y + 44, y + 60]:
            c.circle(port_x, py, 3.2, fill=palette["accent"], opacity=0.95)
            c.line(port_x + 7, py, x + w - 22, py, color="border", width=1.4)

        c.path(
            f"M {x + 54} {y + h - 18} Q {x + 72} {y + h - 26} {x + 89} {y + h - 18} "
            f"Q {x + 102} {y + h - 12} {x + 116} {y + h - 26} "
            f"Q {x + 132} {y + h - 42} {x + 151} {y + h - 28}",
            stroke=palette["accent"],
            fill="none",
            stroke_width=2.2,
        )

        c.text(x + 46, y + 28, title, color=palette["text"], size=16, bold=True, anchor="start")
        c.text(x + 46, y + 50, subtitle, color=palette["subtext"], size=12, anchor="start")

    if badge:
        badge_w = max(74, len(badge) * 6.1)
        badge_x = x + w - badge_w - 14
        badge_y = y + 12
        c.rect(badge_x, badge_y, badge_w, 18, fill=palette["header"], stroke="none", rx=9, opacity=0.96)
        c.text(badge_x + badge_w / 2, badge_y + 12, badge.upper(), color="white", size=10, bold=True)

    if style != "minimal" and status == "propagated":
        _draw_anomaly_burst(c, cx=x + w - 18, cy=y + h / 2, tone="warning", scale=0.52)

    if style != "minimal" and status == "root":
        _draw_root_cause_mark(c, cx=x + w - 20, cy=y + h / 2, tone="negative", scale=0.58)
        crack = [
            (x + w - 55, y + 1),
            (x + w - 34, y + 1),
            (x + w - 40, y + 14),
            (x + w - 26, y + 28),
            (x + w - 44, y + 28),
            (x + w - 38, y + 42),
            (x + w - 58, y + 27),
            (x + w - 46, y + 16),
        ]
        c.polygon(crack, fill=palette["header"], stroke="none", opacity=0.95)


def _draw_path_connector(
    c: DeckSvg,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    tone: str = "accent",
    status: str = "verified",
    label: str | None = None,
) -> None:
    x1, y1 = start
    x2, y2 = end
    stroke_map = {
        "verified": tone,
        "candidate": "warning",
        "context": "secondary",
    }
    stroke = stroke_map[status]
    if status == "verified":
        c.line(x1, y1, x2, y2, color=stroke, width=3.0)
    else:
        c.path(
            f"M {x1} {y1} L {x2} {y2}",
            stroke=stroke,
            fill="none",
            stroke_width=2.4,
            dashed=True,
        )

    midx = (x1 + x2) / 2
    midy = (y1 + y2) / 2
    c.circle(midx, midy, 4.2 if status == "verified" else 3.5, fill=stroke, opacity=0.95 if status == "verified" else 0.75)
    c.circle(x2, y2, 4.6, fill=stroke, opacity=0.95 if status == "verified" else 0.75)
    arrow = [
        (x2, y2),
        (x2 - 9, y2 - 5),
        (x2 - 7, y2),
        (x2 - 9, y2 + 5),
    ]
    c.polygon(arrow, fill=stroke, stroke="none", opacity=0.95 if status == "verified" else 0.75)

    if label:
        c.text(midx, midy - 10, label, color=stroke, size=10, bold=True)


def _draw_intervention_point(
    c: DeckSvg,
    *,
    cx: float,
    cy: float,
    tone: str = "accent",
    label: str | None = None,
) -> None:
    c.circle(cx, cy, 11, fill="bg")
    _draw_circle_outline(c, cx=cx, cy=cy, r=11, tone=tone, stroke_width=2.5)
    c.line(cx - 7, cy, cx + 7, cy, color=tone, width=2.6)
    c.line(cx, cy - 7, cx, cy + 7, color=tone, width=2.6)
    if label:
        c.text(cx, cy + 23, label, color=tone, size=10, bold=True)


def _draw_verification_checkpoint(
    c: DeckSvg,
    *,
    cx: float,
    cy: float,
    tone: str = "positive",
    label: str | None = None,
) -> None:
    c.circle(cx, cy, 10, fill="#EFF8F2")
    _draw_circle_outline(c, cx=cx, cy=cy, r=10, tone=tone, stroke_width=2.2)
    c.line(cx - 4, cy + 1, cx - 1, cy + 4, color=tone, width=2.2)
    c.line(cx - 1, cy + 4, cx + 5, cy - 3, color=tone, width=2.2)
    if label:
        c.text(cx, cy + 23, label, color=tone, size=10, bold=True)


def _draw_evidence_chip_bundle(
    c: DeckSvg,
    *,
    x: float,
    y: float,
    chips: list[tuple[str, str]],
) -> None:
    glyph_map = {
        "metrics": "m",
        "logs": "l",
        "traces": "t",
    }
    tone_map = {
        "metrics": "primary",
        "logs": "accent",
        "traces": "positive",
    }
    for idx, (kind, label) in enumerate(chips):
        chip_x = x + idx * 70
        tone = tone_map.get(kind, "secondary")
        c.rect(chip_x, y, 60, 24, fill="bg_alt", stroke=tone, stroke_width=1.8, rx=12)
        c.circle(chip_x + 12, y + 12, 7, fill=tone, opacity=0.88)
        c.text(chip_x + 12, y + 15, glyph_map.get(kind, "?").upper(), color="white", size=8, bold=True)
        c.text(chip_x + 35, y + 15, label, color="text", size=9, bold=True)


def _draw_circle_outline(
    c: DeckSvg,
    *,
    cx: float,
    cy: float,
    r: float,
    tone: str,
    stroke_width: float = 2.0,
) -> None:
    c.path(
        (
            f"M {cx - r} {cy} "
            f"A {r} {r} 0 1 0 {cx + r} {cy} "
            f"A {r} {r} 0 1 0 {cx - r} {cy}"
        ),
        stroke=tone,
        fill="none",
        stroke_width=stroke_width,
    )


def icon_metrics(size: int = 80) -> DeckSvg:
    c = DeckSvg(size, size, theme=THEME, bg=None)
    c.rect(8, 10, 64, 54, fill="bg_alt", stroke="border", rx=6)
    c.line(16, 54, 28, 40, color="primary", width=4)
    c.line(28, 40, 40, 46, color="primary", width=4)
    c.line(40, 46, 54, 28, color="primary", width=4)
    c.line(54, 28, 66, 34, color="primary", width=4)
    c.circle(28, 40, 3, fill="primary")
    c.circle(54, 28, 3, fill="primary")
    return c


def icon_logs(size: int = 80) -> DeckSvg:
    c = DeckSvg(size, size, theme=THEME, bg=None)
    c.rect(16, 10, 48, 60, fill="bg_alt", stroke="border", rx=5)
    c.line(24, 24, 56, 24, color="accent", width=4)
    c.line(24, 34, 50, 34, color="text_mid", width=3)
    c.line(24, 44, 54, 44, color="text_mid", width=3)
    c.line(24, 54, 44, 54, color="text_mid", width=3)
    return c


def icon_traces(size: int = 80) -> DeckSvg:
    c = DeckSvg(size, size, theme=THEME, bg=None)
    c.circle(18, 24, 8, fill="positive", opacity=0.85)
    c.circle(40, 24, 8, fill="positive", opacity=0.85)
    c.circle(62, 24, 8, fill="positive", opacity=0.85)
    c.circle(40, 54, 8, fill="positive", opacity=0.85)
    c.line(26, 24, 32, 24, color="text_mid", width=3)
    c.line(48, 24, 54, 24, color="text_mid", width=3)
    c.line(40, 32, 40, 46, color="text_mid", width=3)
    return c


def icon_service(size: int = 80) -> DeckSvg:
    c = DeckSvg(size, size, theme=THEME, bg=None)
    _draw_service_module(
        c,
        x=10,
        y=12,
        w=60,
        h=54,
        title="Svc",
        subtitle="healthy",
        status="normal",
    )
    return c


def icon_warning(size: int = 80) -> DeckSvg:
    c = DeckSvg(size, size, theme=THEME, bg=None)
    _draw_anomaly_burst(c, cx=40, cy=40, tone="warning", scale=0.9)
    c.text(40, 69, "ANOM", color="warning", size=9, bold=True)
    return c


def icon_root_cause(size: int = 80) -> DeckSvg:
    c = DeckSvg(size, size, theme=THEME, bg=None)
    _draw_root_cause_mark(c, cx=40, cy=36, tone="negative", scale=0.95)
    c.text(40, 69, "ROOT", color="negative", size=9, bold=True)
    return c


def icon_magnifier(size: int = 80) -> DeckSvg:
    c = DeckSvg(size, size, theme=THEME, bg=None)
    c.path("M 16 34 A 18 18 0 1 0 52 34 A 18 18 0 1 0 16 34",
           stroke="primary", fill="none", stroke_width=5)
    c.line(47, 47, 62, 62, color="primary", width=6)
    return c


def icon_path_chain(size: int = 80) -> DeckSvg:
    c = DeckSvg(size, size, theme=THEME, bg=None)
    c.circle(18, 50, 7, fill="highlight")
    c.circle(40, 28, 7, fill="highlight")
    c.circle(62, 48, 7, fill="highlight")
    c.path("M 18 50 Q 28 40 40 28 Q 52 36 62 48",
           stroke="highlight", fill="none", stroke_width=4)
    return c


def svg_gate_node(letter: str, label: str, tone: str, *, width: float = 2.0, height: float = 1.6) -> SvgImage:
    W, H = 220, 170
    c = DeckSvg(W, H, theme=THEME, bg=None)
    cx, cy = W / 2, 76
    c.circle(cx, cy, 44, fill="bg")
    _draw_circle_outline(c, cx=cx, cy=cy, r=44, tone=tone, stroke_width=3)
    c.text(cx, cy - 8, letter, color=tone, size=24, bold=True)
    c.text(cx, cy + 16, label, color=tone, size=11, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_contract_card(title: str, body: str, tone: str, *, width: float = 3.0, height: float = 1.1) -> SvgImage:
    W, H = 320, 112
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.rect(6, 16, 308, 88, fill="bg_alt", stroke=tone, rx=7, stroke_width=1.8)
    c.rect(6, 16, 92, 24, fill=tone, stroke="none", rx=3)
    c.text(52, 32, title, color="#FFFFFF", size=10, bold=True)
    c.text(160, 68, body, color="text", size=12, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_stage_card(title: str, body: str, tone: str, *, emphasized: bool = False, width: float = 3.0, height: float = 1.25) -> SvgImage:
    W, H = 320, 126
    fill = "bg_accent" if emphasized else "bg_alt"
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.rect(10, 10, 300, 108, fill=fill, stroke=tone, rx=7, stroke_width=1.8)
    c.rect(10, 10, 300, 22, fill=tone, stroke="none", rx=3)
    c.text(160, 25, title, color="#FFFFFF", size=10, bold=True)
    c.text(160, 66, body, color="text", size=12, bold=True)
    c.text(160, 88, "stage", color="text_mid", size=10)
    return SvgImage(svg=c, width=width, height=height)


def svg_probe_chevron(title: str, body: str, tone: str, *, width: float = 3.0, height: float = 1.1) -> SvgImage:
    W, H = 320, 112
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.polygon(
        [(10, 20), (254, 20), (284, 56), (254, 92), (10, 92), (40, 56)],
        fill="bg_alt",
        stroke=tone,
        stroke_width=1.8,
    )
    c.text(146, 45, title, color=tone, size=14, bold=True)
    c.text(146, 69, body, color="text_mid", size=11)
    return SvgImage(svg=c, width=width, height=height)


def svg_metric_stat_card(value: str, label: str, tone: str, *, width: float = 2.1, height: float = 1.6) -> SvgImage:
    W, H = 232, 172
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.rect(10, 14, 212, 144, fill="bg_alt", stroke="border", rx=7, stroke_width=1.4)
    c.text(116, 74, value, color=tone, size=31, bold=True)
    c.text(116, 100, label.upper(), color="text_mid", size=10, bold=True)
    c.rect(64, 114, 104, 22, fill=tone, stroke="none", rx=3)
    c.text(116, 129, "regime signal", color="#FFFFFF", size=9, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_validation_hexagon(*, width: float = 2.6, height: float = 1.7) -> SvgImage:
    W, H = 300, 180
    c = DeckSvg(W, H, theme=THEME, bg=None)
    gate = [(48, 18), (170, 18), (224, 90), (170, 162), (48, 162), (102, 90)]
    c.polygon(gate, fill="#FFF1DE", stroke="warning", stroke_width=2.0)
    c.text(136, 74, "Validation", color="warning", size=16, bold=True)
    c.text(136, 94, "impact > theta", color="warning", size=13, bold=True)
    c.text(136, 113, "retain only meaningful faults", color="text_mid", size=11)
    return SvgImage(svg=c, width=width, height=height)


def svg_validated_case_hexagon(*, width: float = 2.9, height: float = 1.8) -> SvgImage:
    W, H = 300, 180
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.polygon(
        [(26, 36), (224, 36), (244, 90), (224, 144), (26, 144), (6, 90)],
        fill="#EAF3FC",
        stroke="primary",
        stroke_width=2,
    )
    c.text(125, 72, "Validated benchmark case", color="primary", size=15, bold=True)
    c.text(125, 91, "fault -> user-visible impact", color="text_mid", size=11)
    c.text(125, 110, "evaluation-worthy incident", color="text_mid", size=11)
    return SvgImage(svg=c, width=width, height=height)


# ---------------------------------------------------------------------------
# Chart functions — DeckSvg-based, high-DPI quality
# ---------------------------------------------------------------------------

def _fmt_v(v: float) -> str:
    """Format a numeric value for bar labels."""
    if abs(v - round(v)) < 0.005:
        return str(int(round(v)))
    return f"{v:.2f}"


def bar_chart_svg(
    items: list[tuple[str, float, str]],
    *,
    ymax: float | None = None,
    width: float = 5.5,
    height: float = 2.8,
    threshold: float | None = None,
    threshold_label: str | None = None,
) -> SvgImage:
    """Single-series bar chart. items: [(label, value, color_name), ...]"""
    W, H = 800, 380
    ml, mr, mt, mb = 76, 32, 40, 64
    cw = W - ml - mr
    ch = H - mt - mb
    max_v = ymax if ymax is not None else max((v for _, v, _ in items), default=1.0)

    c = DeckSvg(W, H, theme=THEME, bg="bg")

    # X-axis
    c.line(ml, mt + ch, W - mr, mt + ch, color="text_mid", width=2)
    # Gridlines
    for pct in [0.25, 0.5, 0.75]:
        y_g = mt + ch * (1 - pct)
        c.line(ml, y_g, W - mr, y_g, color="border", width=1, dashed=True)
        c.text(ml - 8, y_g, f"{max_v * pct:.2g}", color="text_mid", size=14, anchor="end")
    # Top value label
    c.text(ml - 8, mt, f"{max_v:.2g}", color="text_mid", size=14, anchor="end")

    n = len(items)
    gap = 22
    bar_w = max(38, (cw - gap * (n + 1)) / max(n, 1))

    for idx, (label, value, tone) in enumerate(items):
        x = ml + gap + idx * (bar_w + gap)
        bar_h = 0.0 if max_v <= 0 else ch * (value / max_v)
        y = mt + ch - bar_h
        c.rounded_rect(x, y, bar_w, bar_h, text="", fill=tone, stroke="none", rx=8)
        val_y = max(y - 12, mt + 18)
        c.text(x + bar_w / 2, val_y, _fmt_v(value), color="text", size=21, bold=True)
        c.text(x + bar_w / 2, H - 18, label, color="text_mid", size=15)

    if threshold is not None:
        y_thresh = mt + ch * (1 - threshold / max_v)
        c.line(ml, y_thresh, W - mr, y_thresh, color="positive", width=2, dashed=True)
        if threshold_label:
            c.text(W - mr - 4, y_thresh - 8, threshold_label,
                   color="positive", size=12, anchor="end")

    return SvgImage(svg=c, width=width, height=height)


def grouped_bar_svg(
    group_labels: list[str],
    series_labels: list[str],
    values: list[list[float]],
    tones: list[str],
    *,
    ymax: float = 1.05,
    width: float = 6.4,
    height: float = 3.1,
) -> SvgImage:
    """Grouped bar chart with legend."""
    W, H = 900, 380
    ml, mr, mt, mb = 76, 30, 54, 64   # mt=54 for legend
    cw = W - ml - mr
    ch = H - mt - mb

    c = DeckSvg(W, H, theme=THEME, bg="bg")

    # Legend
    legend_x = ml
    for s_idx, (series, tone) in enumerate(zip(series_labels, tones)):
        lx = legend_x + s_idx * 160
        c.rect(lx, 14, 16, 14, fill=tone, stroke="none", rx=3)
        c.text(lx + 22, 21, series, color="text_mid", size=14, anchor="start")

    # X-axis
    c.line(ml, mt + ch, W - mr, mt + ch, color="text_mid", width=2)
    # Gridlines
    for pct in [0.25, 0.5, 0.75]:
        y_g = mt + ch * (1 - pct)
        c.line(ml, y_g, W - mr, y_g, color="border", width=1, dashed=True)
        c.text(ml - 8, y_g, f"{ymax * pct:.2g}", color="text_mid", size=13, anchor="end")
    c.text(ml - 8, mt, f"{ymax:.2g}", color="text_mid", size=13, anchor="end")

    n_groups = len(group_labels)
    n_series = len(series_labels)
    group_w = cw / max(n_groups, 1)
    bar_w = min(36, (group_w - 28) / max(n_series, 1))

    for g_idx, group in enumerate(group_labels):
        gx = ml + g_idx * group_w + 14
        for s_idx in range(n_series):
            value = values[g_idx][s_idx]
            tone = tones[s_idx % len(tones)]
            x = gx + s_idx * (bar_w + 7)
            bar_h = ch * (value / ymax)
            y = mt + ch - bar_h
            c.rounded_rect(x, y, bar_w, bar_h, text="", fill=tone, stroke="none", rx=6)
            c.text(x + bar_w / 2, max(y - 10, mt + 16), _fmt_v(value),
                   color="text", size=15, bold=True)
        center = gx + (n_series * (bar_w + 7) - 7) / 2
        c.text(center, H - 18, group, color="text_mid", size=14)

    return SvgImage(svg=c, width=width, height=height)


# ---------------------------------------------------------------------------
# Diagram functions — DeckSvg-based rich visuals
# ---------------------------------------------------------------------------

def svg_synthesis_ladder(
    *,
    sub_labels: list[str] | None = None,
    evidence: list[str] | None = None,
    width: float = 10.0,
    height: float = 2.0,
) -> SvgImage:
    """A/B/C ladder with gate semantics and verdict-ready annotation lanes."""
    has_extra = bool(sub_labels or evidence)
    W = 1120
    H = 380 if (sub_labels and evidence) else 320 if has_extra else 250
    cy_node = 114

    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()

    centers_x = [186, 560, 934]
    colors = ["primary", "accent", "positive"]
    letters = ["A", "B", "C"]
    node_labels = ["Realism", "Capability", "Trust"]
    r = 60

    c.line(120, 52, 1000, 52, color="border", width=2)
    c.text(560, 44, "Evaluation contract: A gates B, B gates C", color="text_mid", size=12, bold=True)

    for cx_val, col, letter, lbl in zip(centers_x, colors, letters, node_labels):
        _draw_circle_outline(c, cx=cx_val, cy=cy_node, r=r + 8, tone=col, stroke_width=2.2)
        c.circle(cx_val, cy_node, r, fill=col, opacity=0.92)
        c.text(cx_val, cy_node - 14, letter, color="#FFFFFF", size=29, bold=True)
        c.text(cx_val, cy_node + 18, lbl, color="#FFFFFF", size=15, bold=True)

    c.arrow(252, cy_node, 494, cy_node, color="accent", width=4)
    c.arrow(626, cy_node, 868, cy_node, color="positive", width=4)

    c.text(372, cy_node - 20, "gate", color="accent", size=11, bold=True)
    c.text(746, cy_node - 20, "gate", color="positive", size=11, bold=True)

    if sub_labels:
        for cx_val, text in zip(centers_x, sub_labels):
            c.text(cx_val, cy_node + r + 34, text, color="text_mid", size=12)

    if evidence:
        evidence_y = cy_node + r + (72 if sub_labels else 38)
        for cx_val, text, tone in zip(centers_x, evidence, colors):
            x = cx_val - 90
            c.rect(x + 14, evidence_y - 18, 176, 36, fill="bg_alt", stroke="border", rx=4, stroke_width=1.4)
            c.polygon(
                [(x, evidence_y), (x + 14, evidence_y - 18), (x + 14, evidence_y + 18)],
                fill=tone,
                stroke=tone,
                stroke_width=1.0,
            )
            c.text(cx_val + 8, evidence_y + 4, text, color="text", size=11, bold=True)

    return SvgImage(svg=c, width=width, height=height)


def svg_hardness_map(*, width: float = 10.1, height: float = 3.15) -> SvgImage:
    W, H = 1080, 350
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()

    # Modality deck on the left
    c.rect(34, 46, 284, 250, fill="bg_alt", stroke="border", rx=8, stroke_width=1.6)
    c.text(176, 72, "Telemetry modalities", color="text", size=15, bold=True)
    _evidence_chip(c, 60, 102, "metrics: onset + magnitude", "primary", "M")
    _evidence_chip(c, 60, 146, "logs: local semantic clues", "accent", "L")
    _evidence_chip(c, 60, 190, "traces: cross-service hops", "positive", "T")
    c.rect(60, 236, 232, 34, fill="bg", stroke="primary", rx=5, stroke_width=1.2)
    c.text(176, 257, "three views, one diagnosis", color="primary", size=11, bold=True)

    # Causal path center
    _draw_service_module(c, x=388, y=112, w=154, h=86, title="Root cause", subtitle="db lock", status="root")
    _draw_service_module(c, x=634, y=88, w=154, h=86, title="Mid service", subtitle="retry storm", status="propagated")
    _draw_service_module(c, x=874, y=130, w=154, h=86, title="User symptom", subtitle="timeout", status="propagated")
    c.arrow(542, 154, 632, 132, color="warning", width=2.8)
    c.arrow(788, 132, 872, 166, color="negative", width=2.8)
    c.path("M 540 186 C 640 250, 826 250, 906 212", stroke="primary", fill="none", stroke_width=2, dashed=True)
    c.text(730, 272, "must preserve propagation order", color="primary", size=11, bold=True)

    # Burden bars on top-right
    c.rect(384, 36, 192, 30, fill="#EEF4FA", stroke="primary", rx=4, stroke_width=1.2)
    c.text(480, 56, "Fuse modalities", color="primary", size=11, bold=True)
    c.rect(602, 36, 192, 30, fill="#FFF5EA", stroke="accent", rx=4, stroke_width=1.2)
    c.text(698, 56, "Trace hops", color="accent", size=11, bold=True)
    c.rect(820, 36, 216, 30, fill="#EFF8F2", stroke="positive", rx=4, stroke_width=1.2)
    c.text(928, 56, "Order events causally", color="positive", size=11, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_shortcut_contrast_map(*, width: float = 10.1, height: float = 3.2) -> SvgImage:
    W, H = 1080, 360
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()

    c.rect(24, 26, 500, 294, fill="#FFF2F2", stroke="negative", rx=10, stroke_width=1.8)
    c.rect(556, 26, 500, 294, fill="#EEF5FC", stroke="primary", rx=10, stroke_width=1.8)
    c.text(274, 56, "Shortcut ranking path", color="negative", size=16, bold=True)
    c.text(806, 56, "Causal verification path", color="primary", size=16, bold=True)

    # left fast path
    c.polygon([(70, 114), (238, 114), (266, 144), (238, 174), (70, 174), (42, 144)], fill="#F3B4B4", stroke="negative", stroke_width=1.6)
    c.text(154, 149, "largest anomaly", color="negative", size=13, bold=True)
    c.polygon([(306, 114), (468, 114), (496, 144), (468, 174), (306, 174), (278, 144)], fill="#F6D6A8", stroke="warning", stroke_width=1.6)
    c.text(387, 149, "rank service", color="warning", size=13, bold=True)
    c.arrow(266, 144, 278, 144, color="negative", width=2.4)
    c.rect(126, 232, 296, 44, fill="negative", stroke="none", rx=6)
    c.text(274, 259, "high score possible, weak causal validity", color="#FFFFFF", size=12, bold=True)

    # right audited path
    _evidence_chip(c, 592, 96, "collect metrics/logs/traces", "secondary", "E")
    _evidence_chip(c, 592, 132, "align across dependency graph", "accent", "A")
    _evidence_chip(c, 592, 168, "verify propagation chain", "primary", "V")
    c.arrow(778, 110, 936, 110, color="secondary", width=2.2)
    c.arrow(778, 146, 936, 146, color="accent", width=2.2)
    c.arrow(778, 182, 936, 182, color="primary", width=2.2)
    c.rect(864, 92, 164, 108, fill="bg", stroke="primary", rx=8, stroke_width=1.8)
    c.text(946, 124, "defensible", color="primary", size=13, bold=True)
    c.text(946, 146, "trigger + path", color="primary", size=13, bold=True)
    c.rect(634, 232, 352, 44, fill="primary", stroke="none", rx=6)
    c.text(810, 259, "lower shortcut risk, stronger evaluation signal", color="#FFFFFF", size=12, bold=True)

    c.rect(264, 330, 552, 24, fill="warning", stroke="none", rx=4)
    c.text(540, 346, "Mismatch: hard task, easy benchmark -> inflated progress claims", color="#FFFFFF", size=11, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_question_gate_map(*, width: float = 10.1, height: float = 3.0) -> SvgImage:
    W, H = 1080, 330
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()

    # gate spine
    c.line(84, 138, 996, 138, color="border", width=3)
    gates = [
        (180, "A", "Realism", "primary"),
        (540, "B", "Capability", "accent"),
        (900, "C", "Trust", "positive"),
    ]
    for cx, letter, label, tone in gates:
        c.circle(cx, 138, 44, fill="bg")
        _draw_circle_outline(c, cx=cx, cy=138, r=44, tone=tone, stroke_width=3)
        c.text(cx, 130, letter, color=tone, size=24, bold=True)
        c.text(cx, 154, label, color=tone, size=11, bold=True)

    c.arrow(224, 138, 496, 138, color="accent", width=3.2)
    c.arrow(584, 138, 856, 138, color="positive", width=3.2)
    c.text(360, 121, "A gates B", color="accent", size=11, bold=True)
    c.text(720, 121, "B gates C", color="positive", size=11, bold=True)

    # contract panels
    contracts = [
        (56, "Q-A contract", "Benchmark must be operationally hard.", "primary"),
        (386, "Q-B contract", "Measure LLMs on realistic telemetry.", "accent"),
        (716, "Q-C contract", "Audit causal path, not label only.", "positive"),
    ]
    for x, title, body, tone in contracts:
        c.rect(x, 200, 308, 88, fill="bg_alt", stroke=tone, rx=7, stroke_width=1.8)
        c.rect(x, 200, 92, 24, fill=tone, stroke="none", rx=3)
        c.text(x + 46, 216, title, color="#FFFFFF", size=10, bold=True)
        c.text(x + 156, 252, body, color="text", size=12, bold=True)

    return SvgImage(svg=c, width=width, height=height)


def svg_listening_guide_map(*, width: float = 10.1, height: float = 2.9) -> SvgImage:
    W, H = 1080, 320
    c = DeckSvg(W, H, theme=THEME, bg=None)

    c.rect(34, 26, 1012, 56, fill="primary", stroke="none", rx=6)
    c.text(540, 60, "Decision rule: trust progress only when A + B + C all hold", color="#FFFFFF", size=16, bold=True)

    anchors = [
        (58, "A", "Real task", "Look for propagation-aware failures.", "primary"),
        (390, "B", "Real capability", "Look for realistic-task performance gaps.", "accent"),
        (722, "C", "Real trust", "Look for outcome-vs-process gaps.", "positive"),
    ]
    for x, letter, title, body, tone in anchors:
        c.rect(x, 116, 300, 154, fill="bg_alt", stroke=tone, rx=8, stroke_width=1.8)
        c.circle(x + 32, 150, 16, fill=tone)
        c.text(x + 32, 156, letter, color="#FFFFFF", size=12, bold=True)
        c.text(x + 60, 146, title, color=tone, size=14, bold=True, anchor="start")
        c.text(x + 60, 176, body, color="text_mid", size=11, anchor="start")
        c.line(x + 24, 204, x + 280, 204, color="border", width=1.2, dashed=True)
        c.text(x + 152, 230, f"watchpoint {letter}", color=tone, size=11, bold=True)

    return SvgImage(svg=c, width=width, height=height)


def svg_probe_logic_map(*, width: float = 10.1, height: float = 3.0) -> SvgImage:
    W, H = 1080, 330
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()

    c.rect(40, 32, 1000, 74, fill="#FFF4F4", stroke="negative", rx=8, stroke_width=1.8)
    c.text(540, 60, "Probe claim: if a simple transparent heuristic reaches SOTA, benchmark discrimination is weak", color="negative", size=13, bold=True)
    c.text(540, 84, "SimpleRCA is a diagnostic instrument, not a proposed production method", color="text_mid", size=11)

    chevrons = [
        (70, "Hypothesis", "legacy benchmark is discriminative", "secondary"),
        (360, "Probe", "run SimpleRCA without ML tuning", "warning"),
        (650, "Decision", "compare against reported SOTA", "primary"),
    ]
    for x, title, body, tone in chevrons:
        c.polygon(
            [(x, 146), (x + 244, 146), (x + 274, 182), (x + 244, 218), (x, 218), (x + 30, 182)],
            fill="bg_alt",
            stroke=tone,
            stroke_width=1.8,
        )
        c.text(x + 130, 171, title, color=tone, size=14, bold=True)
        c.text(x + 130, 194, body, color="text_mid", size=11)
    c.arrow(344, 182, 358, 182, color="warning", width=2.2)
    c.arrow(634, 182, 648, 182, color="primary", width=2.2)

    c.rect(68, 250, 456, 54, fill="#FFF9EF", stroke="accent", rx=6, stroke_width=1.2)
    c.text(296, 274, "Fairness: transparent + interpretable + low-capacity baseline", color="accent", size=12, bold=True)
    c.rect(560, 250, 452, 54, fill="#EEF4FA", stroke="primary", rx=6, stroke_width=1.2)
    c.text(786, 274, "Falsification: if SimpleRCA clearly loses, benchmark still separates depth", color="primary", size=12, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_cascade_diagram(*, width: float = 4.8, height: float = 3.25) -> SvgImage:
    """Compact symbolic RCA panel for motivation slides."""
    W, H = 560, 390
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers(colors=["negative", "warning", "primary", "text_mid"])

    def symptom_chip(x: float, y: float) -> None:
        c.rect(x, y, 176, 54, fill="#FFF0F0", stroke="negative", stroke_width=1.8, rx=18)
        c.rect(x + 12, y + 12, 58, 18, fill="negative", stroke="none", rx=9, opacity=0.95)
        c.text(x + 41, y + 24, "SYMPTOM", color="white", size=9, bold=True)
        c.text(x + 88, y + 22, "Checkout SLI", color="text", size=13, bold=True, anchor="start")
        c.text(x + 88, y + 39, "degraded", color="negative", size=11, anchor="start")

    def timer_chip(x: float, y: float) -> None:
        c.rect(x, y, 108, 54, fill="#FFF6EA", stroke="warning", stroke_width=1.6, rx=18)
        c.circle(x + 27, y + 27, 11, fill="none", text="", opacity=1.0)
        c.path(
            (
                f"M {x + 16} {y + 27} "
                f"A 11 11 0 1 0 {x + 38} {y + 27} "
                f"A 11 11 0 1 0 {x + 16} {y + 27}"
            ),
            stroke="warning",
            fill="none",
            stroke_width=2,
        )
        c.line(x + 27, y + 27, x + 27, y + 20, color="warning", width=2)
        c.line(x + 27, y + 27, x + 32, y + 31, color="warning", width=2)
        c.text(x + 50, y + 23, "MTTR", color="warning", size=11, bold=True, anchor="start")
        c.text(x + 50, y + 39, "pressure", color="text_mid", size=10, anchor="start")

    symptom_chip(170, 28)
    timer_chip(422, 164)

    _draw_service_module(
        c,
        x=76,
        y=140,
        w=180,
        h=72,
        title="Observed service",
        subtitle="loud anomaly near the user",
        status="propagated",
        badge="seen first",
    )
    _draw_service_module(
        c,
        x=186,
        y=268,
        w=206,
        h=78,
        title="Root-cause service",
        subtitle="actual trigger starts the cascade",
        status="root",
        badge="trigger",
    )

    c.text(166, 124, "What on-call sees first", color="text_mid", size=11)
    c.text(289, 365, "Real RCA isolates the trigger, not the loudest symptom.", color="primary", size=12, bold=True)

    c.arrow(258, 82, 200, 140, color="negative", width=2.6)
    c.text(226, 98, "symptom surfaces here", color="text_mid", size=10, anchor="start")

    c.arrow(242, 212, 270, 268, color="warning", width=2.6)
    c.circle(252, 232, 4, fill="warning", opacity=0.95)
    c.text(276, 238, "trace back to cause", color="text_mid", size=10, anchor="start")

    c.path(
        "M 302 306 C 350 278, 387 248, 430 193",
        stroke="negative",
        fill="none",
        stroke_width=2.2,
        dashed=True,
    )
    c.text(340, 276, "minutes matter", color="warning", size=10, anchor="start")

    c.path(
        "M 168 240 C 132 258, 128 286, 166 309",
        stroke="border",
        fill="none",
        stroke_width=1.6,
        dashed=True,
    )
    c.text(52, 288, "highest anomaly", color="text_mid", size=11, anchor="start")
    c.text(52, 303, "!= root cause", color="text_mid", size=11, anchor="start")

    return SvgImage(svg=c, width=width, height=height)


def svg_forge_pipeline(*, width: float = 9.5, height: float = 2.5) -> SvgImage:
    """FORGE forward verification pipeline: Intervention → Check → Validated Path."""
    W, H = 1050, 300
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()

    bw, bh = 270, 100
    stages = [
        (30,  90, "Known Intervention",  "injected fault origin",      "secondary"),
        (390, 90, "Forward Verification", "cause \u2192 effect check \u2713", "accent"),
        (750, 90, "Validated Causal Path", "annotated propagation",      "primary"),
    ]

    for (x, y, label, sub, fill) in stages:
        cx_s = x + bw // 2
        cy_s = y + bh // 2
        tc = "white"
        c.rect(x, y, bw, bh, fill=fill, stroke="none", rx=12)
        c.text(cx_s, cy_s - 16, label, color=tc, size=16, bold=True)
        c.text(cx_s, cy_s + 14, sub,   color=tc, size=13)

    # Arrows between stages
    c.arrow(300, 140, 390, 140, color="text_mid", width=3)
    c.arrow(660, 140, 750, 140, color="text_mid", width=3)

    # Footer note
    c.text(W // 2, 248,
           "Backward inference is hard  \u2192  FORGE turns it into tractable forward verification",
           color="text_mid", size=13, anchor="middle")

    return SvgImage(svg=c, width=width, height=height)


def svg_benchmark_pipeline(*, width: float = 9.5, height: float = 3.8) -> SvgImage:
    """6-stage snake pipeline: Foundation\u2192\u2192Injection \u2192 Collection\u2190\u2190Validation(highlighted)."""
    W, H = 1000, 420
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()

    bw, bh = 220, 80
    # (x_left, y_center, label, subtitle, fill)
    stages = [
        (50,  120, "Foundation",   "TrainTicket + observability",    "bg_alt"),
        (390, 120, "Workload",     "Dynamic execution paths",        "bg_alt"),
        (730, 120, "Injection",    "31 fault types, layered",        "bg_alt"),
        (730, 310, "Collection",   "Metrics + logs + traces",        "bg_alt"),
        (390, 310, "Annotation",   "Hierarchical root-cause labels", "bg_alt"),
        (50,  310, "Validation \u2713", "User-facing impact filter", "primary"),
    ]

    for (x, cy, label, sub, fill) in stages:
        cx_s = x + bw // 2
        tc = "white" if fill == "primary" else "text"
        stc = "white" if fill == "primary" else "text_mid"
        border = fill if fill == "primary" else "border"
        c.rounded_rect(x, cy - bh // 2, bw, bh, text="", fill=fill,
                       stroke=border, rx=10, stroke_width=2)
        c.text(cx_s, cy - 14, label, color=tc, size=15, bold=(fill == "primary"))
        c.text(cx_s, cy + 12, sub, color=stc, size=12)

    # Row 1 right arrows
    c.arrow(270, 120, 390, 120, color="text_mid", width=2)  # Foundation → Workload
    c.arrow(610, 120, 730, 120, color="text_mid", width=2)  # Workload → Injection
    # Down arrow (Injection → Collection)
    c.arrow(840, 160, 840, 270, color="text_mid", width=2)
    # Row 2 left arrows
    c.arrow(730, 310, 610, 310, color="text_mid", width=2)  # Collection → Annotation
    c.arrow(390, 310, 270, 310, color="primary",  width=2)  # Annotation → Validation

    return SvgImage(svg=c, width=width, height=height)


def svg_process_comparison(*, width: float = 9.5, height: float = 3.0) -> SvgImage:
    """Side-by-side: Outcome-only (left) vs Process-aware (right)."""
    W, H = 1050, 340
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()

    # ---- Left: Outcome-only ----
    c.rounded_rect(30, 16, 455, 44, text="Outcome-only",
                   fill="bg_alt", stroke="warning", text_color="warning",
                   font_size=16, bold=True, rx=8)
    # Single final-answer box
    c.rect(145, 118, 200, 70, fill="bg_alt", stroke="warning", rx=10, stroke_width=2)
    c.text(245, 148, "Root Cause", color="text", size=14, bold=True)
    c.text(245, 168, "Identified  \u2713", color="positive", size=13)
    c.text(245, 240, "Only final answer verified", color="warning", size=13)
    c.text(245, 260, "Intermediate path: unknown", color="text_mid", size=12)

    # ---- Divider ----
    c.line(525, 24, 525, 310, color="border", width=1, dashed=True)

    # ---- Right: Process-aware ----
    c.rounded_rect(560, 16, 455, 44, text="Process-aware",
                   fill="bg_accent", stroke="primary", text_color="primary",
                   font_size=16, bold=True, rx=8)

    # Causal chain
    bw_n, bh_n = 90, 58
    chain = [(595, 155), (715, 155), (835, 155), (955, 155)]
    chain_labels = ["Root\nCause", "Prop.\nStep 1", "Prop.\nStep 2", "Observed\nSymptom"]
    chain_colors = ["primary", "accent", "accent", "secondary"]
    for (cx_n, cy_n), lbl, col in zip(chain, chain_labels, chain_colors):
        c.rect(cx_n - bw_n // 2, cy_n - bh_n // 2, bw_n, bh_n,
               fill="bg_accent", stroke=col, rx=8, stroke_width=2)
        c.text(cx_n, cy_n - 8, lbl, color=col, size=12, bold=True)
        c.text(cx_n, cy_n + 22, "\u2713", color="positive", size=18, bold=True)

    for i in range(len(chain) - 1):
        x1 = chain[i][0] + bw_n // 2
        x2 = chain[i + 1][0] - bw_n // 2
        c.arrow(x1, 155, x2, 155, color="primary", width=2)

    c.text(775, 240, "Each propagation step verified", color="primary", size=13)
    c.text(775, 260, "Causal path is fully auditable", color="text_mid", size=12)

    return SvgImage(svg=c, width=width, height=height)


def svg_metric_semantics(*, width: float = 8.0, height: float = 3.2) -> SvgImage:
    """Concentric-circle diagram showing PR \u2286 Pass@1."""
    W, H = 900, 360
    c = DeckSvg(W, H, theme=THEME, bg=None)

    cx, cy = 240, 178
    r_out, r_in = 140, 86

    # Outer ring (Pass@1 region)
    c.circle(cx, cy, r_out, fill="secondary", opacity=0.22)
    # Inner region (PR)
    c.circle(cx, cy, r_in, fill="primary", opacity=0.70)

    # Labels
    c.text(cx, cy - 18, "PR", color="white", size=22, bold=True)
    c.text(cx, cy + 12, "= 0.63", color="white", size=15)

    # Outer ring label (below)
    c.text(cx, cy + r_out + 22, "Pass@1 = 0.76  (all correct answers)",
           color="secondary", size=14, bold=True)

    # Gap annotation
    mid_r = (r_out + r_in) / 2
    c.text(cx + int(mid_r) + 10, cy - 18, "\u223c13%", color="text_mid", size=13,
           anchor="start", bold=True)
    c.text(cx + int(mid_r) + 10, cy + 2, "gap", color="text_mid", size=12, anchor="start")

    # Right side: explanation boxes
    box_x = 430
    c.rounded_rect(box_x, 50, 440, 90, text="", fill="secondary",
                   stroke="secondary", rx=10, opacity=0.85)
    c.text(box_x + 220, 78, "Pass@1", color="white", size=17, bold=True)
    c.text(box_x + 220, 100, "Did the model name the correct root cause?",
           color="white", size=13)
    c.text(box_x + 220, 120, "(outcome correctness only)",
           color="white", size=12, italic=True)

    c.rounded_rect(box_x, 170, 440, 90, text="", fill="primary",
                   stroke="primary", rx=10, opacity=0.85)
    c.text(box_x + 220, 198, "Path Reachability (PR)", color="white", size=17, bold=True)
    c.text(box_x + 220, 220, "Does the claimed path lead cause \u2192 symptom?",
           color="white", size=13)
    c.text(box_x + 220, 240, "(process faithfulness)", color="white", size=12, italic=True)

    # Inequality
    c.rounded_rect(box_x + 80, 290, 280, 44, text="PR \u2264 Pass@1  always",
                   fill="warning", stroke="none", text_color="white",
                   font_size=15, bold=True, rx=8)

    return SvgImage(svg=c, width=width, height=height)


def _evidence_chip(c: DeckSvg, x: float, y: float, label: str, tone: str, glyph: str) -> None:
    chip_w = max(92, len(label) * 7 + 34)
    c.rounded_rect(x, y, chip_w, 28, text="", fill="bg_alt", stroke=tone, rx=14, stroke_width=1.6)
    c.circle(x + 16, y + 14, 7, fill=tone, opacity=0.92)
    c.text(x + 16, y + 18, glyph, color="white", size=9, bold=True)
    c.text(x + 32, y + 18, label, color="text", size=12, bold=True, anchor="start")


def svg_evidence_chip(label: str, tone: str, glyph: str, *, width: float = 2.6, height: float = 0.55) -> SvgImage:
    chip_w = max(92, len(label) * 7 + 34)
    W, H = chip_w + 20, 48
    c = DeckSvg(W, H, theme=THEME, bg=None)
    _evidence_chip(c, 10, 10, label, tone, glyph)
    return SvgImage(svg=c, width=width, height=height)


def svg_service_card(
    *,
    title: str,
    subtitle: str,
    status: str = "normal",
    badge: str | None = None,
    style: str = "rich",
    width: float = 2.2,
    height: float = 1.25,
) -> SvgImage:
    W, H = 240, 136
    c = DeckSvg(W, H, theme=THEME, bg=None)
    _draw_service_module(c, x=16, y=22, w=208, h=92, title=title, subtitle=subtitle, status=status, badge=badge, style=style)
    return SvgImage(svg=c, width=width, height=height)


def _svg_markup(obj: DeckSvg | SvgImage) -> str:
    if isinstance(obj, SvgImage):
        obj = obj.svg
    return obj.render()


def _write_svg_asset(output_dir: Path, relative_name: str, obj: DeckSvg | SvgImage) -> str:
    path = output_dir / relative_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_svg_markup(obj), encoding="utf-8")
    return relative_name


def export_svg_assets(output_dir: Path | None = None) -> Path:
    output_dir = output_dir or SVG_ASSET_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[str, str]] = []

    def add(path: str, obj: DeckSvg | SvgImage, note: str) -> None:
        generated.append((_write_svg_asset(output_dir, path, obj), note))

    # Icons
    add("icons/metrics.svg", icon_metrics(), "Metrics icon.")
    add("icons/logs.svg", icon_logs(), "Logs icon.")
    add("icons/traces.svg", icon_traces(), "Traces icon.")
    add("icons/service.svg", icon_service(), "Healthy service icon.")
    add("icons/warning.svg", icon_warning(), "Anomaly burst icon.")
    add("icons/root_cause.svg", icon_root_cause(), "Root-cause marker icon.")
    add("icons/magnifier.svg", icon_magnifier(), "Magnifier icon.")
    add("icons/path_chain.svg", icon_path_chain(), "Causal path icon.")

    # Telemetry chips
    add("chips/metrics_onset_magnitude.svg", svg_evidence_chip("metrics: onset + magnitude", "primary", "M"), "Single telemetry chip.")
    add("chips/logs_local_semantics.svg", svg_evidence_chip("logs: local semantic clues", "accent", "L"), "Single telemetry chip.")
    add("chips/traces_cross_service.svg", svg_evidence_chip("traces: cross-service hops", "positive", "T"), "Single telemetry chip.")
    add("chips/collect_metrics_logs_traces.svg", svg_evidence_chip("collect metrics/logs/traces", "secondary", "E"), "Probe pipeline chip.")
    add("chips/align_dependency_graph.svg", svg_evidence_chip("align across dependency graph", "accent", "A"), "Probe pipeline chip.")
    add("chips/verify_propagation_chain.svg", svg_evidence_chip("verify propagation chain", "primary", "V"), "Probe pipeline chip.")

    # Gate nodes and contracts
    add("gates/gate_a_realism.svg", svg_gate_node("A", "Realism", "primary"), "Single gate node.")
    add("gates/gate_b_capability.svg", svg_gate_node("B", "Capability", "accent"), "Single gate node.")
    add("gates/gate_c_trust.svg", svg_gate_node("C", "Trust", "positive"), "Single gate node.")
    add("contracts/q_a_contract.svg", svg_contract_card("Q-A contract", "Benchmark must be operationally hard.", "primary"), "Single contract card.")
    add("contracts/q_b_contract.svg", svg_contract_card("Q-B contract", "Measure LLMs on realistic telemetry.", "accent"), "Single contract card.")
    add("contracts/q_c_contract.svg", svg_contract_card("Q-C contract", "Audit causal path, not label only.", "positive"), "Single contract card.")

    # Service cards
    add("services/root_cause_db_lock.svg", svg_service_card(title="Root cause", subtitle="db lock", status="root"), "Single service card.")
    add("services/mid_service_retry_storm.svg", svg_service_card(title="Mid service", subtitle="retry storm", status="propagated"), "Single service card.")
    add("services/user_symptom_timeout.svg", svg_service_card(title="User symptom", subtitle="timeout", status="propagated"), "Single service card.")
    add("services/payment_fault_surface.svg", svg_service_card(title="Payment", subtitle="fault surface loud", status="root"), "Single service card.")
    add("services/propagation_source_a.svg", svg_service_card(title="A", subtitle="source", status="root"), "Single service card.")
    add("services/propagation_symptom_b.svg", svg_service_card(title="B", subtitle="symptom", status="propagated"), "Single service card.")
    add("services/observed_service_seen_first.svg", svg_service_card(title="Observed service", subtitle="loud anomaly near the user", status="propagated", badge="seen first"), "Single service card.")
    add("services/root_service_trigger.svg", svg_service_card(title="Root-cause service", subtitle="actual trigger starts the cascade", status="root", badge="trigger"), "Single service card.")

    # Probe chevrons
    add("probe/hypothesis.svg", svg_probe_chevron("Hypothesis", "legacy benchmark is discriminative", "secondary"), "Single chevron card.")
    add("probe/simple_rca_probe.svg", svg_probe_chevron("Probe", "run SimpleRCA without ML tuning", "warning"), "Single chevron card.")
    add("probe/decision_compare_sota.svg", svg_probe_chevron("Decision", "compare against reported SOTA", "primary"), "Single chevron card.")

    # Workflow / stage cards
    add("stages/foundation.svg", svg_stage_card("1 Foundation", "TrainTicket + observability stack", "secondary"), "Single workflow card.")
    add("stages/workload.svg", svg_stage_card("2 Workload", "Dynamic traffic creates path variety", "accent"), "Single workflow card.")
    add("stages/injection.svg", svg_stage_card("3 Injection", "31 layered fault types", "warning"), "Single workflow card.")
    add("stages/collection.svg", svg_stage_card("4 Collection", "Metrics, logs, traces snapshot", "positive"), "Single workflow card.")
    add("stages/validation.svg", svg_stage_card("5 Validation", "Retain only impact-validated cases", "primary", emphasized=True), "Single workflow card.")
    add("stages/annotation.svg", svg_stage_card("6 Annotation", "Hierarchical RCA labels + paths", "secondary"), "Single workflow card.")

    # Validation flow building blocks
    add("validation/validation_gate.svg", svg_validation_hexagon(), "Single validation gate.")
    add("validation/validated_case.svg", svg_validated_case_hexagon(), "Single validated-case hexagon.")

    # Stats cards
    add("stats/validated_cases.svg", svg_metric_stat_card("1,430", "validated cases", "primary"), "Single stats card.")
    add("stats/fault_injections.svg", svg_metric_stat_card("9,152", "fault injections", "accent"), "Single stats card.")
    add("stats/fault_types.svg", svg_metric_stat_card("25", "fault types", "positive"), "Single stats card.")

    manifest = [
        "# Talk 4.15 SVG Assets",
        "",
        "These SVGs are intentionally split into single reusable elements so you can regroup them manually in PowerPoint.",
        "",
        "| Asset | Purpose |",
        "| --- | --- |",
    ]
    manifest.extend(f"| `{path}` | {note} |" for path, note in generated)
    (output_dir / "README.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    return output_dir


def svg_legacy_bias_triad(*, width: float = 10.0, height: float = 3.2) -> SvgImage:
    W, H = 1080, 360
    c = DeckSvg(W, H, theme=THEME, bg=None)

    # Column 1: notched panel
    c.polygon([(34, 58), (334, 58), (334, 288), (66, 288), (34, 258)], fill="#FFF3F3", stroke="negative", stroke_width=2)
    c.rect(54, 30, 134, 28, fill="negative", stroke="none", rx=4)
    c.text(121, 49, "Injection bias", color="#FFFFFF", size=12, bold=True)
    _draw_service_module(c, x=62, y=102, w=128, h=86, title="Payment", subtitle="fault surface loud", status="root")
    c.text(206, 136, "largest anomaly", color="negative", size=12, bold=True, anchor="start")
    c.text(206, 156, "shortcut picks source quickly", color="text_mid", size=11, anchor="start")

    # Column 2: path frame
    c.rect(382, 58, 300, 230, fill="#FFF8EE", stroke="warning", rx=8, stroke_width=2)
    c.rect(402, 30, 162, 28, fill="warning", stroke="none", rx=4)
    c.text(483, 49, "Shallow propagation", color="#FFFFFF", size=12, bold=True)
    _draw_service_module(c, x=424, y=114, w=96, h=70, title="A", subtitle="source", status="root")
    _draw_service_module(c, x=550, y=114, w=96, h=70, title="B", subtitle="symptom", status="propagated")
    c.arrow(522, 149, 548, 149, color="warning", width=2.8)
    c.text(532, 220, "1-2 hops only", color="warning", size=12, bold=True)
    c.text(532, 240, "causal search depth stays low", color="text_mid", size=11)

    # Column 3: telemetry board
    c.rect(730, 58, 316, 230, fill="#EEF4FA", stroke="primary", rx=8, stroke_width=2)
    c.rect(752, 30, 164, 28, fill="primary", stroke="none", rx=4)
    c.text(834, 49, "Signal dominance", color="#FFFFFF", size=12, bold=True)
    _evidence_chip(c, 756, 108, "metrics spike", "primary", "M")
    _evidence_chip(c, 756, 146, "logs expose clue", "accent", "L")
    _evidence_chip(c, 756, 184, "trace lands on culprit", "positive", "T")
    c.text(888, 238, "answer leaks before causal reconstruction", color="primary", size=11, bold=True)

    c.rect(120, 322, 840, 24, fill="warning", stroke="none", rx=4)
    c.text(540, 338, "Legacy construction leaks root-cause clues, allowing shortcut methods to score well", color="#FFFFFF", size=11, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_validation_gate_pipeline(*, width: float = 10.0, height: float = 3.0) -> SvgImage:
    W, H = 1080, 320
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()

    # Stage 1
    c.rect(32, 102, 186, 96, fill="bg_alt", stroke="secondary", rx=6, stroke_width=1.8)
    c.rect(32, 102, 186, 20, fill="secondary", stroke="none", rx=3)
    c.text(125, 116, "Inject faults", color="#FFFFFF", size=11, bold=True)
    c.text(125, 148, "31 fault types", color="text", size=12, bold=True)
    c.text(125, 168, "many are operationally silent", color="text_mid", size=11)

    # Stage 2
    c.rect(280, 102, 214, 96, fill="#FFF6EC", stroke="accent", rx=6, stroke_width=1.8)
    c.rect(280, 102, 214, 20, fill="accent", stroke="none", rx=3)
    c.text(387, 116, "Measure user impact", color="#FFFFFF", size=11, bold=True)
    c.text(387, 148, "SLI degradation required", color="accent", size=12, bold=True)
    c.text(387, 168, "anomaly alone is insufficient", color="text_mid", size=11)

    # Gate
    gate = [(562, 78), (684, 78), (738, 150), (684, 222), (562, 222), (616, 150)]
    c.polygon(gate, fill="#FFF1DE", stroke="warning", stroke_width=2.0)
    c.text(650, 134, "Validation", color="warning", size=16, bold=True)
    c.text(650, 154, "impact > theta", color="warning", size=13, bold=True)
    c.text(650, 173, "retain only meaningful faults", color="text_mid", size=11)

    # Valid output
    c.polygon([(830, 96), (1028, 96), (1048, 150), (1028, 204), (830, 204), (810, 150)],
              fill="#EAF3FC", stroke="primary", stroke_width=2)
    c.text(930, 132, "Validated benchmark case", color="primary", size=15, bold=True)
    c.text(930, 151, "fault -> user-visible impact", color="text_mid", size=11)
    c.text(930, 170, "evaluation-worthy incident", color="text_mid", size=11)

    c.arrow(218, 150, 280, 150, color="text_mid", width=2.8)
    c.arrow(494, 150, 560, 150, color="text_mid", width=2.8)
    c.arrow(738, 150, 810, 150, color="primary", width=3.0)

    # Discard branch
    c.path("M 650 222 C 692 250, 760 266, 850 270", stroke="warning", fill="none", stroke_width=2.0, dashed=True)
    c.rect(842, 254, 186, 30, fill="warning", stroke="none", rx=4)
    c.text(935, 273, "discard silent faults", color="#FFFFFF", size=11, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_framework_grid_panel(*, width: float = 10.0, height: float = 3.5) -> SvgImage:
    W, H = 1080, 400
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()

    cards = [
        (40, 58, "1 Foundation", "TrainTicket + observability stack", "secondary"),
        (390, 58, "2 Workload", "Dynamic traffic creates path variety", "accent"),
        (740, 58, "3 Injection", "31 layered fault types", "warning"),
        (740, 224, "4 Collection", "Metrics, logs, traces snapshot", "positive"),
        (390, 224, "5 Validation", "Retain only impact-validated cases", "primary"),
        (40, 224, "6 Annotation", "Hierarchical RCA labels + paths", "secondary"),
    ]
    for x, y, title, body, tone in cards:
        c.rect(x, y, 300, 108, fill="bg_alt" if tone != "primary" else "bg_accent", stroke=tone, rx=7, stroke_width=1.8)
        c.rect(x, y, 300, 22, fill=tone, stroke="none", rx=3)
        c.text(x + 150, y + 15, title, color="#FFFFFF", size=10, bold=True)
        c.text(x + 150, y + 62, body, color="text", size=12, bold=True)
        c.text(x + 150, y + 83, "stage", color="text_mid", size=10)

    c.arrow(340, 112, 390, 112, color="text_mid", width=2.4)
    c.arrow(690, 112, 740, 112, color="text_mid", width=2.4)
    c.arrow(890, 166, 890, 224, color="text_mid", width=2.4)
    c.arrow(740, 278, 690, 278, color="positive", width=2.4)
    c.arrow(390, 278, 340, 278, color="primary", width=2.4)

    c.rect(286, 338, 506, 34, fill="primary", stroke="none", rx=4)
    c.text(539, 359, "Scale comes from a reusable workflow, not hand-crafted incidents", color="#FFFFFF", size=11, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_benchmark_stats_panel(*, width: float = 6.6, height: float = 2.8) -> SvgImage:
    W, H = 760, 320
    c = DeckSvg(W, H, theme=THEME, bg=None)

    # Top metric strip
    c.rect(24, 34, 712, 136, fill="bg_alt", stroke="border", rx=7, stroke_width=1.4)
    metrics = [("1,430", "validated cases", "primary"), ("9,152", "fault injections", "accent"), ("25", "fault types", "positive")]
    for idx, (value, label, tone) in enumerate(metrics):
        x = 60 + idx * 232
        c.line(x - 32, 48, x - 32, 154, color="border", width=1.2) if idx > 0 else None
        c.text(x + 58, 92, value, color=tone, size=31, bold=True)
        c.text(x + 58, 118, label.upper(), color="text_mid", size=10, bold=True)
        c.rect(x + 6, 128, 104, 22, fill=tone, stroke="none", rx=3)
        c.text(x + 58, 143, "regime signal", color="#FFFFFF", size=9, bold=True)

    # Bottom supporting dimensions
    c.polygon([(42, 202), (236, 202), (254, 234), (236, 266), (42, 266), (24, 234)], fill="#EEF4FA", stroke="secondary", stroke_width=1.5)
    c.text(138, 229, "6 fault families", color="secondary", size=13, bold=True)
    c.text(138, 248, "cross-layer coverage", color="text_mid", size=10)
    c.polygon([(282, 202), (476, 202), (494, 234), (476, 266), (282, 266), (264, 234)], fill="#EEF4FA", stroke="secondary", stroke_width=1.5)
    c.text(378, 229, "50 services", color="secondary", size=13, bold=True)
    c.text(378, 248, "richer topology", color="text_mid", size=10)
    c.polygon([(522, 202), (716, 202), (734, 234), (716, 266), (522, 266), (504, 234)], fill="#FFF5EA", stroke="warning", stroke_width=1.5)
    c.text(618, 229, "dynamic workload", color="warning", size=13, bold=True)
    c.text(618, 248, "harder propagation", color="text_mid", size=10)
    return SvgImage(svg=c, width=width, height=height)


def svg_dual_regime_panel(*, width: float = 10.0, height: float = 3.3) -> SvgImage:
    W, H = 1080, 360
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.rect(36, 28, 486, 292, fill="bg_alt", stroke="border", rx=8, stroke_width=1.4)
    c.rect(558, 28, 486, 292, fill="#FFF7F2", stroke="warning", rx=8, stroke_width=1.4)
    c.text(279, 52, "Accuracy collapse", color="text", size=16, bold=True)
    c.text(801, 52, "Runtime escalation", color="text", size=16, bold=True)

    def pair(x: float, y: float, old_v: float, new_v: float, title: str) -> None:
        max_h = 136
        base_y = y + 168
        c.rect(x - 18, y + 24, 216, 184, fill="bg", stroke="border", rx=6, stroke_width=1.0)
        for idx, (label, v, tone) in enumerate([("legacy", old_v, "secondary"), ("realistic", new_v, "negative")]):
            bx = x + idx * 102
            bh = max_h * v
            c.rect(bx, base_y - bh, 64, bh, fill=tone, stroke="none", rx=4)
            c.text(bx + 32, max(base_y - bh - 9, y + 36), _fmt_v(v), color="text", size=14, bold=True)
            c.text(bx + 32, base_y + 20, label, color="text_mid", size=10)
        c.text(x + 48, y + 200, title, color="text_mid", size=11, bold=True)

    pair(86, 70, 0.75, 0.21, "Avg Top@1")
    pair(300, 70, 0.87, 0.37, "Best Top@1")
    c.rect(86, 286, 372, 24, fill="negative", stroke="none", rx=4)
    c.text(272, 302, "verdict: legacy ranking does not transfer", color="#FFFFFF", size=11, bold=True)

    # runtime runway
    c.line(620, 210, 1000, 210, color="border", width=2.2)
    c.polygon([(620, 194), (708, 194), (724, 210), (708, 226), (620, 226), (604, 210)], fill="#EEF4FA", stroke="secondary", stroke_width=1.8)
    c.text(664, 208, "seconds", color="secondary", size=13, bold=True)
    c.polygon([(782, 170), (1000, 170), (1026, 210), (1000, 250), (782, 250), (756, 210)], fill="#FCECEC", stroke="negative", stroke_width=1.8)
    c.text(892, 202, "hours", color="negative", size=22, bold=True)
    c.text(892, 223, "~12x runtime escalation", color="negative", size=11, bold=True)
    c.arrow(724, 210, 756, 210, color="negative", width=2.8)
    c.rect(620, 286, 372, 24, fill="warning", stroke="none", rx=4)
    c.text(806, 302, "verdict: realism raises operational cost sharply", color="#FFFFFF", size=11, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_failure_modes_panel(*, width: float = 10.0, height: float = 3.1) -> SvgImage:
    W, H = 1080, 350
    c = DeckSvg(W, H, theme=THEME, bg=None)
    # Runtime panel
    c.polygon([(28, 60), (336, 60), (336, 290), (54, 290), (28, 260)], fill="#FFF2F2", stroke="negative", stroke_width=2)
    c.rect(48, 32, 112, 28, fill="negative", stroke="none", rx=4)
    c.text(104, 50, "RUNTIME", color="#FFFFFF", size=11, bold=True)
    c.text(182, 96, "Scalability limits", color="negative", size=16, bold=True)
    c.line(74, 212, 132, 176, color="negative", width=4)
    c.line(132, 176, 188, 162, color="negative", width=4)
    c.line(188, 162, 246, 112, color="negative", width=4)
    c.text(182, 248, "toy-graph pipelines do not scale", color="text_mid", size=11)

    # Signal panel
    c.rect(386, 60, 300, 230, fill="#FFF7EC", stroke="warning", rx=8, stroke_width=2)
    c.rect(406, 32, 96, 28, fill="warning", stroke="none", rx=4)
    c.text(454, 50, "SIGNAL", color="#FFFFFF", size=11, bold=True)
    c.text(536, 96, "Observability blind spots", color="warning", size=16, bold=True)
    _evidence_chip(c, 420, 132, "metrics absent", "primary", "M")
    _evidence_chip(c, 420, 168, "logs noisy", "accent", "L")
    _evidence_chip(c, 420, 204, "trace sparse", "positive", "T")
    c.text(536, 248, "incomplete signals break causal tracing", color="text_mid", size=11)

    # Model panel
    c.rect(734, 60, 320, 230, fill="#EEF4FA", stroke="primary", rx=8, stroke_width=2)
    c.rect(754, 32, 96, 28, fill="primary", stroke="none", rx=4)
    c.text(802, 50, "MODEL", color="#FFFFFF", size=11, bold=True)
    c.text(894, 96, "Modeling bottlenecks", color="primary", size=16, bold=True)
    _draw_service_module(c, x=764, y=132, w=122, h=78, title="Assumed", subtitle="single-hop", status="normal")
    _draw_service_module(c, x=910, y=132, w=122, h=78, title="Reality", subtitle="cross-hop", status="propagated")
    c.arrow(886, 170, 906, 170, color="primary", width=2.6)
    c.text(894, 248, "legacy assumptions fail on realistic chains", color="text_mid", size=11)

    c.rect(182, 318, 716, 24, fill="negative", stroke="none", rx=4)
    c.text(540, 334, "Collapse drivers: scalability + observability + modeling limits", color="#FFFFFF", size=11, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_task_contract(*, width: float = 10.0, height: float = 3.0) -> SvgImage:
    W, H = 1080, 320
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()
    c.rect(34, 20, 248, 26, fill="accent", stroke="none", rx=4)
    c.text(158, 37, "GOAL-DRIVEN RCA CONTRACT", color="white", size=11, bold=True)
    c.rect(34, 68, 262, 186, fill="#F5F8FC", stroke="secondary", stroke_width=1.8, rx=10)
    c.polygon(
        [(52, 96), (252, 96), (272, 118), (252, 140), (52, 140), (32, 118)],
        fill="bg",
        stroke="secondary",
        stroke_width=1.4,
    )
    c.text(152, 114, "Operator query", color="secondary", size=15, bold=True)
    c.text(152, 136, "\"Why is checkout timing out?\"", color="text", size=12)
    c.text(64, 178, "Goal anchors retrieval:", color="secondary", size=11, bold=True, anchor="start")
    c.text(64, 198, "time of failure", color="text_mid", size=11, anchor="start")
    c.text(64, 216, "candidate component", color="text_mid", size=11, anchor="start")
    c.text(64, 234, "reason / causal explanation", color="text_mid", size=11, anchor="start")

    c.rect(356, 54, 370, 210, fill="#FFF8EF", stroke="accent", stroke_width=1.8, rx=10)
    c.rect(382, 78, 126, 24, fill="accent", stroke="none", rx=4)
    c.text(445, 95, "Telemetry bundle", color="white", size=11, bold=True)
    _draw_evidence_chip_bundle(
        c,
        x=384,
        y=116,
        chips=[("metrics", "metrics"), ("logs", "logs"), ("traces", "traces")],
    )
    c.line(394, 180, 690, 180, color="border", width=1.2, dashed=True)
    c.text(541, 198, "heterogeneous evidence must be fused over time", color="text_mid", size=11)
    c.rect(408, 214, 122, 28, fill="warning", stroke="none", rx=14)
    c.text(469, 232, "68 GB context", color="white", size=11, bold=True)
    c.rect(548, 214, 146, 28, fill="bg", stroke="accent", stroke_width=1.3, rx=14)
    c.text(621, 232, "multi-hop search", color="accent", size=11, bold=True)

    c.rect(786, 58, 262, 202, fill="#EEF4FA", stroke="primary", stroke_width=1.8, rx=10)
    c.rect(804, 78, 108, 24, fill="primary", stroke="none", rx=4)
    c.text(858, 95, "Output tuple", color="white", size=11, bold=True)
    c.rect(816, 118, 206, 72, fill="bg", stroke="primary", stroke_width=1.4, rx=8)
    c.text(919, 144, "(time, component,", color="text", size=14, bold=True)
    c.text(919, 168, "reason)", color="text", size=14, bold=True)
    c.text(917, 212, "not a generic ranked label", color="primary", size=11, bold=True)
    c.text(917, 232, "the task is protocol-like and goal conditioned", color="text_mid", size=10)

    c.arrow(296, 160, 356, 160, color="accent", width=2.6)
    c.arrow(726, 160, 786, 160, color="primary", width=2.6)
    return SvgImage(svg=c, width=width, height=height)


def svg_scale_pressure_panel(*, width: float = 10.0, height: float = 3.1) -> SvgImage:
    W, H = 1080, 340
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.rect(36, 28, 286, 252, fill="#F4F8FC", stroke="secondary", stroke_width=1.6, rx=10)
    c.rect(54, 44, 126, 24, fill="negative", stroke="none", rx=4)
    c.text(117, 61, "Scale pressure", color="white", size=11, bold=True)
    c.text(96, 118, "335", color="primary", size=34, bold=True)
    c.text(96, 144, "failure cases", color="text_mid", size=11, bold=True)
    c.text(96, 196, "68 GB", color="negative", size=34, bold=True)
    c.text(96, 222, "raw telemetry", color="text_mid", size=11, bold=True)
    c.text(226, 118, "3", color="accent", size=30, bold=True)
    c.text(226, 144, "systems", color="text_mid", size=11, bold=True)
    c.line(188, 86, 188, 244, color="border", width=1.4)
    c.text(178, 262, "size is part of the task", color="secondary", size=10, bold=True)

    c.rect(368, 28, 676, 252, fill="#FFF8EF", stroke="accent", stroke_width=1.6, rx=10)
    c.text(706, 54, "Telemetry mass arrives as a mixed reasoning load", color="accent", size=15, bold=True)
    _draw_evidence_chip_bundle(
        c,
        x=406,
        y=88,
        chips=[("metrics", "anomaly"), ("logs", "local clues"), ("traces", "hop chain")],
    )
    c.rect(402, 132, 270, 86, fill="bg", stroke="border", stroke_width=1.2, rx=8)
    c.text(537, 156, "heterogeneous signals", color="text", size=13, bold=True)
    c.text(537, 178, "different granularity", color="text_mid", size=11)
    c.text(537, 196, "different failure clues", color="text_mid", size=11)

    c.polygon(
        [(716, 92), (950, 92), (984, 130), (950, 168), (716, 168), (682, 130)],
        fill="#FFF1E3",
        stroke="warning",
        stroke_width=1.8,
    )
    c.text(833, 124, "Reasoning burden", color="warning", size=15, bold=True)
    c.text(833, 145, "long context + noisy evidence", color="text_mid", size=11)
    c.line(556, 238, 916, 238, color="negative", width=4)
    c.circle(634, 238, 8, fill="negative")
    c.circle(752, 238, 8, fill="negative")
    c.circle(870, 238, 8, fill="negative")
    c.text(734, 226, "multi-hop causal search", color="negative", size=11, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_capability_section_reset(*, width: float = 10.0, height: float = 2.5) -> SvgImage:
    W, H = 1080, 280
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.rect(50, 42, 980, 22, fill="accent", stroke="none", rx=4)
    c.rect(50, 42, 210, 22, fill="primary", stroke="none", rx=4)
    c.text(155, 58, "ACT III", color="white", size=10, bold=True)
    c.text(292, 58, "Question B: Capability", color="white", size=10, bold=True)
    c.text(58, 126, "Now we ask the LLM capability question", color="accent", size=28, bold=True, anchor="start")
    c.text(58, 162, "Can LLMs actually diagnose RCA under real telemetry?", color="text", size=18, bold=True, anchor="start")
    c.rect(58, 204, 370, 34, fill="bg_alt", stroke="secondary", stroke_width=1.3, rx=17)
    c.text(243, 225, "recap: realism already rules out easy benchmark wins", color="secondary", size=11, bold=True)
    c.line(464, 221, 1012, 221, color="border", width=1.6, dashed=True)
    c.text(738, 208, "the next slides measure real diagnostic ability, not benchmark saturation", color="text_mid", size=11, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_capability_gap_stage(*, width: float = 10.0, height: float = 3.0) -> SvgImage:
    W, H = 1080, 340
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.rect(44, 36, 994, 30, fill="negative", stroke="none", rx=5)
    c.text(541, 56, "Verdict: even strong models remain in a low-usable RCA regime", color="white", size=13, bold=True)

    # usable-zone band
    chart_x, chart_y = 82, 104
    chart_w, chart_h = 520, 174
    c.rect(chart_x, chart_y, chart_w, chart_h, fill="bg_alt", stroke="border", stroke_width=1.2, rx=8)
    c.rect(chart_x, chart_y, chart_w, 62, fill="#FCECEC", stroke="none", rx=8, opacity=0.85)
    c.text(chart_x + chart_w / 2, chart_y + 38, "low-usable RCA zone", color="negative", size=16, bold=True)
    base_y = chart_y + chart_h - 22
    c.line(chart_x + 46, base_y, chart_x + chart_w - 32, base_y, color="text_mid", width=2)
    bar_specs = [
        ("Oracle", 5.37, "secondary", chart_x + 118),
        ("Sampled", 3.88, "negative", chart_x + 300),
    ]
    max_v = 12.0
    for label, value, tone, x in bar_specs:
        bh = (chart_h - 56) * (value / max_v)
        y = base_y - bh
        c.rounded_rect(x, y, 78, bh, fill=tone, stroke="none", rx=10)
        c.text(x + 39, y - 12, _fmt_v(value), color="text", size=22, bold=True)
        c.text(x + 39, base_y + 24, label, color="text_mid", size=13)
    c.line(chart_x + 46, chart_y + 102, chart_x + chart_w - 32, chart_y + 102, color="warning", width=2, dashed=True)
    c.text(chart_x + chart_w - 38, chart_y + 92, "still far from reliable", color="warning", size=11, bold=True, anchor="end")
    c.arrow(chart_x + 208, chart_y + 124, chart_x + 292, chart_y + 146, color="negative", width=2.3)
    c.text(chart_x + 248, chart_y + 122, "sampling hurts further", color="negative", size=11, bold=True)

    c.rect(652, 94, 352, 184, fill="#EEF4FA", stroke="primary", stroke_width=1.5, rx=10)
    c.rect(674, 114, 126, 24, fill="primary", stroke="none", rx=4)
    c.text(737, 131, "Why this matters", color="white", size=11, bold=True)
    c.text(826, 162, "Capability gap", color="primary", size=18, bold=True)
    c.text(826, 190, "real RCA remains outside", color="text_mid", size=12)
    c.text(826, 210, "current LLM comfort zones", color="text_mid", size=12)
    c.line(696, 232, 960, 232, color="border", width=1.2, dashed=True)
    c.text(828, 258, "focus object = the gap, not the raw score", color="primary", size=11, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_agent_three_phase_track(*, width: float = 10.0, height: float = 3.0) -> SvgImage:
    W, H = 1080, 340
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()
    c.rect(58, 42, 964, 26, fill="accent", stroke="none", rx=4)
    c.text(540, 60, "Agent helps -> ceiling remains -> residual usability gap stays open", color="white", size=12, bold=True)

    phases = [
        (74, 96, 260, 182, "Phase 1", "Gain", "tool use doubles best score", "accent"),
        (408, 96, 262, 182, "Phase 2", "Ceiling", "absolute performance still low", "warning"),
        (744, 96, 262, 182, "Phase 3", "Residual gap", "not yet reliable for operators", "negative"),
    ]
    for x, y, w, h, kicker, title, body, tone in phases:
        c.rect(x, y, w, h, fill="bg_alt", stroke=tone, stroke_width=1.7, rx=10)
        c.rect(x + 18, y + 16, 74, 22, fill=tone, stroke="none", rx=4)
        c.text(x + 55, y + 31, kicker, color="white", size=10, bold=True)
        c.text(x + w / 2, y + 76, title, color=tone, size=18, bold=True)
        c.text(x + w / 2, y + 108, body, color="text_mid", size=12)
    c.arrow(334, 186, 408, 186, color="warning", width=2.5)
    c.arrow(670, 186, 744, 186, color="negative", width=2.5)

    # embed quantitative gain only in phase 1
    c.rounded_rect(110, 158, 58, 86, fill="secondary", stroke="none", rx=8)
    c.rounded_rect(194, 120, 58, 124, fill="primary", stroke="none", rx=8)
    c.text(139, 150, "5.37", color="text", size=14, bold=True)
    c.text(223, 112, "11.34", color="text", size=14, bold=True)
    c.text(139, 262, "base", color="text_mid", size=11)
    c.text(223, 262, "agent", color="text_mid", size=11)

    c.line(454, 214, 620, 214, color="warning", width=2, dashed=True)
    c.text(537, 202, "higher than base, still below useful ceiling", color="warning", size=10, bold=True)
    c.rect(788, 224, 176, 30, fill="negative", stroke="none", rx=15)
    c.text(876, 243, "gap not closed", color="white", size=11, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_takeaway_b_panel(*, width: float = 10.0, height: float = 2.5) -> SvgImage:
    W, H = 1080, 280
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.rect(58, 58, 964, 56, fill="accent", stroke="none", rx=6)
    c.text(540, 92, "Takeaway B: better tasks expose the true capability gap", color="white", size=19, bold=True)
    c.rect(58, 148, 266, 44, fill="bg_alt", stroke="negative", stroke_width=1.4, rx=22)
    c.text(191, 176, "realistic tasks remove benchmark illusion", color="negative", size=11, bold=True)
    c.line(356, 170, 820, 170, color="border", width=1.6, dashed=True)
    c.circle(540, 170, 18, fill="accent", opacity=0.92)
    c.text(540, 176, "GAP", color="white", size=9, bold=True)
    c.text(846, 174, "current LLM agents remain far from reliable RCA", color="accent", size=12, bold=True, anchor="start")
    c.rect(58, 224, 964, 22, fill="positive", stroke="none", rx=4)
    c.text(540, 239, "bridge: next we ask whether correct labels can still hide invalid reasoning paths", color="white", size=10, bold=True)
    return SvgImage(svg=c, width=width, height=height)


def svg_label_reasoning_tension(*, width: float = 9.5, height: float = 3.0) -> SvgImage:
    W, H = 1030, 330
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()
    c.rounded_rect(36, 46, 426, 228, text="", fill="#F4FBF7", stroke="positive", rx=20, stroke_width=2)
    c.rounded_rect(568, 46, 426, 228, text="", fill="#FFF4E8", stroke="warning", rx=20, stroke_width=2)
    c.text(249, 84, "Correct final label", color="positive", size=18, bold=True)
    c.text(781, 84, "Wrong reasoning path", color="warning", size=18, bold=True)

    _draw_service_module(c, x=84, y=118, w=132, h=82, title="Search svc", subtitle="actual fault", status="root")
    _draw_service_module(c, x=270, y=118, w=132, h=82, title="Checkout", subtitle="observed symptom", status="propagated")
    c.arrow(216, 159, 268, 159, color="positive", width=2.8)
    c.text(249, 226, "Final diagnosis names the right service.", color="text_mid", size=12)

    _draw_service_module(c, x=614, y=116, w=132, h=82, title="Cache svc", subtitle="shortcut clue", status="propagated")
    _draw_service_module(c, x=818, y=116, w=132, h=82, title="Checkout", subtitle="symptom", status="propagated")
    c.arrow(746, 157, 816, 157, color="warning", width=2.8)
    c.path("M 680 102 C 754 62, 864 64, 910 104", stroke="negative", fill="none", stroke_width=2.2, dashed=True)
    c.text(781, 226, "Answer can be right even when the causal story is unsupported.", color="text_mid", size=12)

    c.rounded_rect(366, 286, 304, 30, text="Correct label != trustworthy process", fill="negative", stroke="none",
                   text_color="white", font_size=13, bold=True, rx=15)
    return SvgImage(svg=c, width=width, height=height)


def svg_outcome_blind_spot(*, width: float = 9.5, height: float = 3.0) -> SvgImage:
    W, H = 1030, 330
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()
    c.rounded_rect(36, 42, 282, 236, text="", fill="bg_alt", stroke="warning", rx=20, stroke_width=2)
    c.text(177, 78, "Outcome-only check", color="warning", size=18, bold=True)
    c.rounded_rect(84, 112, 186, 90, text="", fill="#FFF7EA", stroke="warning", rx=16, stroke_width=1.8)
    c.text(177, 146, "Was the label correct?", color="text", size=14, bold=True)
    c.text(177, 170, "Yes / No", color="warning", size=17, bold=True)
    c.text(177, 232, "Intermediate steps stay invisible", color="text_mid", size=12)

    c.rounded_rect(368, 42, 626, 236, text="", fill="bg_accent", stroke="primary", rx=20, stroke_width=2)
    c.text(681, 78, "Process-aware verification", color="primary", size=18, bold=True)
    steps = [
        (454, "Intervention", "secondary"),
        (604, "Propagation 1", "accent"),
        (754, "Propagation 2", "accent"),
        (904, "Symptom", "primary"),
    ]
    for x, label, tone in steps:
        c.rounded_rect(x - 52, 126, 104, 70, text="", fill="bg", stroke=tone, rx=14, stroke_width=2)
        c.text(x, 152, label, color=tone, size=12, bold=True)
        c.text(x, 176, "verified", color="positive", size=11, bold=True)
    for i in range(len(steps) - 1):
        c.arrow(steps[i][0] + 52, 161, steps[i + 1][0] - 52, 161, color="primary", width=2.6)
    c.rounded_rect(520, 222, 322, 32, text="Trust requires the path to be checkable, not just the answer.", fill="primary",
                   stroke="none", text_color="white", font_size=12, bold=True, rx=15)
    return SvgImage(svg=c, width=width, height=height)


def svg_stepwise_supervision_upgrade(*, width: float = 9.5, height: float = 3.1) -> SvgImage:
    W, H = 1030, 340
    c = DeckSvg(W, H, theme=THEME, bg=None)
    c.arrow_markers()
    c.rounded_rect(40, 52, 388, 230, text="", fill="bg_alt", stroke="warning", rx=20, stroke_width=2)
    c.rounded_rect(602, 52, 388, 230, text="", fill="bg_accent", stroke="primary", rx=20, stroke_width=2)
    c.text(234, 88, "Before: outcome label only", color="warning", size=17, bold=True)
    c.text(796, 88, "After: step-wise supervision", color="primary", size=17, bold=True)

    c.rounded_rect(128, 136, 212, 76, text="", fill="#FFF7EA", stroke="warning", rx=14, stroke_width=1.8)
    c.text(234, 164, "Root cause answer", color="text", size=14, bold=True)
    c.text(234, 186, "single end label", color="text_mid", size=11)

    xs = [654, 758, 862, 966]
    labels = ["Cause", "Step 1", "Step 2", "Symptom"]
    tones = ["secondary", "accent", "accent", "primary"]
    for x, label, tone in zip(xs, labels, tones):
        c.rounded_rect(x - 42, 140, 84, 66, text="", fill="bg", stroke=tone, rx=14, stroke_width=2)
        c.text(x, 164, label, color=tone, size=12, bold=True)
        c.text(x, 186, "check", color="positive", size=10, bold=True)
    for i in range(3):
        c.arrow(xs[i] + 42, 173, xs[i + 1] - 42, 173, color="primary", width=2.6)
        c.circle((xs[i] + xs[i + 1]) / 2, 173, 8, fill="positive", opacity=0.92)
        c.text((xs[i] + xs[i + 1]) / 2, 177, "v", color="white", size=8, bold=True)
    c.rounded_rect(332, 296, 364, 30, text="500 instances gain independently verifiable causal hops.", fill="positive",
                   stroke="none", text_color="white", font_size=12, bold=True, rx=15)
    return SvgImage(svg=c, width=width, height=height)


def svg_future_pillars(*, width: float = 10.0, height: float = 3.0) -> SvgImage:
    W, H = 1080, 330
    c = DeckSvg(W, H, theme=THEME, bg=None)
    pillars = [
        (40, "Data", "Harder incidents\nconcurrent faults\npartial observability", "primary"),
        (390, "Agents", "Retrieval planning\nhypothesis revision\ntool orchestration", "accent"),
        (740, "Training", "Process-aware labels\ncausal supervision\ndiagnostic curricula", "positive"),
    ]
    for x, title, body, tone in pillars:
        c.rounded_rect(x, 54, 300, 194, text="", fill="bg_alt", stroke=tone, rx=20, stroke_width=2)
        c.rounded_rect(x + 92, 26, 116, 34, text=title, fill=tone, stroke="none",
                       text_color="white", font_size=13, bold=True, rx=16)
        for idx, line in enumerate(body.splitlines()):
            c.text(x + 150, 114 + idx * 28, line, color="text", size=13, bold=(idx == 0))
    c.arrow_markers()
    c.arrow(340, 151, 390, 151, color="accent", width=2.4)
    c.arrow(690, 151, 740, 151, color="positive", width=2.4)
    c.rounded_rect(352, 282, 378, 30, text="Goal: evaluations that support reliable AI operations, not just higher headline scores.",
                   fill="bg_accent", stroke="primary", text_color="primary", font_size=12, bold=True, rx=15)
    return SvgImage(svg=c, width=width, height=height)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def callout_box(
    title: str, body: str, color: str = "primary", *, width: float | None = None
) -> HStack:
    """Callout with left accent bar: accent | title / body."""
    accent = Box(text="", color=color, border=color, width=0.07, height=1.2)
    text_col = VStack(
        gap=0.07,
        children=[
            TextBlock(text=title, font_size="body", bold=True, color=color),
            TextBlock(text=body, font_size="caption", color="text"),
        ],
    )
    return HStack(gap=0.12, width=width, children=[accent, text_col])


def verdict_band(
    title: str,
    body: str,
    color: str = "primary",
    *,
    width: float | None = None,
) -> VStack:
    return VStack(
        gap=0.04,
        width=width,
        children=[
            Box(text=title.upper(), color=color, border=color, text_color="white", height=0.22),
            TextBlock(text=body, font_size="caption", color="text"),
        ],
    )


def metric_chip(label: str, value: str, tone: str = "primary", *, width: float = 1.45) -> VStack:
    return VStack(
        gap=0.01,
        width=width,
        children=[
            TextBlock(text=label.upper(), font_size="small", color=tone, bold=True),
            TextBlock(text=value, font_size="heading", color="text", bold=True),
        ],
    )


def metric_cluster(
    items: list[tuple[str, str, str]],
    *,
    cols: int = 3,
    verdict: tuple[str, str, str] | None = None,
) -> VStack:
    children: list = [
        Grid(
            cols=cols,
            gap=0.18,
            children=[metric_chip(label, value, tone) for label, value, tone in items],
        )
    ]
    if verdict:
        children.append(verdict_band(verdict[0], verdict[1], verdict[2]))
    return VStack(gap=0.12, children=children)


def metric_card(label: str, value: str, tone: str = "primary", *, width: float | None = None):
    return VStack(
        gap=0.02,
        width=width,
        children=[
            TextBlock(text=label.upper(), font_size=8, color=tone, bold=True),
            TextBlock(text=value, font_size="heading", bold=True, color="text"),
        ],
    )


def stat_grid(items: list[tuple[str, str, str]], *, cols: int | None = None):
    return Grid(
        cols=cols or min(3, len(items)),
        gap=0.20,
        children=[metric_card(label, value, tone) for label, value, tone in items],
    )


def triad_callout(items: list[tuple[str, str, str]]) -> Grid:
    """3-column grid of callout boxes."""
    return Grid(
        cols=3,
        gap=0.22,
        children=[callout_box(title, body, tone) for title, body, tone in items],
    )


def modality_triptych():
    cards = []
    for label, icon_fn, tone in [
        ("Metrics", icon_metrics, "primary"),
        ("Logs",    icon_logs,    "accent"),
        ("Traces",  icon_traces,  "positive"),
    ]:
        cards.append(
            VStack(
                gap=0.05,
                children=[
                    SvgImage(svg=icon_fn(), width=0.82, height=0.82),
                    TextBlock(text=label, font_size="caption", bold=True, color=tone),
                ],
            )
        )
    return HStack(gap=0.32, children=cards)


def modality_signal_cards():
    cards = []
    for label, icon_fn, tone, body in [
        ("Metrics", icon_metrics, "primary", "When does the anomaly start?"),
        ("Logs", icon_logs, "accent", "What fails locally and semantically?"),
        ("Traces", icon_traces, "positive", "Where does the fault propagate next?"),
    ]:
        cards.append(
            VStack(
                gap=0.08,
                width=2.9,
                children=[
                    HStack(
                        gap=0.10,
                        children=[
                            SvgImage(svg=icon_fn(), width=0.64, height=0.64),
                            TextBlock(text=label, font_size="body", bold=True, color=tone),
                        ],
                    ),
                    TextBlock(text=body, font_size="caption", color="text_mid"),
                ],
            )
        )
    return HStack(gap=0.24, children=cards)


def section_banner(
    act_label: str,
    question: str,
    *,
    tone: str = "accent",
    recap: str | None = None,
) -> VStack:
    children: list = [
        Box(text=act_label.upper(), color=tone, border=tone, text_color="white", height=0.24),
        TextBlock(text=question, font_size="subtitle", bold=True, color=tone),
    ]
    if recap:
        children.append(
            HStack(
                gap=0.10,
                children=[
                    Badge(text="Recap", color="secondary"),
                    TextBlock(text=recap, font_size="caption", color="text_mid"),
                ],
            )
        )
    return VStack(gap=0.08, children=children)


def takeaway_strip(
    claim: str,
    *,
    bridge: str | None = None,
    tone: str = "positive",
    support: str | None = None,
) -> VStack:
    children: list = [callout_box("Takeaway", claim, tone)]
    if support:
        children.append(Box(text=support, color=tone, border=tone, text_color="white", height=0.22))
    if bridge:
        children.append(TextBlock(text=bridge, font_size="caption", color="text_mid"))
    return VStack(gap=0.08, children=children)


def triptych_panel(
    items: list[tuple[str, str, str]],
    *,
    tones: list[str] | None = None,
) -> Grid:
    tones = tones or ["primary", "accent", "positive"]
    panels = []
    for idx, (title, body, kicker) in enumerate(items):
        tone = tones[min(idx, len(tones) - 1)]
        panels.append(
            VStack(
                gap=0.05,
                children=[
                    Box(text=title.upper(), color=tone, border=tone, text_color="white", height=0.24),
                    TextBlock(text=body, font_size="caption", color="text"),
                    TextBlock(text=kicker, font_size="small", color=tone, bold=True),
                ],
            )
        )
    return Grid(cols=3, gap=0.22, children=panels)


def stage_marker(token: str, title: str, subtitle: str, tone: str = "primary", *, width: float = 1.75) -> VStack:
    return VStack(
        gap=0.08,
        width=width,
        children=[
            Circle(text=token, color=tone, radius=0.28),
            TextBlock(text=title, font_size="caption", bold=True, color=tone, align="center"),
            TextBlock(text=subtitle, font_size="small", color="text_mid", align="center"),
        ],
    )


def arrow_text(symbol: str = "→", tone: str = "accent", *, width: float = 0.36) -> TextBlock:
    return TextBlock(text=symbol, font_size=18, bold=True, color=tone, width=width, align="center")


def path_track(
    steps: list[tuple[str, str, str, str]],
    *,
    arrow_tone: str = "accent",
    width: float | None = None,
) -> HStack:
    children: list = []
    for idx, (token, title, subtitle, tone) in enumerate(steps):
        children.append(stage_marker(token, title, subtitle, tone))
        if idx < len(steps) - 1:
            children.append(arrow_text(tone=arrow_tone))
    return HStack(gap=0.12, width=width, children=children)


def insight_panel(
    title: str,
    body: str,
    tone: str = "primary",
    *,
    width: float | None = None,
    fill: str = "bg_alt",
) -> VStack:
    return VStack(
        gap=0.06,
        width=width,
        children=[
            Box(text=title.upper(), color=tone, border=tone, text_color="white", height=0.24),
            Box(text=body, color=fill, border=tone, text_color="text", font_size="caption", bold=False),
        ],
    )


def metric_track(
    label: str,
    value: float,
    tone: str = "primary",
    *,
    max_value: float = 1.0,
    track_width: float = 2.85,
    value_fmt: str | None = None,
) -> VStack:
    frac = 0.0 if max_value <= 0 else max(0.0, min(1.0, value / max_value))
    fill_w = max(0.22, track_width * frac)
    rest_w = max(0.22, track_width - fill_w)
    shown = value_fmt if value_fmt is not None else _fmt_v(value)
    return VStack(
        gap=0.04,
        children=[
            HStack(
                gap=0.10,
                children=[
                    TextBlock(text=label, font_size="caption", bold=True, color="text", width=1.55),
                    TextBlock(text=shown, font_size="caption", bold=True, color=tone),
                ],
            ),
            HStack(
                gap=0.04,
                children=[
                    Box(text="", color=tone, border=tone, width=fill_w, height=0.16),
                    Box(text="", color="bg_alt", border="border", width=rest_w, height=0.16),
                ],
            ),
        ],
    )


def roadmap_track(milestones: list[tuple[str, str, str]], *, width: float = 9.4, height: float = 1.95) -> HStack:
    children: list = []
    for idx, (title, subtitle, tone) in enumerate(milestones):
        children.append(stage_marker(str(idx + 1), title, subtitle, tone, width=2.2))
        if idx < len(milestones) - 1:
            children.append(arrow_text("→", tone="secondary", width=0.4))
    return HStack(gap=0.18, width=width, children=children)


def manifesto_close(
    title: str,
    subtitle: str,
    *,
    action: str | None = None,
    tone: str = "primary",
) -> VStack:
    children: list = [
        TextBlock(text=title, font_size="title", color=tone, bold=True),
        TextBlock(text=subtitle, font_size="body", color="text_mid"),
    ]
    if action:
        children.append(Box(text=action, color=tone, border=tone, text_color="white", height=0.26))
    return VStack(gap=0.10, children=children)


def verification_loop_svg(*, width: float = 7.6, height: float = 2.4) -> VStack:
    loop = path_track(
        [
            ("I", "Intervene", "known fault", "accent"),
            ("F", "Forward", "cause -> effect", "primary"),
            ("V", "Verify", "match observed path", "positive"),
            ("J", "Judge", "faithful or not", "negative"),
        ],
        arrow_tone="accent",
        width=width,
    )
    return VStack(
        gap=0.10,
        width=width,
        children=[
            loop,
            HStack(
                gap=0.12,
                children=[
                    Badge(text="audit loop", color="secondary"),
                    TextBlock(
                        text="Intervention makes trust evaluation operational: we verify the predicted forward path before granting credit.",
                        font_size="caption",
                        color="text_mid",
                    ),
                ],
            ),
        ],
    )


def step_supervision_track_svg(*, width: float = 8.6, height: float = 2.0) -> HStack:
    return path_track(
        [
            ("S", "Symptom", "user-visible", "secondary"),
            ("1", "Hop 1", "service edge", "positive"),
            ("2", "Hop 2", "service edge", "positive"),
            ("R", "Root", "initiating fault", "negative"),
        ],
        arrow_tone="positive",
        width=width,
    )


def flow_pipeline(
    labels: list[str],
    colors: list[str] | None = None,
    *,
    direction: str = "horizontal",
) -> HStack | VStack:
    """Connected chain of RoundedBoxes with Arrows."""
    if colors is None:
        colors = ["bg_alt"] * len(labels)
    boxes = [
        RoundedBox(
            text=label,
            color=color,
            border="border" if color in {"bg_alt", "bg", "bg_accent"} else color,
            text_color="text" if color in {"bg_alt", "bg", "bg_accent"} else "white",
            font_size="caption",
            bold=True,
            height=0.88,
            size_mode_x="fit",
        )
        for label, color in zip(labels, colors)
    ]
    children: list = []
    arrow_dir = "vertical" if direction in {"vertical", "down"} else "horizontal"
    for idx, box in enumerate(boxes):
        children.append(box)
        if idx < len(boxes) - 1:
            children.append(
                Arrow(
                    from_component=box,
                    to_component=boxes[idx + 1],
                    color="text_light",
                    direction=arrow_dir,
                )
            )
    Container = VStack if arrow_dir == "vertical" else HStack
    return Container(gap=0.14, children=children)


def framework_grid(stages: list[tuple[str, str, str]]):
    return Grid(
        cols=3,
        gap=0.18,
        children=[
            RoundedBox(
                text=f"{title}\n{body}",
                color=tone,
                border="border" if tone in {"bg_alt", "bg", "bg_accent"} else tone,
                text_color="text" if tone in {"bg_alt", "bg", "bg_accent"} else "white",
                bold=False,
                font_size="caption",
                height=1.15,
                size_mode_x="fit",
            )
            for title, body, tone in stages
        ],
    )


# ---------------------------------------------------------------------------
# Slide data
# ---------------------------------------------------------------------------

SLIDE_DATA = [
    {"no": "01", "title": "Building Trustworthy RCA Evaluation for LLM Agents",    "kind": "hero",                "notes": "开场直接定义主线：真实性、能力、可信性三层评测是同一条研究链。"},
    {"no": "02", "title": "RCA Failures Are Expensive and Cascading",               "kind": "stakes",              "notes": "让听众先认可问题价值：不是学术游戏，而是恢复时间和业务损失问题。"},
    {"no": "03", "title": "RCA Requires Causal Multi-Modal Multi-Hop Reasoning",    "kind": "hardness",            "notes": "解释难点本质：跨模态、跨服务、跨时间的因果推断，不是简单匹配。"},
    {"no": "04", "title": "Current Evaluation Often Rewards Shortcut Correlations", "kind": "shortcut_compare",    "notes": "把冲突抛出来：任务很难，但分数看起来很高，说明评测可能出了偏差。"},
    {"no": "05", "title": "Our Evaluation Ladder Uses Three Questions",             "kind": "question_ladder",     "notes": "明确方法论：先确认题目是否真实，再测能力，最后测推理是否可信。"},
    {"no": "06", "title": "What You Should Remember in 45 Minutes",                 "kind": "takeaway_grid",       "notes": "先给听众记忆钩子，后面所有证据都回扣这三句话。"},
    {"no": "07", "title": "If a Simple Heuristic Wins, the Benchmark Is Too Easy",  "kind": "probe",               "notes": "用反证法建立可信度：先不用复杂模型，看简单规则能否打平。"},
    {"no": "08", "title": "SimpleRCA Matches or Beats SOTA on Public Benchmarks",   "kind": "simplerca_chart",     "notes": "强调含义而非逐项读数：如果简单方法都够用，评测区分度就不足。"},
    {"no": "09", "title": "Legacy Benchmarks Are Oversimplified by Construction",   "kind": "legacy_triad",        "notes": "解释机制：根因症状太显眼、传播太短，导致相关性捷径有效。"},
    {"no": "10", "title": "Realism Needs Impact-Validated Failures, Not Raw Injections", "kind": "validation_pipeline", "notes": "点出关键原则：并非注入就算故障，必须对用户指标产生影响才有评测价值。"},
    {"no": "11", "title": "Fault-Propagation-Aware Benchmarking at Scale",          "kind": "framework",           "notes": "这页强调可复现的大规模构建能力，不是一次性手工造数据。"},
    {"no": "12", "title": "New Benchmark Stats Show a Different Difficulty Regime", "kind": "benchmark_stats",     "notes": "只保留改变结论的数字，让听众看到难度分布已经变了。"},
    {"no": "13", "title": "Performance Collapses Under Realistic Conditions",       "kind": "collapse",            "notes": "这页是第一个冲击证据：过去的高分在新基准下显著下滑。"},
    {"no": "14", "title": "Three Failure Modes Explain the Collapse",               "kind": "failure_modes",       "notes": "从分数下降转到系统性瓶颈，为后续 LLM 评测打地基。"},
    {"no": "15", "title": "Takeaway A: Benchmark Realism Determines What Progress Means", "kind": "takeaway_a",   "notes": "结论收口：先把题目做对，后续谈模型能力才有意义。"},
    {"no": "16", "title": "Now We Ask the LLM Capability Question",                 "kind": "transition_b",        "notes": "过渡语义：不是 LLM 万能论，而是在真实任务上做能力测量。"},
    {"no": "17", "title": "OpenRCA Defines RCA as a Goal-Driven Task",             "kind": "task_contract",       "notes": "强调任务定义升级：不是只猜组件，而是时间、组件、原因三元目标。"},
    {"no": "18", "title": "OpenRCA Forces Reasoning Over Real Telemetry Scale",    "kind": "openrca_scale",       "notes": "让听众感知真实规模压力：上下文长、数据异构、噪声高。"},
    {"no": "19", "title": "Current LLMs Struggle on Core RCA Tasks",               "kind": "llm_scores",          "notes": "这里要讲能力边界：在真实条件下，现有模型远未达到可用水平。"},
    {"no": "20", "title": "Execution-Based RCA-Agent Helps but Does Not Close the Gap", "kind": "agent_gain",     "notes": "客观评价 agent：有提升，但不是问题已解决，仍是低可用区间。"},
    {"no": "21", "title": "Takeaway B: Better Tasks Expose the True Capability Gap","kind": "takeaway_b",          "notes": "收束第二问：不是模型没进步，而是离可靠诊断还有明显距离。"},
    {"no": "22", "title": "Correct Labels Can Still Hide Wrong Reasoning",          "kind": "label_vs_reasoning",  "notes": "关键转折页：引出第三问，为什么答对不等于可托付。"},
    {"no": "23", "title": "Outcome-Only Evaluation Misses Process Failures",       "kind": "outcome_only",        "notes": "把问题说清：缺的是过程可验证性，不是再多一个最终标签。"},
    {"no": "24", "title": "FORGE Uses Forward Verification from Known Interventions","kind": "forge_pipeline",     "notes": "强调 FORGE 的核心优势：把难做的逆向推断变成可操作的正向验证。"},
    {"no": "25", "title": "OpenRCA 2.0 Adds Step-Wise Causal Supervision",        "kind": "stepwise_supervision","notes": "讲清新增了什么监督信号：每一步传播链都可核对。"},
    {"no": "26", "title": "Process Metrics Separate Identification from Reasoning","kind": "metrics_panel",       "notes": "把指标解释成会不会找对和能不能讲对因果链两层能力。"},
    {"no": "27", "title": "Best-Model Gap Shows Hidden Reasoning Defects",         "kind": "best_gap",            "notes": "重点句：最高分模型也存在明显过程缺口，不能只看最终命中率。"},
    {"no": "28", "title": "The Trust Gap Widens Across 7 LLMs",                   "kind": "avg_gap",             "notes": "从个例到总体：过程失真是普遍现象，不是某个模型偶然失误。"},
    {"no": "29", "title": "Takeaway C: Trustworthy RCA Needs Causal-Path Faithfulness", "kind": "takeaway_c",    "notes": "第三问收束：可托付诊断必须答对且推理链可验证。"},
    {"no": "30", "title": "One Research Line: Realism \u2192 Capability \u2192 Trust",  "kind": "synthesis",      "notes": "再次强调不是三篇拼盘，而是同一评测框架的连续推进。"},
    {"no": "31", "title": "Next Agenda: Process-Aware Data, Agents, and Training",  "kind": "future_agenda",      "notes": "未来方向要和评测闭环挂钩：数据更真、agent更强、训练更重过程。"},
    {"no": "32", "title": "Build Evaluations That Make Reliable AI Operations Possible", "kind": "closing",       "notes": "结尾句收在先把评测做对，可靠运维智能体才可能成立。"},
]

BACKUP_DATA = [
    ("B1", "Why SimpleRCA Wins on Legacy Benchmarks"),
    ("B2", "Fault-Propagation-Aware Benchmark Construction Details"),
    ("B3", "OpenRCA Task Granularity and Metric Definition"),
    ("B4", "RCA-Agent Strengths and Failure Patterns"),
    ("B5", "FORGE Verification Pipeline and Metric Math"),
    ("B6", "Threats to Validity Across the Whole Research Line"),
]


# ---------------------------------------------------------------------------
# Main slide builder
# ---------------------------------------------------------------------------

def build_slide(prs: Presentation, spec: dict):  # noqa: C901
    kind = spec["kind"]
    title = spec["title"]
    notes = spec["notes"]

    # -------- hero (slide 01) --------
    if kind == "hero":
        sb = prs.slide(background="bg_accent")
        stage_bar = Box(
            text="RESEARCH-LINE THESIS  |  REALISM -> CAPABILITY -> TRUST",
            color="primary",
            border="primary",
            text_color="white",
            height=0.28,
        )
        thesis = VStack(
            gap=0.08,
            width=6.4,
            children=[
                TextBlock(text="Building Trustworthy RCA Evaluation for LLM Agents", font_size=31, bold=True, color="primary"),
                TextBlock(
                    text="A 45-minute evaluation contract: progress is credible only when benchmark realism, capability under telemetry scale, and causal-path trust all hold.",
                    font_size="body",
                    color="text_mid",
                ),
            ],
        )
        ladder = VStack(
            gap=0.14,
            width=6.6,
            children=[
                Box(text="Evaluation contract: A gates B, B gates C", color="bg_alt", border="border", text_color="text_mid", height=0.22),
                HStack(
                    gap=0.18,
                    children=[
                        svg_gate_node("A", "Realism", "primary"),
                        arrow_text(tone="accent"),
                        svg_gate_node("B", "Capability", "accent"),
                        arrow_text(tone="positive"),
                        svg_gate_node("C", "Trust", "positive"),
                    ],
                ),
                HStack(
                    gap=0.16,
                    children=[
                        verdict_band("A", "Is the task operationally real?", "primary", width=2.0),
                        verdict_band("B", "Can agents reason over real telemetry?", "accent", width=2.0),
                        verdict_band("C", "Can we audit the causal path?", "positive", width=2.0),
                    ],
                ),
                HStack(
                    gap=0.14,
                    children=[
                        Badge(text="FSE'26", color="primary"),
                        Badge(text="OpenRCA", color="accent"),
                        Badge(text="OpenRCA 2.0", color="positive"),
                    ],
                ),
            ],
        )
        payoff = VStack(
            gap=0.13,
            width=3.0,
            children=[
                Box(text="TALK CONTRACT", color="negative", border="negative", text_color="white", height=0.24),
                TextBlock(
                    text="High RCA scores are not enough if tasks are easy or reasoning paths are unaudited.",
                    font_size="caption",
                    color="text",
                ),
                Box(text="FSE'26  ->  ICLR'25  ->  OpenRCA 2.0", color="bg_alt", border="border", text_color="text_mid", height=0.22),
                Box(text="Audience payoff: when reported gains should not be trusted", color="warning", border="warning", text_color="white", height=0.24),
            ],
        )
        body = VStack(
            gap=0.20,
            children=[
                stage_bar,
                thesis,
                HStack(gap=0.24, children=[ladder, payoff]),
            ],
        )
        sb.layout(Padding(child=body, all=0.42))
        sb.notes(notes)
        sb.animate([[stage_bar, thesis], [ladder], [payoff]])
        return sb

    sb = prs.slide(title=title)

    # -------- stakes (slide 02) --------
    if kind == "stakes":
        metrics = HStack(
            gap=0.12,
            children=[
                Box(text="Downtime  $23k/min", color="negative", border="negative", text_color="white", height=0.26),
                Box(text="On-call  24/7", color="warning", border="warning", text_color="white", height=0.26),
                Box(text="Major incidents  billions", color="accent", border="accent", text_color="white", height=0.26),
                Box(text="Objective  faster MTTR", color="primary", border="primary", text_color="white", height=0.26),
            ],
        )
        right_col = VStack(
            gap=0.12,
            width=4.6,
            children=[
                Box(text="INCIDENT PRESSURE TRACK", color="bg_alt", border="border", text_color="text_mid", height=0.2),
                callout_box("MTTR pressure", "Each extra minute compounds user impact and operator load.", "negative"),
                callout_box("RCA bottleneck", "Recovery speed depends on finding trigger, not just spotting symptoms.", "primary"),
                Box(text="Verdict: RCA quality directly controls recovery velocity", color="warning", border="warning", text_color="white", height=0.24),
            ],
        )
        body = VStack(
            gap=0.18,
            children=[
                metrics,
                HStack(
                    gap=0.28,
                    children=[
                        VStack(
                            gap=0.14,
                            width=5.2,
                            children=[
                                HStack(
                                    gap=0.12,
                                    children=[
                                        SvgImage(svg=icon_warning(), width=0.72, height=0.72),
                                        Box(text="Checkout SLI degraded", color="negative", border="negative", text_color="white", height=0.26, width=2.2),
                                        arrow_text(tone="negative"),
                                        svg_service_card(title="Observed service", subtitle="loud anomaly near the user", status="propagated", badge="seen first", width=2.1, height=1.2),
                                    ],
                                ),
                                HStack(
                                    gap=0.12,
                                    children=[
                                        Spacer(width=0.70),
                                        arrow_text("↓", tone="warning", width=0.24),
                                        SvgImage(svg=icon_root_cause(), width=0.72, height=0.72),
                                        svg_service_card(title="Root-cause service", subtitle="actual trigger starts the cascade", status="root", badge="trigger", width=2.4, height=1.28),
                                    ],
                                ),
                                Box(text="Real RCA isolates the trigger, not the loudest symptom.", color="bg_alt", border="primary", text_color="primary", height=0.24),
                            ],
                        ),
                        right_col,
                    ],
                ),
            ],
        )
        sb.layout(body)
        sb.notes(notes)
        sb.animate([[metrics], [body.children[1].children[0]], [right_col]])

    elif kind == "hardness":
        header = Box(
            text="RCA is causal reconstruction: fuse modalities, trace multi-hop propagation, preserve temporal order.",
            color="primary",
            border="primary",
            text_color="white",
            height=0.28,
        )
        map_diagram = HStack(
            gap=0.24,
            children=[
                VStack(
                    gap=0.12,
                    width=3.0,
                    children=[
                        Box(text="Telemetry modalities", color="bg_alt", border="border", text_color="text", height=0.24),
                        svg_evidence_chip("metrics: onset + magnitude", "primary", "M", width=2.8),
                        svg_evidence_chip("logs: local semantic clues", "accent", "L", width=2.8),
                        svg_evidence_chip("traces: cross-service hops", "positive", "T", width=2.8),
                        Box(text="three views, one diagnosis", color="bg", border="primary", text_color="primary", height=0.24),
                    ],
                ),
                VStack(
                    gap=0.12,
                    width=6.9,
                    children=[
                        HStack(
                            gap=0.12,
                            children=[
                                Box(text="Fuse modalities", color="bg", border="primary", text_color="primary", height=0.24, width=2.05),
                                Box(text="Trace hops", color="#FFF5EA", border="accent", text_color="accent", height=0.24, width=1.9),
                                Box(text="Order events causally", color="#EFF8F2", border="positive", text_color="positive", height=0.24, width=2.45),
                            ],
                        ),
                        HStack(
                            gap=0.12,
                            children=[
                                svg_service_card(title="Root cause", subtitle="db lock", status="root", width=2.15, height=1.22),
                                arrow_text(tone="warning"),
                                svg_service_card(title="Mid service", subtitle="retry storm", status="propagated", width=2.15, height=1.22),
                                arrow_text(tone="negative"),
                                svg_service_card(title="User symptom", subtitle="timeout", status="propagated", width=2.15, height=1.22),
                            ],
                        ),
                        Box(text="must preserve propagation order", color="bg_alt", border="primary", text_color="primary", height=0.22),
                    ],
                ),
            ],
        )
        verdict = Box(text="Focus object: root-cause path under heterogeneous evidence", color="bg_alt", border="border", text_color="text_mid", height=0.22)
        sb.layout(VStack(gap=0.16, children=[header, map_diagram, verdict]))
        sb.notes(notes)
        sb.animate([[header], [map_diagram], [verdict]])

    elif kind == "shortcut_compare":
        diagram = HStack(
            gap=0.26,
            children=[
                VStack(
                    gap=0.14,
                    width=4.9,
                    children=[
                        Box(text="Shortcut ranking path", color="negative", border="negative", text_color="white", height=0.26),
                        HStack(gap=0.12, children=[Box(text="largest anomaly", color="#F3B4B4", border="negative", text_color="negative", height=0.5, width=2.1), arrow_text(tone="negative"), Box(text="rank service", color="#F6D6A8", border="warning", text_color="warning", height=0.5, width=2.0)]),
                        Box(text="high score possible, weak causal validity", color="negative", border="negative", text_color="white", height=0.28),
                    ],
                ),
                VStack(
                    gap=0.12,
                    width=4.9,
                    children=[
                        Box(text="Causal verification path", color="primary", border="primary", text_color="white", height=0.26),
                        HStack(gap=0.08, children=[svg_evidence_chip("collect metrics/logs/traces", "secondary", "E", width=2.3), arrow_text(tone="secondary"), Box(text="defensible trigger + path", color="bg", border="primary", text_color="primary", height=0.5, width=2.1)]),
                        HStack(gap=0.08, children=[svg_evidence_chip("align across dependency graph", "accent", "A", width=2.3), arrow_text(tone="accent"), Box(text="auditable alignment", color="bg", border="accent", text_color="accent", height=0.5, width=2.1)]),
                        HStack(gap=0.08, children=[svg_evidence_chip("verify propagation chain", "primary", "V", width=2.3), arrow_text(tone="primary"), Box(text="lower shortcut risk", color="primary", border="primary", text_color="white", height=0.5, width=2.1)]),
                    ],
                ),
            ],
        )
        tension = Box(
            text="Implication: high benchmark score can still encode shortcut behavior.",
            color="warning",
            border="warning",
            text_color="white",
            height=0.25,
        )
        sb.layout(VStack(gap=0.14, children=[diagram, tension]))
        sb.notes(notes)
        sb.animate([[diagram], [tension]])

    elif kind == "question_ladder":
        gate_map = VStack(
            gap=0.16,
            children=[
                HStack(
                    gap=0.18,
                    children=[
                        svg_gate_node("A", "Realism", "primary"),
                        arrow_text(tone="accent"),
                        svg_gate_node("B", "Capability", "accent"),
                        arrow_text(tone="positive"),
                        svg_gate_node("C", "Trust", "positive"),
                    ],
                ),
                HStack(
                    gap=0.18,
                    children=[
                        svg_contract_card("Q-A contract", "Benchmark must be operationally hard.", "primary", width=3.1),
                        svg_contract_card("Q-B contract", "Measure LLMs on realistic telemetry.", "accent", width=3.1),
                        svg_contract_card("Q-C contract", "Audit causal path, not label only.", "positive", width=3.1),
                    ],
                ),
            ],
        )
        dependency = Box(
            text="Methodology contract: validate task realism -> measure capability -> audit trust",
            color="warning",
            border="warning",
            text_color="white",
            height=0.24,
        )
        sb.layout(VStack(gap=0.14, children=[gate_map, dependency]))
        sb.notes(notes)
        sb.animate([[gate_map], [dependency]])

    elif kind == "takeaway_grid":
        guide = HStack(
            gap=0.20,
            children=[
                VStack(gap=0.10, width=3.2, children=[svg_gate_node("A", "Real task", "primary", width=1.6, height=1.3), insight_panel("Real task", "Look for propagation-aware failures.", "primary", fill="bg")]),
                VStack(gap=0.10, width=3.2, children=[svg_gate_node("B", "Real capability", "accent", width=1.6, height=1.3), insight_panel("Real capability", "Look for realistic-task performance gaps.", "accent", fill="bg")]),
                VStack(gap=0.10, width=3.2, children=[svg_gate_node("C", "Real trust", "positive", width=1.6, height=1.3), insight_panel("Real trust", "Look for outcome-vs-process gaps.", "positive", fill="bg")]),
            ],
        )
        sb.layout(VStack(gap=0.14, children=[guide]))
        sb.notes(notes)
        sb.animate([[guide]])

    elif kind == "probe":
        probe_board = VStack(
            gap=0.16,
            children=[
                Box(text="Probe claim: if a simple transparent heuristic reaches SOTA, benchmark discrimination is weak", color="negative", border="negative", text_color="white", height=0.26),
                HStack(
                    gap=0.18,
                    children=[
                        svg_probe_chevron("Hypothesis", "legacy benchmark is discriminative", "secondary"),
                        arrow_text(tone="warning"),
                        svg_probe_chevron("Probe", "run SimpleRCA without ML tuning", "warning"),
                        arrow_text(tone="primary"),
                        svg_probe_chevron("Decision", "compare against reported SOTA", "primary"),
                    ],
                ),
                HStack(
                    gap=0.20,
                    children=[
                        Box(text="Fairness: transparent + interpretable + low-capacity baseline", color="#FFF9EF", border="accent", text_color="accent", height=0.34, width=4.8),
                        Box(text="Falsification: if SimpleRCA clearly loses, benchmark still separates depth", color="#EEF4FA", border="primary", text_color="primary", height=0.34, width=4.8),
                    ],
                ),
            ],
        )
        criterion = Box(
            text="Falsifiable criterion: SimpleRCA parity/wins indicates low benchmark discrimination.",
            color="warning",
            border="warning",
            text_color="white",
            height=0.24,
        )
        sb.layout(VStack(gap=0.14, children=[probe_board, criterion]))
        sb.notes(notes)
        sb.animate([[probe_board], [criterion]])

    elif kind == "simplerca_chart":
        chart = grouped_bar_svg(
            ["RE2", "RE3", "Nezha", "Eadro"],
            ["SOTA", "SimpleRCA"],
            [[0.67, 0.80], [0.50, 0.83], [0.87, 0.93], [0.99, 0.81]],
            ["secondary", "primary"],
            ymax=1.05,
            width=6.4,
            height=3.2,
        )
        summary = VStack(
            gap=0.10,
            width=3.8,
            children=[
                Box(text="EMPIRICAL SHOCK", color="negative", border="negative", text_color="white", height=0.22),
                TextBlock(
                    text="SimpleRCA matches or exceeds SOTA on 3/4 public benchmarks.",
                    font_size="caption",
                    color="text",
                ),
                Box(text="3/4 wins  |  +0.33 max gap  |  Eadro exception", color="bg_alt", border="border", text_color="text_mid", height=0.22),
                Box(text="Verdict: benchmark diagnosis, not model victory", color="warning", border="warning", text_color="white", height=0.24),
            ],
        )
        take = TextBlock(text="Implication: parity by a low-capacity probe means the benchmark is structurally easy.", font_size="caption", color="text_mid")
        sb.layout(VStack(gap=0.20, children=[HStack(gap=0.30, children=[summary, chart]), take]))
        sb.notes(notes)
        sb.animate([[summary], [chart], [take]])

    elif kind == "legacy_triad":
        diag = HStack(
            gap=0.20,
            children=[
                VStack(gap=0.10, width=3.2, children=[Box(text="Injection bias", color="negative", border="negative", text_color="white", height=0.24), svg_service_card(title="Payment", subtitle="fault surface loud", status="root", style="minimal", width=1.95, height=1.2), TextBlock(text="largest anomaly -> shortcut picks source quickly", font_size="caption", color="text_mid")]),
                VStack(gap=0.10, width=3.2, children=[Box(text="Shallow propagation", color="warning", border="warning", text_color="white", height=0.24), HStack(gap=0.10, children=[svg_service_card(title="A", subtitle="source", status="root", style="minimal", width=1.35, height=1.0), arrow_text(tone="warning", width=0.2), svg_service_card(title="B", subtitle="symptom", status="propagated", style="minimal", width=1.35, height=1.0)]), TextBlock(text="1-2 hops only -> causal search depth stays low", font_size="caption", color="text_mid")]),
                VStack(gap=0.10, width=3.2, children=[Box(text="Signal dominance", color="primary", border="primary", text_color="white", height=0.24), svg_evidence_chip("metrics spike", "primary", "M", width=2.3), svg_evidence_chip("logs expose clue", "accent", "L", width=2.3), svg_evidence_chip("trace lands on culprit", "positive", "T", width=2.3)]),
            ],
        )
        note = Box(
            text="Mechanism verdict: legacy benchmark construction leaks root-cause cues.",
            color="warning",
            border="warning",
            text_color="white",
            height=0.24,
        )
        sb.layout(VStack(gap=0.14, children=[diag, note]))
        sb.notes(notes)
        sb.animate([[diag], [note]])

    elif kind == "validation_pipeline":
        flow = HStack(
            gap=0.16,
            children=[
                insight_panel("Inject faults", "31 fault types; many are operationally silent.", "secondary", width=2.2, fill="bg"),
                arrow_text(tone="secondary", width=0.24),
                insight_panel("Measure user impact", "SLI degradation required; anomaly alone is insufficient.", "accent", width=2.4, fill="#FFF6EC"),
                arrow_text(tone="warning", width=0.24),
                svg_validation_hexagon(width=2.2, height=1.6),
                arrow_text(tone="primary", width=0.24),
                svg_validated_case_hexagon(width=2.5, height=1.7),
            ],
        )
        principle = Box(
            text="Key design rule: retain only failures with measurable user-facing SLI impact.",
            color="warning",
            border="warning",
            text_color="white",
            height=0.24,
        )
        sb.layout(VStack(gap=0.14, children=[flow, principle]))
        sb.notes(notes)
        sb.animate([[flow], [principle]])

    elif kind == "framework":
        pipeline = VStack(
            gap=0.16,
            children=[
                HStack(gap=0.16, children=[svg_stage_card("1 Foundation", "TrainTicket + observability stack", "secondary"), arrow_text(tone="secondary", width=0.24), svg_stage_card("2 Workload", "Dynamic traffic creates path variety", "accent"), arrow_text(tone="warning", width=0.24), svg_stage_card("3 Injection", "31 layered fault types", "warning")]),
                HStack(gap=0.16, children=[Spacer(width=6.52), arrow_text("↓", tone="secondary", width=0.24)]),
                HStack(gap=0.16, children=[svg_stage_card("6 Annotation", "Hierarchical RCA labels + paths", "secondary"), arrow_text("←", tone="primary", width=0.24), svg_stage_card("5 Validation", "Retain only impact-validated cases", "primary", emphasized=True), arrow_text("←", tone="positive", width=0.24), svg_stage_card("4 Collection", "Metrics, logs, traces snapshot", "positive")]),
            ],
        )
        verdict = Box(
            text="Construction is reproducible: workflow-driven generation replaces hand-crafted incidents.",
            color="primary",
            border="primary",
            text_color="white",
            height=0.24,
        )
        sb.layout(VStack(gap=0.12, children=[pipeline, verdict]))
        sb.notes(notes)
        sb.animate([[pipeline], [verdict]])

    elif kind == "benchmark_stats":
        cards = VStack(
            gap=0.14,
            width=6.8,
            children=[
                HStack(gap=0.12, children=[svg_metric_stat_card("1,430", "validated cases", "primary"), svg_metric_stat_card("9,152", "fault injections", "accent"), svg_metric_stat_card("25", "fault types", "positive")]),
                HStack(gap=0.10, children=[Badge(text="6 fault families", color="secondary"), Badge(text="50 services", color="secondary"), Badge(text="dynamic workload", color="warning")]),
            ],
        )
        support = VStack(
            gap=0.12,
            width=3.0,
            children=[
                Box(text="REGIME SHIFT", color="negative", border="negative", text_color="white", height=0.22),
                TextBlock(
                    text="More services, more validated failures, and dynamic workloads change the difficulty regime.",
                    font_size="caption",
                    color="text_mid",
                ),
                Box(text="Meaning: retained incidents are operationally meaningful", color="accent", border="accent", text_color="white", height=0.22),
            ],
        )
        sb.layout(HStack(gap=0.34, children=[cards, support]))
        sb.notes(notes)
        sb.animate([[cards], [support]])

    elif kind == "collapse":
        diag = HStack(
            gap=0.24,
            children=[
                VStack(
                    gap=0.12,
                    width=4.8,
                    children=[
                        Box(text="Accuracy collapse", color="bg_alt", border="border", text_color="text", height=0.24),
                        metric_track("Avg Top@1", 0.75, "secondary", track_width=3.0, value_fmt="legacy 0.75"),
                        metric_track("Avg Top@1", 0.21, "negative", track_width=3.0, value_fmt="realistic 0.21"),
                        metric_track("Best Top@1", 0.87, "secondary", track_width=3.0, value_fmt="legacy 0.87"),
                        metric_track("Best Top@1", 0.37, "negative", track_width=3.0, value_fmt="realistic 0.37"),
                        Box(text="verdict: legacy ranking does not transfer", color="negative", border="negative", text_color="white", height=0.24),
                    ],
                ),
                VStack(
                    gap=0.12,
                    width=4.8,
                    children=[
                        Box(text="Runtime escalation", color="#FFF7F2", border="warning", text_color="text", height=0.24),
                        HStack(gap=0.10, children=[Badge(text="seconds", color="secondary"), arrow_text(tone="negative", width=0.24), Badge(text="hours", color="negative")]),
                        Box(text="~12x runtime escalation", color="negative", border="negative", text_color="white", height=0.30),
                        TextBlock(text="Realism raises operational cost sharply, so retrieval and execution efficiency become first-class evaluation concerns.", font_size="caption", color="text_mid"),
                    ],
                ),
            ],
        )
        footer = Box(
            text="Result: realism breaks both accuracy and efficiency assumptions.",
            color="negative",
            border="negative",
            text_color="white",
            height=0.24,
        )
        sb.layout(VStack(gap=0.14, children=[diag, footer]))
        sb.notes(notes)
        sb.animate([[diag], [footer]])

    elif kind == "failure_modes":
        panel = HStack(
            gap=0.18,
            children=[
                VStack(gap=0.10, width=3.2, children=[Box(text="RUNTIME", color="negative", border="negative", text_color="white", height=0.24), SvgImage(svg=icon_metrics(), width=0.9, height=0.9), TextBlock(text="toy-graph pipelines do not scale", font_size="caption", color="text_mid")]),
                VStack(gap=0.10, width=3.2, children=[Box(text="SIGNAL", color="warning", border="warning", text_color="white", height=0.24), svg_evidence_chip("metrics absent", "primary", "M", width=2.2), svg_evidence_chip("logs noisy", "accent", "L", width=2.2), svg_evidence_chip("trace sparse", "positive", "T", width=2.2)]),
                VStack(gap=0.10, width=3.2, children=[Box(text="MODEL", color="primary", border="primary", text_color="white", height=0.24), HStack(gap=0.10, children=[svg_service_card(title="Assumed", subtitle="single-hop", status="normal", width=1.45, height=1.0), arrow_text(tone="primary", width=0.20), svg_service_card(title="Reality", subtitle="cross-hop", status="propagated", width=1.45, height=1.0)]), TextBlock(text="legacy assumptions fail on realistic chains", font_size="caption", color="text_mid")]),
            ],
        )
        sb.layout(VStack(gap=0.12, children=[panel]))
        sb.notes(notes)
        sb.animate([[panel]])

    elif kind == "takeaway_a":
        headline = Box(
            text="TAKEAWAY A  |  Benchmark realism determines what progress means",
            color="primary",
            border="primary",
            text_color="white",
            height=0.3,
        )
        body = TextBlock(
            text="Without realistic failures, leaderboard gains are weak evidence. First fix the task, then measure capability.",
            font_size="subtitle",
            bold=True,
            color="primary",
            align="center",
        )
        kicker = Box(
            text="Question A answered: realism is the prerequisite for meaningful RCA evaluation.",
            color="bg_alt",
            border="border",
            text_color="text_mid",
            height=0.24,
        )
        bridge = Box(text="NEXT -> Question B: capability under real telemetry", color="accent", border="accent", text_color="white", height=0.24)
        sb.layout(VStack(gap=0.26, children=[Spacer(height=0.40), headline, body, kicker, bridge]))
        sb.notes(notes)
        sb.animate([[headline, body], [kicker, bridge]])

    elif kind == "transition_b":
        reset = section_banner(
            "Act III",
            "Question B: Can LLMs actually diagnose RCA under real telemetry?",
            tone="accent",
            recap="Realism already rules out easy benchmark wins.",
        )
        sb.layout(Padding(child=VStack(gap=0.18, children=[reset, Box(text="The next slides measure real diagnostic ability, not benchmark saturation.", color="bg_alt", border="secondary", text_color="text_mid", height=0.26)]), all=0.28))
        sb.notes(notes)
        sb.animate([[reset]])

    elif kind == "task_contract":
        pipeline = HStack(
            gap=0.20,
            children=[
                insight_panel("Operator query", "\"Why is checkout timing out?\"\nGoal anchors retrieval in time, component, and explanation.", "secondary", width=3.0, fill="bg"),
                VStack(gap=0.10, width=3.2, children=[Box(text="Telemetry bundle", color="accent", border="accent", text_color="white", height=0.24), HStack(gap=0.08, children=[SvgImage(svg=icon_metrics(), width=0.68, height=0.68), SvgImage(svg=icon_logs(), width=0.68, height=0.68), SvgImage(svg=icon_traces(), width=0.68, height=0.68)]), Box(text="68 GB context  |  multi-hop search", color="#FFF8EF", border="accent", text_color="accent", height=0.28)]),
                insight_panel("Output tuple", "(time, component, reason)\nnot a generic ranked label.", "primary", width=3.0, fill="bg_accent"),
            ],
        )
        contract = Box(
            text="Task contract: goal-conditioned retrieval -> telemetry fusion -> structured RCA tuple",
            color="primary",
            border="primary",
            text_color="white",
            height=0.26,
        )
        sb.layout(VStack(gap=0.18, children=[pipeline, contract]))
        sb.notes(notes)
        sb.animate([[pipeline], [contract]])

    elif kind == "openrca_scale":
        cards = HStack(
            gap=0.22,
            children=[
                VStack(gap=0.10, width=2.8, children=[Box(text="Scale pressure", color="negative", border="negative", text_color="white", height=0.24), stat_grid([("failure cases", "335", "primary"), ("raw telemetry", "68 GB", "negative"), ("systems", "3", "accent")], cols=1), TextBlock(text="size is part of the task", font_size="small", color="secondary", bold=True)]),
                VStack(gap=0.12, width=6.8, children=[Box(text="Telemetry mass arrives as a mixed reasoning load", color="accent", border="accent", text_color="white", height=0.24), HStack(gap=0.10, children=[svg_evidence_chip("anomaly", "primary", "M", width=1.7), svg_evidence_chip("local clues", "accent", "L", width=1.8), svg_evidence_chip("hop chain", "positive", "T", width=1.8)]), insight_panel("Heterogeneous signals", "Different granularity and different failure clues must be fused before causal search can even begin.", "accent", fill="bg"), path_track([("1", "Collect", "mixed evidence", "secondary"), ("2", "Fuse", "temporal context", "accent"), ("3", "Trace", "multi-hop cause", "negative")], arrow_tone="negative", width=5.5)]),
            ],
        )
        burden = Box(
            text="Scale burden: realistic RCA forces long-context, heterogeneous, multi-hop reasoning.",
            color="negative",
            border="negative",
            text_color="white",
            height=0.26,
        )
        sb.layout(VStack(gap=0.16, children=[cards, burden]))
        sb.notes(notes)
        sb.animate([[cards], [burden]])

    elif kind == "llm_scores":
        chart = HStack(
            gap=0.24,
            children=[
                VStack(gap=0.12, width=5.6, children=[Box(text="Verdict: even strong models remain in a low-usable RCA regime", color="negative", border="negative", text_color="white", height=0.26), metric_track("Oracle", 5.37, "secondary", max_value=12.0, track_width=4.0, value_fmt="5.37"), metric_track("Sampled", 3.88, "negative", max_value=12.0, track_width=4.0, value_fmt="3.88"), Box(text="still far from reliable", color="warning", border="warning", text_color="white", height=0.24)]),
                VStack(gap=0.12, width=4.0, children=[insight_panel("Why this matters", "Real RCA remains outside current LLM comfort zones.", "primary", fill="bg"), TextBlock(text="Focus object = the gap, not the raw score.", font_size="caption", color="text_mid")]),
            ],
        )
        sb.layout(chart)
        sb.notes(notes)
        sb.animate([[chart]])

    elif kind == "agent_gain":
        chart = VStack(
            gap=0.16,
            children=[
                Box(text="Agent helps -> ceiling remains -> residual usability gap stays open", color="accent", border="accent", text_color="white", height=0.24),
                HStack(gap=0.18, children=[insight_panel("Phase 1 · Gain", "Tool use doubles best score.", "accent", width=3.1, fill="bg"), insight_panel("Phase 2 · Ceiling", "Absolute performance still stays low.", "warning", width=3.1, fill="bg_alt"), insight_panel("Phase 3 · Residual gap", "Not yet reliable for operators.", "negative", width=3.1, fill="bg")]),
                HStack(gap=0.16, children=[metric_track("base", 5.37, "secondary", max_value=12.0, track_width=2.2, value_fmt="5.37"), metric_track("agent", 11.34, "primary", max_value=12.0, track_width=2.2, value_fmt="11.34"), Box(text="gap not closed", color="negative", border="negative", text_color="white", height=0.30, width=2.0)]),
            ],
        )
        sb.layout(chart)
        sb.notes(notes)
        sb.animate([[chart]])

    elif kind == "takeaway_b":
        panel = VStack(
            gap=0.16,
            children=[
                Box(text="Takeaway B: better tasks expose the true capability gap", color="accent", border="accent", text_color="white", height=0.42),
                HStack(gap=0.14, children=[Box(text="realistic tasks remove benchmark illusion", color="bg_alt", border="negative", text_color="negative", height=0.30, width=3.0), Circle(text="GAP", color="accent", radius=0.26), TextBlock(text="current LLM agents remain far from reliable RCA", font_size="body", bold=True, color="accent")]),
                Box(text="bridge: next we ask whether correct labels can still hide invalid reasoning paths", color="positive", border="positive", text_color="white", height=0.24),
            ],
        )
        sb.layout(Padding(child=panel, all=0.16))
        sb.notes(notes)
        sb.animate([[panel]])

    elif kind == "label_vs_reasoning":
        shallow = VStack(
            gap=0.16,
            width=5.05,
            children=[
                insight_panel(
                    "Correct label only",
                    "The model lands on the right component by following the loudest symptom, not the true causal path.",
                    "warning",
                ),
                path_track(
                    [
                        ("S", "Symptom", "largest anomaly", "warning"),
                        ("R", "Rank", "most visible service", "warning"),
                        ("L", "Label", "root guessed right", "primary"),
                    ],
                    arrow_tone="warning",
                ),
                verdict_band(
                    "Looks correct",
                    "Outcome credit is easy to over-grant when the path is never inspected.",
                    "warning",
                ),
            ],
        )
        faithful = VStack(
            gap=0.16,
            width=5.05,
            children=[
                insight_panel(
                    "Faithful reasoning",
                    "The diagnosis earns trust only after the propagation chain is reconstructed and each hop is defensible.",
                    "positive",
                ),
                path_track(
                    [
                        ("E", "Evidence", "multi-modal clues", "secondary"),
                        ("T", "Trace", "verified path", "positive"),
                        ("RC", "Root", "causal origin", "negative"),
                    ],
                    arrow_tone="positive",
                ),
                verdict_band(
                    "Actually trustworthy",
                    "Correct label plus auditable path makes the answer usable in operations.",
                    "positive",
                ),
            ],
        )
        question = Badge(text="Question C: can we verify the causal reasoning path?", color="positive")
        sb.layout(VStack(gap=0.18, children=[question, HStack(gap=0.24, children=[shallow, faithful])]))
        sb.notes(notes)
        sb.animate([[question], [shallow], [faithful]])

    elif kind == "outcome_only":
        top = insight_panel(
            "Outcome-only evaluation",
            "Checks whether the final answer matches the label. Fast and useful, but shallow.",
            "secondary",
            width=10.2,
        )
        middle = Padding(
            left=0.55,
            right=0.55,
            child=insight_panel(
                "Process-aware audit",
                "Checks whether the explanation can survive path-level scrutiny, intervention logic, and causal consistency.",
                "accent",
                fill="bg",
            ),
        )
        bottom = Padding(
            left=1.15,
            right=1.15,
            child=insight_panel(
                "Trust verdict",
                "Only this lower layer can distinguish lucky labels from faithful reasoning.",
                "positive",
                fill="bg_accent",
            ),
        )
        note = verdict_band(
            "Blind spot",
            "Outcome-only evaluation misses the process failures that matter most once RCA is delegated to an agent.",
            "warning",
        )
        sb.layout(VStack(gap=0.14, children=[top, middle, bottom, note]))
        sb.notes(notes)
        sb.animate([[top], [middle], [bottom, note]])

    elif kind == "forge_pipeline":
        gate_row = verification_loop_svg(width=9.5)
        why = callout_box(
            "Why FORGE matters",
            "Backward inference is intractable. "
            "FORGE converts it into a tractable cause-to-effect forward verification.",
            "primary",
        )
        audit = verdict_band(
            "Verification loop",
            "The trusted answer is the one whose forward causal consequences line up with the observed telemetry.",
            "positive",
        )
        sb.layout(VStack(gap=0.24, children=[gate_row, audit, why]))
        sb.notes(notes)
        sb.animate([[gate_row], [audit], [why]])

    elif kind == "stepwise_supervision":
        before = insight_panel(
            "Answer-only supervision",
            "One final label can look correct even if all intermediate hops are unsupported.",
            "warning",
            width=3.1,
        )
        after_flow = step_supervision_track_svg(width=6.9)
        upgrade = callout_box(
            "OpenRCA 2.0 upgrade",
            "500 instances now annotated with step-wise causal supervision - each propagation hop is independently verifiable.",
            "positive",
        )
        sb.layout(VStack(gap=0.18, children=[HStack(gap=0.26, children=[before, after_flow]), upgrade]))
        sb.notes(notes)
        sb.animate([[before], [after_flow], [upgrade]])

    elif kind == "metrics_panel":
        ring = VStack(
            gap=0.10,
            width=3.0,
            children=[
                Circle(text="PR", color="primary", radius=0.78),
                TextBlock(text="faithful subset", font_size="caption", color="primary", bold=True, align="center"),
                TextBlock(text="Path Reachability counts only answers whose causal chain survives verification.", font_size="caption", color="text_mid", align="center"),
            ],
        )
        metrics = VStack(
            gap=0.16,
            width=6.5,
            children=[
                insight_panel(
                    "Metric decomposition",
                    "Pass@1 asks 'did the model identify the right component?' PR asks 'did it earn that answer with a valid path?'",
                    "primary",
                    fill="bg",
                ),
                metric_track("Pass@1", 0.76, "secondary", max_value=1.0, value_fmt="0.76"),
                metric_track("Path Reachability", 0.63, "primary", max_value=1.0, value_fmt="0.63"),
                metric_track("Unfaithful residue", 0.13, "negative", max_value=0.30, value_fmt="0.13"),
            ],
        )
        note = verdict_band("Containment view", "PR is the faithful subset of correct answers; it is necessarily bounded by Pass@1.", "primary")
        sb.layout(VStack(gap=0.18, children=[HStack(gap=0.40, children=[ring, metrics]), note]))
        sb.notes(notes)
        sb.animate([[ring, metrics], [note]])

    elif kind == "best_gap":
        chart = VStack(
            gap=0.16,
            width=6.1,
            children=[
                metric_track("Pass@1", 0.76, "primary", max_value=1.0, value_fmt="0.76"),
                HStack(
                    gap=0.14,
                    children=[
                        Box(text="FRACTURE", color="negative", border="negative", text_color="white", width=1.6, height=0.28),
                        TextBlock(text="0.13 hidden process gap", font_size="caption", bold=True, color="negative"),
                    ],
                ),
                metric_track("Path Reachability", 0.63, "negative", max_value=1.0, value_fmt="0.63"),
            ],
        )
        text = VStack(
            gap=0.12,
            width=3.6,
            children=[
                Badge(text="Best model still breaks trust", color="negative"),
                callout_box(
                    "Best-model trust gap",
                    "Even the strongest model loses substantial credit once the path must be verified.",
                    "negative",
                ),
                TextBlock(
                    text="Interpretation: outcome accuracy alone overstates how trustworthy the model really is.",
                    font_size="caption",
                    color="text_mid",
                ),
            ],
        )
        sb.layout(HStack(gap=0.38, children=[chart, text]))
        sb.notes(notes)
        sb.animate([[chart], [text]])

    elif kind == "avg_gap":
        facets = Grid(
            cols=3,
            gap=0.22,
            children=[
                VStack(
                    gap=0.10,
                    children=[
                        Badge(text="Average", color="primary"),
                        metric_track("Pass@1", 0.52, "primary", max_value=1.0, track_width=1.35, value_fmt="0.52"),
                        metric_track("PR", 0.43, "negative", max_value=1.0, track_width=1.35, value_fmt="0.43"),
                    ],
                ),
                VStack(
                    gap=0.10,
                    children=[
                        Badge(text="Best", color="accent"),
                        metric_track("Pass@1", 0.76, "primary", max_value=1.0, track_width=1.35, value_fmt="0.76"),
                        metric_track("PR", 0.63, "negative", max_value=1.0, track_width=1.35, value_fmt="0.63"),
                    ],
                ),
                VStack(
                    gap=0.10,
                    children=[
                        Badge(text="Weakest", color="secondary"),
                        metric_track("Pass@1", 0.22, "primary", max_value=1.0, track_width=1.35, value_fmt="0.22"),
                        metric_track("PR", 0.18, "negative", max_value=1.0, track_width=1.35, value_fmt="0.18"),
                    ],
                ),
            ],
        )
        summary = VStack(
            gap=0.12,
            width=3.2,
            children=[
                Badge(text="Across 7 LLMs", color="negative"),
                callout_box(
                    "Population trend",
                    "Average falls from 0.52 to 0.43, and the gap persists at both ends of the model range.",
                    "negative",
                ),
                TextBlock(
                    text="Audience takeaway: process failure is a population effect, not a single-model pathology.",
                    font_size="caption",
                    color="text_mid",
                ),
            ],
        )
        sb.layout(HStack(gap=0.28, children=[facets, summary]))
        sb.notes(notes)
        sb.animate([[facets], [summary]])

    elif kind == "takeaway_c":
        trust_mark = HStack(
            gap=0.16,
            children=[
                Circle(text="C", color="positive", radius=0.34),
                VStack(
                    gap=0.04,
                    children=[
                        TextBlock(text="Trust now has a visible criterion", font_size="body", bold=True, color="positive"),
                        TextBlock(text="Reliable RCA needs causal-path faithfulness, not only correct final labels.", font_size="caption", color="text_mid"),
                    ],
                ),
            ],
        )
        claim = Box(
            text="Outcome accuracy is necessary; audited causal reasoning is what makes the diagnosis trustworthy.",
            color="bg_accent",
            border="positive",
            text_color="text",
            font_size="body",
            bold=True,
            height=0.95,
        )
        support = verdict_band(
            "Question C answered",
            "PR exposes the process failures hidden behind good labels.",
            "positive",
        )
        bridge = HStack(gap=0.12, children=[Badge(text="Next", color="primary"), TextBlock(text="Synthesize realism, capability, and trust.", font_size="caption", color="text_mid")])
        sb.layout(VStack(gap=0.22, children=[Spacer(height=0.26), trust_mark, claim, support, bridge]))
        sb.notes(notes)
        sb.animate([[trust_mark], [claim], [support, bridge]])

    elif kind == "synthesis":
        ladder = svg_synthesis_ladder(
            evidence=[
                "legacy 0.75\nvs realistic 0.21",
                "best agent\n11.34",
                "0.76 Pass@1\nvs 0.63 PR",
            ],
            width=6.3,
            height=2.7,
        )
        rail = VStack(
            gap=0.14,
            width=3.7,
            children=[
                Badge(text="Research-line synthesis", color="primary"),
                insight_panel(
                    "Dependency chain",
                    "These studies are sequenced on purpose: realism makes capability claims meaningful, and trust auditing tells us what part of that capability is actually deployable.",
                    "primary",
                    fill="bg",
                ),
                HStack(gap=0.10, children=[Badge(text="Realism", color="primary"), Badge(text="Capability", color="accent"), Badge(text="Trust", color="positive")]),
            ],
        )
        thesis = verdict_band(
            "One research line",
            "This is not a stitched paper summary; it is one evaluation program with explicit methodological dependencies.",
            "primary",
        )
        sb.layout(VStack(gap=0.20, children=[HStack(gap=0.26, children=[ladder, rail]), thesis]))
        sb.notes(notes)
        sb.animate([[ladder], [rail], [thesis]])

    elif kind == "future_agenda":
        roadmap = roadmap_track(
            [
                ("Data", "process-aware traces", "primary"),
                ("Agents", "verification-first tooling", "accent"),
                ("Training", "path-supervised reasoning", "positive"),
            ],
            width=10.1,
        )
        support = verdict_band(
            "Agenda",
            "The next step is not just stronger models; it is tighter coupling between data realism, tool-using agents, and process-aware supervision.",
            "accent",
        )
        sb.layout(VStack(gap=0.26, children=[roadmap, support]))
        sb.notes(notes)
        sb.animate([[roadmap], [support]])

    elif kind == "closing":
        close = VStack(
            gap=0.18,
            children=[
                Box(text="REALISM -> CAPABILITY -> TRUST", color="primary", border="primary", text_color="white", height=0.24),
                TextBlock(
                    text="Build Evaluations That Make Reliable AI Operations Possible",
                    font_size=26,
                    bold=True,
                    color="primary",
                ),
                TextBlock(
                    text="Evaluation quality is both the bottleneck and the lever for trustworthy RCA automation.",
                    font_size="body",
                    color="text_mid",
                ),
                HStack(
                    gap=0.18,
                    children=[
                        stage_marker("1", "Fix the task", "benchmark realism", "primary", width=2.2),
                        arrow_text(tone="accent"),
                        stage_marker("2", "Measure honestly", "capability under pressure", "accent", width=2.5),
                        arrow_text(tone="positive"),
                        stage_marker("3", "Trust carefully", "audit the causal path", "positive", width=2.4),
                    ],
                ),
                Box(text="Questions?", color="warning", border="warning", text_color="white", height=0.28, width=1.7),
            ],
        )
        sb.layout(Padding(child=close, all=0.55))
        sb.notes(notes)
        sb.animate([[close]])

    else:
        raise ValueError(f"Unknown slide kind: {kind}")

    return sb


# ---------------------------------------------------------------------------
# Backup slide builder
# ---------------------------------------------------------------------------

def build_backup_slide(prs: Presentation, slide_no: str, title: str):  # noqa: C901
    sb = prs.slide(title=title, reference=slide_no)
    notes = f"Backup slide {slide_no}: use during Q&A as needed."

    if slide_no == "B1":
        case = VStack(
            gap=0.18,
            children=[
                HStack(
                    gap=0.18,
                    children=[
                        insight_panel(
                            "Case pattern",
                            "In RE2 / RE3 the root-cause service often shows the largest anomaly spike, so ranking by magnitude is already enough.",
                            "negative",
                            width=3.3,
                            fill="bg",
                        ),
                        path_track(
                            [
                                ("A", "Anomaly", "top spike", "warning"),
                                ("R", "Rank", "pick top node", "warning"),
                                ("W", "Win", "root recovered", "primary"),
                            ],
                            arrow_tone="warning",
                            width=4.3,
                        ),
                    ],
                ),
                insight_panel(
                    "Probe meaning",
                    "SimpleRCA wins without topology or causal reasoning, so the benchmark is exposing the answer through shortcut signals.",
                    "primary",
                    fill="bg_accent",
                ),
            ],
        )
        verdict = verdict_band(
            "Shortcut mechanism",
            "Its parity with SOTA is evidence of benchmark saturation, not a case for heuristic RCA in production.",
            "warning",
        )
        sb.layout(VStack(gap=0.22, children=[case, verdict]))
        sb.notes(notes)
        sb.animate([[case], [verdict]])

    elif slide_no == "B2":
        pipeline = VStack(
            gap=0.14,
            width=5.7,
            children=[
                HStack(gap=0.12, children=[svg_stage_card("Foundation", "TrainTicket + observability", "secondary", width=1.8), arrow_text(tone="secondary", width=0.2), svg_stage_card("Workload", "Dynamic execution paths", "secondary", width=1.8), arrow_text(tone="secondary", width=0.2), svg_stage_card("Injection", "31 fault types, layered", "secondary", width=1.8)]),
                HStack(gap=0.12, children=[Spacer(width=4.92), arrow_text("↓", tone="secondary", width=0.2)]),
                HStack(gap=0.12, children=[svg_stage_card("Validation", "User-facing impact filter", "primary", emphasized=True, width=1.8), arrow_text("←", tone="primary", width=0.2), svg_stage_card("Annotation", "Hierarchical RCA labels", "secondary", width=1.8), arrow_text("←", tone="secondary", width=0.2), svg_stage_card("Collection", "Metrics + logs + traces", "secondary", width=1.8)]),
            ],
        )
        criteria = VStack(
            gap=0.14,
            width=4.1,
            children=[
                insight_panel(
                    "Validation gate",
                    "Only failures with measurable user-facing degradation are retained, removing silent injections from the dataset.",
                    "primary",
                    fill="bg",
                ),
                insight_panel(
                    "Construction discipline",
                    "Dynamic workloads plus aligned metrics, logs, and traces keep the benchmark realistic and reproducible.",
                    "accent",
                    fill="bg_alt",
                ),
                verdict_band(
                    "Why this matters",
                    "The six-stage pipeline matters because of its retention rule: it filters raw faults into diagnostically meaningful incidents.",
                    "warning",
                ),
            ],
        )
        sb.layout(HStack(gap=0.26, children=[pipeline, criteria]))
        sb.notes(notes)
        sb.animate([[pipeline], [criteria]])

    elif slide_no == "B3":
        taxonomy = triptych_panel(
            [
                ("Detect", "Decide whether an incident exists and is worth diagnosing.", "entry into RCA"),
                ("Locate", "Recover the triggering service / component under noisy telemetry.", "root-cause target"),
                ("Report", "Produce timing, reason, and impact in structured form.", "operator-facing output"),
            ],
            tones=["secondary", "accent", "primary"],
        )
        rules = VStack(
            gap=0.12,
            children=[
                insight_panel(
                    "Metric rule panel",
                    "Pass@1 captures answer correctness; PR and edge-level checks determine whether the model's causal story is valid.",
                    "accent",
                    width=3.4,
                    fill="bg",
                ),
                metric_track("Pass@1", 1.0, "secondary", max_value=1.0, value_fmt="top-1 exact"),
                metric_track("PR", 0.82, "primary", max_value=1.0, value_fmt="PR <= Pass@1"),
                metric_track("Edge precision", 0.68, "warning", max_value=1.0, value_fmt="TP/(TP+FP)"),
            ],
        )
        support = verdict_band(
            "Task granularity",
            "OpenRCA evaluates more than naming a service: it asks whether the incident can be reconstructed into a usable diagnostic report.",
            "primary",
        )
        sb.layout(VStack(gap=0.22, children=[taxonomy, HStack(gap=0.24, children=[rules, support])]))
        sb.notes(notes)
        sb.animate([[taxonomy], [rules, support]])

    elif slide_no == "B4":
        gain = path_track(
            [
                ("Q", "Query", "incident prompt", "secondary"),
                ("T", "Tooling", "filter + retrieve", "accent"),
                ("O", "Output", "structured RCA", "positive"),
            ],
            arrow_tone="accent",
            width=4.6,
        )
        failures = VStack(
            gap=0.10,
            width=4.9,
            children=[
                insight_panel("Context overflow", "68 GB telemetry can still exceed the agent's search and memory budget.", "negative", fill="bg"),
                insight_panel("Tool misuse", "Incorrect filters or ranking calls prune the evidence the agent most needs.", "warning", fill="bg_alt"),
                insight_panel("Premature stop", "The agent may stop after a partial hop and never reach the initiating component.", "secondary", fill="bg"),
            ],
        )
        support = verdict_band(
            "Agent profile",
            "Agents help because RCA is interactive, but tool use alone does not solve retrieval quality or stopping-policy failures.",
            "accent",
        )
        sb.layout(VStack(gap=0.20, children=[HStack(gap=0.24, children=[gain, failures]), support]))
        sb.notes(notes)
        sb.animate([[gain], [failures], [support]])

    elif slide_no == "B5":
        pipeline = svg_forge_pipeline(width=5.2, height=2.15)
        formulas = VStack(
            gap=0.12,
            width=4.0,
            children=[
                insight_panel(
                    "Metric math",
                    "Pass@1 asks whether the root cause is named correctly; PR asks whether the claimed cause can actually reach the symptom through a valid path.",
                    "primary",
                    fill="bg",
                ),
                metric_track("Pass@1", 1.0, "secondary", value_fmt="[0, 1]"),
                metric_track("PR", 0.82, "primary", value_fmt="[0, Pass@1]"),
                metric_track("Edge precision", 0.68, "accent", value_fmt="[0, 1]"),
                metric_track("Step coverage", 0.74, "positive", value_fmt="[0, 1]"),
            ],
        )
        note = verdict_band(
            "Key invariant",
            "PR <= Pass@1 always: a model cannot earn path faithfulness without first identifying the correct root cause.",
            "warning",
        )
        sb.layout(VStack(gap=0.18, children=[HStack(gap=0.28, children=[pipeline, formulas]), note]))
        sb.notes(notes)
        sb.animate([[pipeline], [formulas], [note]])

    elif slide_no == "B6":
        board = Grid(
            cols=2,
            gap=0.18,
            children=[
                insight_panel("Single platform", "Impact: limited stack diversity.\nMitigation: add partner datasets.", "highlight", fill="bg"),
                insight_panel("Oracle telemetry", "Impact: capability may be overstated.\nMitigation: sampled-telemetry track already added.", "warning", fill="bg"),
                insight_panel("Synthetic injection", "Impact: compound cascades remain under-covered.\nMitigation: concurrent-fault agenda.", "negative", fill="bg"),
                insight_panel("Ground truth source", "Impact: labels may diverge from operator intent.\nMitigation: human validation subset.", "primary", fill="bg"),
                insight_panel("English-only prompts", "Impact: multilingual applicability unknown.\nMitigation: orthogonal to current claim.", "secondary", fill="bg"),
                insight_panel("FORGE closed world", "Impact: verification assumes known intervention.\nMitigation: unknown-fault track planned.", "accent", fill="bg"),
            ],
        )
        footer = verdict_band(
            "Threat profile",
            "These limitations mostly define the next expansion frontier of the evaluation program rather than invalidating its current conclusions.",
            "highlight",
        )
        sb.layout(VStack(gap=0.20, children=[board, footer]))
        sb.notes(notes)
        sb.animate([[board], [footer]])

    else:
        body = VStack(gap=0.24, children=[
            callout_box("Backup use", "Use only if asked during Q&A.", "highlight"),
            Badge(text=slide_no, color="warning"),
        ])
        sb.layout(body)
        sb.notes(notes)
        sb.animate([[body]])

    return sb


# ---------------------------------------------------------------------------
# Presentation builder
# ---------------------------------------------------------------------------

def build_presentation(output_path: Path | None = None, render_preview: bool = False):
    prs = Presentation(theme=THEME)
    asset_dir = export_svg_assets()

    for spec in SLIDE_DATA:
        build_slide(prs, spec)

    for slide_no, title in BACKUP_DATA:
        build_backup_slide(prs, slide_no, title)

    save_path = output_path or OUTPUT_FILE
    prs.save(str(save_path))
    print(f"Saved: {save_path}  ({len(SLIDE_DATA)} main + {len(BACKUP_DATA)} backup slides)")
    print(f"SVG assets: {asset_dir}")

    report = prs.review()
    if report["total_issues"]:
        print(f"Layout issues: {report['total_issues']}")
        for iss in report["issues"][:10]:
            print(f"  [{iss['type']}] {iss['detail']}")
    else:
        print("Layout review: no issues.")

    if render_preview:
        preview_dir = save_path.with_suffix("").with_name(f"{save_path.stem}_preview")
        preview_dir.mkdir(parents=True, exist_ok=True)
        for png in preview_dir.glob("slide_*.png"):
            png.unlink()
        prs.preview(output_dir=str(preview_dir))
        print(f"Previews saved to: {preview_dir}")

    return prs


if __name__ == "__main__":
    build_presentation()
