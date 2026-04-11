from __future__ import annotations

import sys
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
from paperops.slides.components.svg_canvas import SvgCanvas

OUTPUT_FILE = Path(__file__).with_name("talk_4_15.pptx")

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
# Icon functions — SvgCanvas-based, theme-aware
# ---------------------------------------------------------------------------

def _draw_anomaly_burst(
    c: SvgCanvas,
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
    c: SvgCanvas,
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
    c: SvgCanvas,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    status: str = "normal",
    badge: str | None = None,
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

    if status == "propagated":
        _draw_anomaly_burst(c, cx=x + w - 18, cy=y + h / 2, tone="warning", scale=0.52)

    if status == "root":
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


def icon_metrics(size: int = 80) -> SvgCanvas:
    c = SvgCanvas(size, size, theme=THEME, bg=None)
    c.rect(8, 10, 64, 54, fill="bg_alt", stroke="border", rx=6)
    c.line(16, 54, 28, 40, color="primary", width=4)
    c.line(28, 40, 40, 46, color="primary", width=4)
    c.line(40, 46, 54, 28, color="primary", width=4)
    c.line(54, 28, 66, 34, color="primary", width=4)
    c.circle(28, 40, 3, fill="primary")
    c.circle(54, 28, 3, fill="primary")
    return c


def icon_logs(size: int = 80) -> SvgCanvas:
    c = SvgCanvas(size, size, theme=THEME, bg=None)
    c.rect(16, 10, 48, 60, fill="bg_alt", stroke="border", rx=5)
    c.line(24, 24, 56, 24, color="accent", width=4)
    c.line(24, 34, 50, 34, color="text_mid", width=3)
    c.line(24, 44, 54, 44, color="text_mid", width=3)
    c.line(24, 54, 44, 54, color="text_mid", width=3)
    return c


def icon_traces(size: int = 80) -> SvgCanvas:
    c = SvgCanvas(size, size, theme=THEME, bg=None)
    c.circle(18, 24, 8, fill="positive", opacity=0.85)
    c.circle(40, 24, 8, fill="positive", opacity=0.85)
    c.circle(62, 24, 8, fill="positive", opacity=0.85)
    c.circle(40, 54, 8, fill="positive", opacity=0.85)
    c.line(26, 24, 32, 24, color="text_mid", width=3)
    c.line(48, 24, 54, 24, color="text_mid", width=3)
    c.line(40, 32, 40, 46, color="text_mid", width=3)
    return c


def icon_service(size: int = 80) -> SvgCanvas:
    c = SvgCanvas(size, size, theme=THEME, bg=None)
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


def icon_warning(size: int = 80) -> SvgCanvas:
    c = SvgCanvas(size, size, theme=THEME, bg=None)
    _draw_anomaly_burst(c, cx=40, cy=40, tone="warning", scale=0.9)
    c.text(40, 69, "ANOM", color="warning", size=9, bold=True)
    return c


def icon_root_cause(size: int = 80) -> SvgCanvas:
    c = SvgCanvas(size, size, theme=THEME, bg=None)
    _draw_root_cause_mark(c, cx=40, cy=36, tone="negative", scale=0.95)
    c.text(40, 69, "ROOT", color="negative", size=9, bold=True)
    return c


def icon_magnifier(size: int = 80) -> SvgCanvas:
    c = SvgCanvas(size, size, theme=THEME, bg=None)
    c.path("M 16 34 A 18 18 0 1 0 52 34 A 18 18 0 1 0 16 34",
           stroke="primary", fill="none", stroke_width=5)
    c.line(47, 47, 62, 62, color="primary", width=6)
    return c


def icon_path_chain(size: int = 80) -> SvgCanvas:
    c = SvgCanvas(size, size, theme=THEME, bg=None)
    c.circle(18, 50, 7, fill="highlight")
    c.circle(40, 28, 7, fill="highlight")
    c.circle(62, 48, 7, fill="highlight")
    c.path("M 18 50 Q 28 40 40 28 Q 52 36 62 48",
           stroke="highlight", fill="none", stroke_width=4)
    return c


# ---------------------------------------------------------------------------
# Chart functions — SvgCanvas-based, high-DPI quality
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

    c = SvgCanvas(W, H, theme=THEME, bg="bg")

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

    c = SvgCanvas(W, H, theme=THEME, bg="bg")

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
# Diagram functions — SvgCanvas-based rich visuals
# ---------------------------------------------------------------------------

def svg_synthesis_ladder(
    *,
    sub_labels: list[str] | None = None,
    evidence: list[str] | None = None,
    width: float = 10.0,
    height: float = 2.0,
) -> SvgImage:
    """A/B/C three-node ladder. Core recurring identity visual."""
    has_extra = bool(sub_labels or evidence)
    W = 1100
    H = 350 if (sub_labels and evidence) else 300 if has_extra else 230
    cy_node = 108

    c = SvgCanvas(W, H, theme=THEME, bg=None)
    c.arrow_markers()

    centers_x = [180, 550, 920]
    colors = ["primary", "accent", "positive"]
    letters = ["A", "B", "C"]
    node_labels = ["Realism", "Capability", "Trust"]
    r = 62

    # Circles
    for cx_val, col, letter, lbl in zip(centers_x, colors, letters, node_labels):
        c.circle(cx_val, cy_node, r, fill=col)
        c.text(cx_val, cy_node - 16, letter, color="white", size=30, bold=True)
        c.text(cx_val, cy_node + 18, lbl, color="white", size=15)

    # Connecting arrows
    for i in range(2):
        x1 = centers_x[i] + r + 6
        x2 = centers_x[i + 1] - r - 6
        c.arrow(x1, cy_node, x2, cy_node, color=colors[i + 1], width=4)

    if sub_labels:
        for cx_val, text in zip(centers_x, sub_labels):
            c.text(cx_val, cy_node + r + 30, text, color="text_mid", size=13)

    if evidence:
        evidence_y = cy_node + r + (64 if sub_labels else 30)
        for cx_val, text in zip(centers_x, evidence):
            c.rounded_rect(
                cx_val - 74,
                evidence_y - 18,
                148,
                36,
                text=text,
                fill="bg_alt",
                stroke="border",
                text_color="text_mid",
                font_size=12,
                rx=8,
            )

    return SvgImage(svg=c, width=width, height=height)


def svg_cascade_diagram(*, width: float = 4.8, height: float = 3.25) -> SvgImage:
    """Compact symbolic RCA panel for motivation slides."""
    W, H = 560, 390
    c = SvgCanvas(W, H, theme=THEME, bg=None)
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
    c = SvgCanvas(W, H, theme=THEME, bg=None)
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
    c = SvgCanvas(W, H, theme=THEME, bg=None)
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
    c = SvgCanvas(W, H, theme=THEME, bg=None)
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
    c = SvgCanvas(W, H, theme=THEME, bg=None)

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


def _evidence_chip(c: SvgCanvas, x: float, y: float, label: str, tone: str, glyph: str) -> None:
    chip_w = max(92, len(label) * 7 + 34)
    c.rounded_rect(x, y, chip_w, 28, text="", fill="bg_alt", stroke=tone, rx=14, stroke_width=1.6)
    c.circle(x + 16, y + 14, 7, fill=tone, opacity=0.92)
    c.text(x + 16, y + 18, glyph, color="white", size=9, bold=True)
    c.text(x + 32, y + 18, label, color="text", size=12, bold=True, anchor="start")


def svg_legacy_bias_triad(*, width: float = 10.0, height: float = 3.2) -> SvgImage:
    W, H = 1080, 360
    c = SvgCanvas(W, H, theme=THEME, bg=None)
    cols = [
        (36, "Injection bias", "Root service looks obviously abnormal right after injection.", "negative"),
        (372, "Shallow propagation", "Symptoms stay near the source, so search depth stays low.", "warning"),
        (708, "Signal dominance", "Telemetry exposes the answer before multi-hop reasoning is needed.", "primary"),
    ]
    for x, title, body, tone in cols:
        c.rounded_rect(x, 56, 300, 230, text="", fill="bg_alt", stroke=tone, rx=18, stroke_width=2)
        c.rounded_rect(x + 20, 26, 126, 34, text=title, fill=tone, stroke="none",
                       text_color="white", font_size=13, bold=True, rx=16)
        if title == "Injection bias":
            _draw_service_module(c, x=x + 30, y=96, w=122, h=86, title="Payment svc", subtitle="fault surface is loud",
                                 status="root", badge="obvious")
            c.text(x + 172, 128, "largest anomaly", color="negative", size=12, bold=True, anchor="start")
            c.text(x + 172, 149, "=> shortcut ranking wins", color="text_mid", size=11, anchor="start")
        elif title == "Shallow propagation":
            _draw_service_module(c, x=x + 36, y=112, w=92, h=70, title="A", subtitle="source", status="root")
            _draw_service_module(c, x=x + 168, y=112, w=92, h=70, title="B", subtitle="symptom", status="propagated")
            c.arrow(x + 132, 147, x + 166, 147, color="warning", width=2.8)
            c.text(x + 36, 208, "1-2 hops only", color="warning", size=12, bold=True, anchor="start")
            c.text(x + 36, 228, "true cause stays visually nearby", color="text_mid", size=11, anchor="start")
        else:
            _evidence_chip(c, x + 28, 108, "Metrics spike", "primary", "M")
            _evidence_chip(c, x + 28, 146, "Logs mention exception", "accent", "L")
            _evidence_chip(c, x + 28, 184, "Trace ends at culprit", "positive", "T")
            c.text(x + 28, 228, "Answer leaks through modality cues", color="primary", size=12, bold=True, anchor="start")
            c.text(x + 28, 248, "before causal reconstruction is necessary", color="text_mid", size=11, anchor="start")
        c.text(x + 24, 312, body, color="text_mid", size=12, anchor="start")
    return SvgImage(svg=c, width=width, height=height)


def svg_validation_gate_pipeline(*, width: float = 10.0, height: float = 3.0) -> SvgImage:
    W, H = 1080, 320
    c = SvgCanvas(W, H, theme=THEME, bg=None)
    c.arrow_markers()

    c.rounded_rect(34, 104, 180, 92, text="", fill="bg_alt", stroke="secondary", rx=18, stroke_width=2)
    c.text(124, 134, "Inject faults", color="secondary", size=17, bold=True)
    c.text(124, 158, "31 fault types", color="text_mid", size=12)
    c.text(124, 176, "many are silent", color="text_mid", size=12)

    c.rounded_rect(274, 104, 188, 92, text="", fill="bg_alt", stroke="accent", rx=18, stroke_width=2)
    c.text(368, 134, "Measure impact", color="accent", size=17, bold=True)
    c.text(368, 158, "SLI / user-facing degradation", color="text_mid", size=12)
    c.text(368, 176, "not just anomaly existence", color="text_mid", size=12)

    gate = [(550, 76), (668, 76), (718, 150), (668, 224), (550, 224), (600, 150)]
    c.polygon(gate, fill="#FFF4E8", stroke=THEME.colors["warning"], stroke_width=2, opacity=1.0)
    c.text(625, 136, "Validation", color="warning", size=16, bold=True)
    c.text(625, 158, "impact > theta", color="warning", size=13, bold=True)
    c.text(625, 178, "keep only operational cases", color="text_mid", size=11)

    c.rounded_rect(820, 94, 220, 112, text="", fill="bg_accent", stroke="primary", rx=18, stroke_width=2)
    c.text(930, 130, "Validated benchmark case", color="primary", size=17, bold=True)
    c.text(930, 154, "fault -> user-visible impact", color="text_mid", size=12)
    c.text(930, 173, "worth evaluating RCA on", color="text_mid", size=12)

    c.arrow(214, 150, 274, 150, color="text_mid", width=2.8)
    c.arrow(462, 150, 548, 150, color="text_mid", width=2.8)
    c.arrow(718, 150, 818, 150, color="primary", width=3.2)

    c.path("M 634 150 C 688 222, 752 252, 842 256", stroke="warning", fill="none", stroke_width=2.2, dashed=True)
    c.rounded_rect(804, 244, 180, 34, text="discard silent faults", fill="warning", stroke="none",
                   text_color="white", font_size=12, bold=True, rx=16)
    return SvgImage(svg=c, width=width, height=height)


def svg_framework_grid_panel(*, width: float = 10.0, height: float = 3.5) -> SvgImage:
    W, H = 1080, 400
    c = SvgCanvas(W, H, theme=THEME, bg=None)
    c.arrow_markers()
    cards = [
        (40, 58, "Foundation", "TrainTicket + observability stack", "secondary"),
        (390, 58, "Workload", "Dynamic traffic to induce varied paths", "accent"),
        (740, 58, "Injection", "31 layered fault types", "warning"),
        (40, 224, "Validation", "Retain only impact-validated cases", "primary"),
        (390, 224, "Collection", "Metrics, logs, and traces snapshot", "positive"),
        (740, 224, "Annotation", "Hierarchical RCA labels and path data", "secondary"),
    ]
    for x, y, title, body, tone in cards:
        c.rounded_rect(x, y, 300, 108, text="", fill="bg_alt" if tone != "primary" else "bg_accent",
                       stroke=tone, rx=18, stroke_width=2)
        c.rounded_rect(x + 18, y + 16, 118, 26, text=title, fill=tone, stroke="none",
                       text_color="white", font_size=12, bold=True, rx=13)
        c.text(x + 26, y + 64, body, color="text", size=13, anchor="start")
    c.arrow(340, 112, 390, 112, color="text_mid", width=2.6)
    c.arrow(690, 112, 740, 112, color="text_mid", width=2.6)
    c.arrow(890, 166, 890, 224, color="text_mid", width=2.6)
    c.arrow(740, 278, 690, 278, color="positive", width=2.6)
    c.arrow(390, 278, 340, 278, color="primary", width=2.6)
    c.rounded_rect(392, 340, 296, 34, text="Scale comes from a reusable construction workflow, not hand-crafted incidents.",
                   fill="bg_alt", stroke="border", text_color="text_mid", font_size=11, rx=14)
    return SvgImage(svg=c, width=width, height=height)


def svg_benchmark_stats_panel(*, width: float = 6.6, height: float = 2.8) -> SvgImage:
    W, H = 760, 320
    c = SvgCanvas(W, H, theme=THEME, bg=None)
    stats = [
        (34, 42, "1,430", "validated cases", "primary"),
        (272, 42, "9,152", "fault injections", "accent"),
        (510, 42, "25", "fault types", "positive"),
    ]
    for x, y, value, label, tone in stats:
        c.rounded_rect(x, y, 216, 106, text="", fill="bg_alt", stroke=tone, rx=18, stroke_width=2)
        c.text(x + 108, y + 48, value, color=tone, size=28, bold=True)
        c.text(x + 108, y + 76, label.upper(), color="text_mid", size=11, bold=True)
    c.rounded_rect(48, 196, 188, 70, text="", fill="bg_accent", stroke="secondary", rx=14, stroke_width=1.8)
    c.text(142, 226, "6 fault families", color="secondary", size=14, bold=True)
    c.text(142, 246, "cross-layer coverage", color="text_mid", size=11)
    c.rounded_rect(286, 196, 188, 70, text="", fill="bg_accent", stroke="secondary", rx=14, stroke_width=1.8)
    c.text(380, 226, "50 services", color="secondary", size=14, bold=True)
    c.text(380, 246, "richer topology than legacy", color="text_mid", size=11)
    c.rounded_rect(524, 196, 188, 70, text="", fill="bg_accent", stroke="warning", rx=14, stroke_width=1.8)
    c.text(618, 226, "Dynamic workload", color="warning", size=14, bold=True)
    c.text(618, 246, "harder propagation patterns", color="text_mid", size=11)
    return SvgImage(svg=c, width=width, height=height)


def svg_dual_regime_panel(*, width: float = 10.0, height: float = 3.3) -> SvgImage:
    W, H = 1080, 360
    c = SvgCanvas(W, H, theme=THEME, bg=None)
    c.text(178, 42, "Accuracy regime shift", color="text", size=16, bold=True)
    c.text(732, 42, "Runtime regime shift", color="text", size=16, bold=True)

    def pair(x: float, y: float, old_v: float, new_v: float, title: str) -> None:
        max_h = 150
        base_y = y + 172
        for idx, (label, v, tone) in enumerate([("Legacy", old_v, "secondary"), ("Realistic", new_v, "negative")]):
            bx = x + idx * 112
            bh = max_h * v
            c.rounded_rect(bx, base_y - bh, 70, bh, text="", fill=tone, stroke="none", rx=8)
            c.text(bx + 35, base_y - bh - 12, _fmt_v(v), color="text", size=16, bold=True)
            c.text(bx + 35, base_y + 24, label, color="text_mid", size=12)
        c.text(x + 86, y + 198, title, color="text_mid", size=12)

    pair(60, 70, 0.75, 0.21, "Avg Top@1")
    pair(294, 70, 0.87, 0.37, "Best Top@1")

    c.rounded_rect(560, 88, 214, 170, text="", fill="bg_alt", stroke="warning", rx=18, stroke_width=2)
    c.text(667, 126, "Legacy", color="secondary", size=16, bold=True)
    c.text(667, 150, "seconds", color="text_mid", size=12)
    c.rounded_rect(808, 62, 214, 196, text="", fill="#FCECEC", stroke="negative", rx=18, stroke_width=2)
    c.text(915, 116, "Realistic benchmark", color="negative", size=16, bold=True)
    c.text(915, 140, "hours", color="negative", size=22, bold=True)
    c.text(915, 165, "~12x escalation", color="text_mid", size=12)
    c.arrow(774, 173, 806, 173, color="negative", width=3)

    c.rounded_rect(118, 300, 340, 34, text="Leaderboard ordering changes because the task regime changes.", fill="bg_alt",
                   stroke="border", text_color="text_mid", font_size=11, rx=15)
    c.rounded_rect(658, 300, 276, 34, text="Cost grows with realism, not just data volume.", fill="bg_alt",
                   stroke="border", text_color="text_mid", font_size=11, rx=15)
    return SvgImage(svg=c, width=width, height=height)


def svg_failure_modes_panel(*, width: float = 10.0, height: float = 3.1) -> SvgImage:
    W, H = 1080, 350
    c = SvgCanvas(W, H, theme=THEME, bg=None)
    cards = [
        (38, "Scalability limits", "Pipelines that were acceptable on toy graphs become too slow to finish.", "negative", "RUNTIME"),
        (374, "Observability blind spots", "Relevant signals are missing, conflicting, or buried across modalities.", "warning", "SIGNAL"),
        (710, "Modeling bottlenecks", "Legacy assumptions fail under realistic propagation structures.", "primary", "MODEL"),
    ]
    for x, title, body, tone, tag in cards:
        c.rounded_rect(x, 58, 300, 230, text="", fill="bg_alt", stroke=tone, rx=18, stroke_width=2)
        c.rounded_rect(x + 20, 30, 94, 30, text=tag, fill=tone, stroke="none",
                       text_color="white", font_size=11, bold=True, rx=14)
        c.text(x + 24, 100, title, color=tone, size=17, bold=True, anchor="start")
        if tag == "RUNTIME":
            c.line(x + 40, 182, x + 102, 146, color="negative", width=4)
            c.line(x + 102, 146, x + 164, 138, color="negative", width=4)
            c.line(x + 164, 138, x + 226, 92, color="negative", width=4)
        elif tag == "SIGNAL":
            _evidence_chip(c, x + 24, 136, "Metrics absent", "primary", "M")
            _evidence_chip(c, x + 24, 172, "Logs noisy", "accent", "L")
            _evidence_chip(c, x + 24, 208, "Trace sparse", "positive", "T")
        else:
            _draw_service_module(c, x=x + 24, y=132, w=116, h=78, title="Assumed", subtitle="single-hop", status="normal")
            _draw_service_module(c, x=x + 164, y=132, w=116, h=78, title="Reality", subtitle="cross-hop", status="propagated")
        c.text(x + 24, 262, body, color="text_mid", size=12, anchor="start")
    return SvgImage(svg=c, width=width, height=height)


def svg_task_contract(*, width: float = 10.0, height: float = 3.0) -> SvgImage:
    W, H = 1080, 320
    c = SvgCanvas(W, H, theme=THEME, bg=None)
    c.arrow_markers()
    c.rounded_rect(34, 80, 240, 130, text="", fill="bg_alt", stroke="secondary", rx=18, stroke_width=2)
    c.text(154, 112, "Natural-language query", color="secondary", size=16, bold=True)
    c.text(154, 140, "\"Why is checkout timing out?\"", color="text", size=13)
    c.text(154, 162, "Operator goal anchors the task", color="text_mid", size=11)

    c.rounded_rect(362, 60, 336, 170, text="", fill="bg_accent", stroke="accent", rx=20, stroke_width=2)
    c.text(530, 94, "Telemetry bundle", color="accent", size=17, bold=True)
    _evidence_chip(c, 404, 118, "metrics", "primary", "M")
    _evidence_chip(c, 404, 156, "logs", "accent", "L")
    _evidence_chip(c, 532, 118, "traces", "positive", "T")
    c.rounded_rect(512, 152, 148, 32, text="68 GB context", fill="warning", stroke="none",
                   text_color="white", font_size=12, bold=True, rx=14)
    c.text(530, 208, "Model must reason over scale, heterogeneity, and time.", color="text_mid", size=11)

    c.rounded_rect(786, 74, 260, 142, text="", fill="#F4F8FC", stroke="primary", rx=18, stroke_width=2)
    c.text(916, 108, "Structured RCA output", color="primary", size=16, bold=True)
    c.text(916, 138, "(time, component, reason)", color="text", size=14, bold=True)
    c.text(916, 162, "not just a ranked service label", color="text_mid", size=11)
    c.text(916, 184, "goal-driven task contract", color="text_mid", size=11)

    c.arrow(274, 145, 362, 145, color="text_mid", width=2.8)
    c.arrow(698, 145, 786, 145, color="primary", width=2.8)
    return SvgImage(svg=c, width=width, height=height)


def svg_scale_pressure_panel(*, width: float = 10.0, height: float = 3.1) -> SvgImage:
    W, H = 1080, 340
    c = SvgCanvas(W, H, theme=THEME, bg=None)
    cards = [
        (38, "335", "failure cases", "primary"),
        (202, "3", "enterprise systems", "accent"),
        (366, "68 GB", "raw telemetry", "negative"),
    ]
    for x, value, label, tone in cards:
        c.rounded_rect(x, 54, 142, 96, text="", fill="bg_alt", stroke=tone, rx=16, stroke_width=2)
        c.text(x + 71, 96, value, color=tone, size=24, bold=True)
        c.text(x + 71, 122, label.upper(), color="text_mid", size=10, bold=True)
    c.rounded_rect(560, 48, 474, 220, text="", fill="bg_alt", stroke="border", rx=20, stroke_width=1.6)
    c.text(797, 82, "Heterogeneous telemetry arrives together", color="text", size=16, bold=True)
    _evidence_chip(c, 596, 114, "metrics: anomaly onset", "primary", "M")
    _evidence_chip(c, 596, 154, "logs: local failure clues", "accent", "L")
    _evidence_chip(c, 596, 194, "traces: cross-service hops", "positive", "T")
    c.rounded_rect(792, 120, 192, 82, text="", fill="bg_accent", stroke="warning", rx=16, stroke_width=1.8)
    c.text(888, 148, "Reasoning burden", color="warning", size=15, bold=True)
    c.text(888, 170, "long context + noisy evidence", color="text_mid", size=11)
    c.text(888, 188, "multi-hop causal search", color="text_mid", size=11)
    c.arrow(730, 236, 852, 236, color="warning", width=2.4)
    return SvgImage(svg=c, width=width, height=height)


def svg_label_reasoning_tension(*, width: float = 9.5, height: float = 3.0) -> SvgImage:
    W, H = 1030, 330
    c = SvgCanvas(W, H, theme=THEME, bg=None)
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
    c = SvgCanvas(W, H, theme=THEME, bg=None)
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
    c = SvgCanvas(W, H, theme=THEME, bg=None)
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
    c = SvgCanvas(W, H, theme=THEME, bg=None)
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
        meta = HStack(
            gap=0.12,
            children=[
                Badge(text="Research-line talk", color="primary"),
                Badge(text="Benchmark realism", color="accent"),
                Badge(text="Trustworthy RCA", color="positive"),
            ],
        )
        thesis = VStack(
            gap=0.10,
            children=[
                TextBlock(text="Building Trustworthy RCA Evaluation for LLM Agents",
                          font_size=30, bold=True, color="primary"),
                TextBlock(
                    text="A research-line talk on benchmark realism, realistic LLM capability, and causal faithfulness.",
                    font_size="body",
                    color="text_mid",
                ),
                TextBlock(
                    text="Audience payoff: what reported RCA progress really means, and when it should not be trusted.",
                    font_size="caption",
                    color="text_mid",
                ),
            ],
        )
        payoff = VStack(
            gap=0.14,
            width=3.3,
            children=[
                callout_box(
                    "Why this talk matters",
                    "High RCA scores are not enough if the benchmark is easy or the reasoning path is invalid.",
                    "negative",
                ),
                HStack(
                    gap=0.10,
                    children=[
                        Badge(text="FSE'26", color="primary"),
                        Badge(text="ICLR'25", color="accent"),
                        Badge(text="Process trust", color="positive"),
                    ],
                ),
            ],
        )
        ladder = svg_synthesis_ladder(
            sub_labels=[
                "Is the task\noperationally real?",
                "Can agents reason\nover real telemetry?",
                "Can we audit the\ncausal path?",
            ],
            evidence=["FSE'26", "OpenRCA", "OpenRCA 2.0"],
            width=6.1,
            height=2.65,
        )
        body = VStack(
            gap=0.24,
            children=[
                meta,
                thesis,
                HStack(gap=0.24, children=[ladder, payoff]),
            ],
        )
        sb.layout(Padding(child=body, all=0.42))
        sb.notes(notes)
        sb.animate([[meta, thesis], [ladder, payoff]])
        return sb

    sb = prs.slide(title=title)

    # -------- stakes (slide 02) --------
    if kind == "stakes":
        metrics = stat_grid(
            [
                ("Downtime cost", "$23k / min", "negative"),
                ("Ops support", "24 / 7", "warning"),
                ("Incident cost", "billions", "accent"),
                ("Need", "faster MTTR", "primary"),
            ],
            cols=4,
        )
        right_col = VStack(
            gap=0.16,
            width=4.9,
            children=[
                callout_box(
                    "Why MTTR dominates operations",
                    "Every extra diagnostic minute compounds user impact while on-call engineers are still narrowing hypotheses.",
                    "negative",
                ),
                callout_box(
                    "Why RCA is the bottleneck",
                    "The issue is not just detecting an anomaly; it is isolating the triggering component before the cascade grows.",
                    "primary",
                ),
                BulletList(
                    items=[
                        "Symptoms arrive before causes become obvious",
                        "Wrong ranking burns the narrow recovery window",
                    ]
                ),
            ],
        )
        body = VStack(
            gap=0.24,
            children=[
                metrics,
                HStack(gap=0.34, children=[svg_cascade_diagram(width=4.9, height=3.7), right_col]),
                Badge(text="RCA quality is a recovery-speed lever, not just a monitoring metric.", color="warning"),
            ],
        )
        sb.layout(body)
        sb.notes(notes)
        sb.animate([[metrics], [right_col], [body.children[1].children[0]], [body.children[2]]])

    elif kind == "hardness":
        mods = modality_signal_cards()
        chain = flow_pipeline(
            ["DB deadlock\n(root cause)", "Service A\nlatency \u2191", "Service B\ntimeout", "User SLI\ndegraded"],
            ["secondary", "warning", "warning", "negative"],
        )
        burdens = triad_callout(
            [
                ("Fuse signals", "Correlate metrics, logs, and traces without over-trusting any single modality.", "primary"),
                ("Trace hops", "Symptoms may appear several services away from the injected fault.", "accent"),
                ("Order events", "The diagnostic path must respect propagation timing, not just anomaly magnitude.", "positive"),
            ]
        )
        header = callout_box(
            "Why this is intrinsically hard",
            "RCA is a causal reconstruction task: identify the trigger, follow propagation, and explain why the user-visible symptom appeared.",
            "primary",
        )
        sb.layout(VStack(gap=0.22, children=[header, mods, chain, burdens]))
        sb.notes(notes)
        sb.animate([[header], [mods], [chain], [burdens]])

    elif kind == "shortcut_compare":
        shortcut = VStack(
            gap=0.16,
            children=[
                callout_box(
                    "Shortcut path",
                    "Pick the loudest symptom and stop at the service that looks most abnormal.",
                    "negative",
                ),
                flow_pipeline(
                    ["Highest anomaly", "Most abnormal service", "Predict root cause"],
                    ["negative", "warning", "bg_alt"],
                ),
                BulletList(
                    items=[
                        "Uses symptom intensity as proxy for causality",
                        "Needs only shallow signal ranking",
                        "Looks good when benchmarks expose the source too directly",
                    ]
                ),
            ],
        )
        causal = VStack(
            gap=0.16,
            children=[
                callout_box(
                    "Causal reasoning",
                    "Verify how the failure propagates across services before naming the trigger.",
                    "primary",
                ),
                flow_pipeline(
                    ["Collect", "Align", "Verify", "Trigger"],
                    ["bg_alt", "secondary", "accent", "primary"],
                ),
                BulletList(
                    items=[
                        "Requires multi-hop dependency tracking",
                        "Must reconcile heterogeneous evidence",
                        "Produces a defensible explanation, not just a label",
                    ]
                ),
            ],
        )
        mismatch = stat_grid(
            [
                ("Risk", "shortcut rewarded", "warning"),
                ("Symptom", "high scores", "negative"),
                ("Meaning", "progress inflated", "primary"),
            ]
        )
        tension = Badge(text="When shortcut methods score well, benchmark difficulty and reported progress are misaligned.", color="warning")
        sb.layout(VStack(gap=0.20, children=[HStack(gap=0.16, children=[shortcut, causal]), mismatch, tension]))
        sb.notes(notes)
        sb.animate([[shortcut], [causal], [mismatch, tension]])

    elif kind == "question_ladder":
        ladder = svg_synthesis_ladder(
            sub_labels=[
                "Task validity",
                "Capability under\nreal telemetry",
                "Reasoning audit",
            ],
            evidence=["A gates B", "B enables C", "C makes trust visible"],
            width=5.6,
            height=2.95,
        )
        contract = VStack(
            gap=0.16,
            width=5.0,
            children=[
                callout_box("Question A", "If the benchmark is easy, later scores are not evidence of real RCA ability.", "primary"),
                callout_box("Question B", "Once the task is credible, we can ask whether LLM agents actually solve it.", "accent"),
                callout_box("Question C", "Even correct answers must be audited for causal-path faithfulness.", "positive"),
            ],
        )
        dependency = Badge(text="Methodology contract: validate the task -> measure capability -> audit trust.", color="warning")
        sb.layout(VStack(gap=0.22, children=[HStack(gap=0.36, children=[ladder, contract]), dependency]))
        sb.notes(notes)
        sb.animate([[ladder], [contract], [dependency]])

    elif kind == "takeaway_grid":
        intro = callout_box(
            "Three anchors for the rest of the talk",
            "The next evidence slides answer one question only: are we measuring a hard task, a capable agent, or a trustworthy diagnosis?",
            "primary",
        )
        grid = Grid(
            cols=3,
            gap=0.24,
            children=[
                VStack(gap=0.08, children=[Badge(text="A", color="primary"), TextBlock(text="Real task", font_size="body", bold=True, color="primary"), TextBlock(text="RCA is causal, multi-hop, and multi-modal.", font_size="caption", color="text_mid"), TextBlock(text="Watch for: propagation-aware failures.", font_size="caption", color="text")]),
                VStack(gap=0.08, children=[Badge(text="B", color="accent"), TextBlock(text="Real capability", font_size="body", bold=True, color="accent"), TextBlock(text="Reported progress matters only on realistic telemetry.", font_size="caption", color="text_mid"), TextBlock(text="Watch for: performance drop under realistic tasks.", font_size="caption", color="text")]),
                VStack(gap=0.08, children=[Badge(text="C", color="positive"), TextBlock(text="Real trust", font_size="body", bold=True, color="positive"), TextBlock(text="Correct labels are not enough without a valid path.", font_size="caption", color="text_mid"), TextBlock(text="Watch for: outcome vs process gap.", font_size="caption", color="text")]),
            ],
        )
        rule = Badge(text="Decision rule: only when A + B + C line up do we trust the progress claim.", color="warning")
        sb.layout(VStack(gap=0.22, children=[intro, grid, rule]))
        sb.notes(notes)
        sb.animate([[intro], [grid], [rule]])

    elif kind == "probe":
        claim = callout_box(
            "Probe logic",
            "SimpleRCA is a diagnostic instrument: if it approaches SOTA, the benchmark is not forcing real causal reasoning.",
            "negative",
        )
        flow = flow_pipeline(
            ["Hypothesis:\npublic benchmark is discriminative", "Probe:\ntransparent SimpleRCA", "Decision:\ncheck parity vs SOTA"],
            ["bg_alt", "warning", "primary"],
        )
        why_fair = HStack(
            gap=0.28,
            children=[
                callout_box("Why this is fair", "The probe is deliberately simple, interpretable, and zero-ML.", "accent"),
                callout_box("What would falsify it", "If SimpleRCA clearly trails SOTA, the benchmark still separates shallow and deep reasoning.", "primary"),
            ],
        )
        criterion = Badge(text="Falsifiable criterion: simple parity or wins -> the benchmark has low discrimination.", color="warning")
        sb.layout(VStack(gap=0.20, children=[claim, flow, why_fair, criterion]))
        sb.notes(notes)
        sb.animate([[claim], [flow], [why_fair], [criterion]])

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
            gap=0.18,
            width=3.8,
            children=[
                callout_box(
                    "First empirical shock",
                    "SimpleRCA wins on 3 of 4 public benchmarks, which means benchmark scores alone are not strong evidence of real RCA ability.",
                    "negative",
                ),
                stat_grid(
                    [
                        ("Wins", "3 / 4", "negative"),
                        ("Largest gap", "+0.33", "accent"),
                        ("Only resistant set", "Eadro", "secondary"),
                    ]
                ),
                BulletList(
                    items=[
                        "RE2: +0.13 over SOTA",
                        "RE3: +0.33 over SOTA",
                        "Nezha: +0.06 over SOTA",
                    ]
                ),
            ],
        )
        take = Badge(
            text="Interpretation: parity is a benchmark diagnosis, not a model victory.",
            color="warning",
        )
        sb.layout(VStack(gap=0.20, children=[HStack(gap=0.30, children=[summary, chart]), take]))
        sb.notes(notes)
        sb.animate([[summary], [chart], [take]])

    elif kind == "legacy_triad":
        diag = svg_legacy_bias_triad(width=10.1, height=3.3)
        note = Badge(text="Mechanism: shortcuts work because benchmark construction leaks the answer.", color="warning")
        sb.layout(VStack(gap=0.16, children=[diag, note]))
        sb.notes(notes)
        sb.animate([[diag], [note]])

    elif kind == "validation_pipeline":
        flow = svg_validation_gate_pipeline(width=10.1, height=3.05)
        principle = callout_box(
            "Key design decision",
            "Only failures that produce measurable user-facing SLI degradation "
            "have operational relevance for benchmarking.",
            "warning",
        )
        sb.layout(VStack(gap=0.18, children=[flow, principle]))
        sb.notes(notes)
        sb.animate([[flow], [principle]])

    elif kind == "framework":
        pipeline = svg_framework_grid_panel(width=10.15, height=3.55)
        sb.layout(Padding(child=pipeline, all=0.04))
        sb.notes(notes)
        sb.animate([[pipeline]])

    elif kind == "benchmark_stats":
        cards = svg_benchmark_stats_panel(width=6.8, height=2.8)
        support = VStack(
            gap=0.12,
            width=3.0,
            children=[
                Badge(text="Regime shift", color="negative"),
                TextBlock(
                    text="More services, more validated failures, and dynamic workloads change the difficulty regime.",
                    font_size="caption",
                    color="text_mid",
                ),
                callout_box(
                    "Why these numbers matter",
                    "Scale alone is not the point; the retained cases better reflect incidents worth diagnosing.",
                    "accent",
                ),
            ],
        )
        sb.layout(HStack(gap=0.34, children=[cards, support]))
        sb.notes(notes)
        sb.animate([[cards], [support]])

    elif kind == "collapse":
        diag = svg_dual_regime_panel(width=10.1, height=3.35)
        footer = Badge(text="Result: realism breaks both accuracy and efficiency assumptions.", color="negative")
        sb.layout(VStack(gap=0.14, children=[diag, footer]))
        sb.notes(notes)
        sb.animate([[diag], [footer]])

    elif kind == "failure_modes":
        panel = svg_failure_modes_panel(width=10.1, height=3.1)
        sb.layout(panel)
        sb.notes(notes)
        sb.animate([[panel]])

    elif kind == "takeaway_a":
        main = callout_box(
            "Takeaway A",
            "Without realistic failures, algorithmic progress claims are weak. "
            "Benchmark design determines what 'progress' means.",
            "primary",
        )
        bridge = Badge(text="Next: capability under realistic telemetry  ->", color="accent")
        kicker = TextBlock(
            text="Question A answered: realism is the prerequisite for meaningful RCA evaluation.",
            font_size="caption",
            color="text_mid",
            align="center",
        )
        sb.layout(VStack(gap=0.36, children=[Spacer(height=0.42), main, kicker, bridge]))
        sb.notes(notes)
        sb.animate([[main], [kicker, bridge]])

    elif kind == "transition_b":
        act_label = Badge(text="Question B: Capability", color="accent")
        question = TextBlock(
            text="Can LLMs Actually Diagnose RCA Under Real Telemetry?",
            font_size=24, bold=True, color="accent", align="center",
        )
        recap = RoundedBox(
            text="Realism established: new benchmark reveals realistic challenge",
            color="bg_accent", border="primary",
            text_color="primary", font_size="caption", bold=False,
            height=0.7, size_mode_x="fit",
        )
        sb.layout(Padding(
            child=VStack(gap=0.26, children=[act_label, question, recap]),
            all=0.55,
        ))
        sb.notes(notes)
        sb.animate([[act_label, question], [recap]])

    elif kind == "task_contract":
        pipeline = svg_task_contract(width=10.15, height=3.0)
        contract = callout_box(
            "Task contract",
            "RCA is framed as a goal-driven retrieval task: the model must produce "
            "a structured triple, not just a single ranked label.",
            "primary",
        )
        sb.layout(VStack(gap=0.28, children=[pipeline, contract]))
        sb.notes(notes)
        sb.animate([[pipeline], [contract]])

    elif kind == "openrca_scale":
        cards = svg_scale_pressure_panel(width=10.1, height=3.1)
        burden = callout_box(
            "Why scale matters",
            "Real telemetry is noisy, long-context, and heterogeneous - exactly the reasoning load missing from legacy benchmarks.",
            "primary",
        )
        sb.layout(VStack(gap=0.18, children=[cards, burden]))
        sb.notes(notes)
        sb.animate([[cards], [burden]])

    elif kind == "llm_scores":
        chart = bar_chart_svg(
            [("Oracle\ntelemetry", 5.37, "secondary"), ("Sampled\ntelemetry", 3.88, "negative")],
            ymax=12.0, width=5.0, height=2.8,
        )
        label = VStack(
            gap=0.12,
            width=3.6,
            children=[
                Badge(text="Still near the floor", color="negative"),
                callout_box(
                    "Capability gap",
                    "Even with oracle telemetry, the best score is only 5.37. Sampling hurts further.",
                    "negative",
                ),
                TextBlock(
                    text="Interpretation: realistic RCA remains far outside current frontier-model comfort zones.",
                    font_size="caption",
                    color="text_mid",
                ),
            ],
        )
        sb.layout(HStack(gap=0.38, children=[chart, label]))
        sb.notes(notes)
        sb.animate([[chart], [label]])

    elif kind == "agent_gain":
        chart = bar_chart_svg(
            [("Base LLM\n(best)", 5.37, "secondary"), ("RCA-agent\n(best)", 11.34, "primary")],
            ymax=15.0,
            width=5.0,
            height=2.8,
            threshold=11.34,
            threshold_label="still far from reliable",
        )
        gap = VStack(
            gap=0.12,
            width=3.6,
            children=[
                Badge(text="Agent helps, gap remains", color="accent"),
                callout_box(
                    "Interpretation",
                    "Tool use roughly doubles best performance, but the absolute level is still low.",
                    "accent",
                ),
                TextBlock(
                    text="Useful takeaway: execution support matters, yet evaluation still reveals a large usability gap.",
                    font_size="caption",
                    color="text_mid",
                ),
            ],
        )
        sb.layout(HStack(gap=0.38, children=[chart, gap]))
        sb.notes(notes)
        sb.animate([[chart], [gap]])

    elif kind == "takeaway_b":
        main = callout_box(
            "Takeaway B",
            "Realistic tasks expose a substantial diagnostic gap. "
            "Current LLM agents are far from reliable RCA performance.",
            "accent",
        )
        bridge = Badge(text="Next: trustworthiness beyond outcome labels  ->", color="positive")
        kicker = TextBlock(
            text="Question B answered: stronger tasks reveal a real capability deficit, not just leaderboard reshuffling.",
            font_size="caption",
            color="text_mid",
            align="center",
        )
        sb.layout(VStack(gap=0.36, children=[Spacer(height=0.42), main, kicker, bridge]))
        sb.notes(notes)
        sb.animate([[main], [kicker, bridge]])

    elif kind == "label_vs_reasoning":
        diagram = svg_label_reasoning_tension(width=9.65, height=3.0)
        question = Badge(text="Question C: can we verify the causal reasoning path?", color="positive")
        sb.layout(VStack(gap=0.18, children=[diagram, question]))
        sb.notes(notes)
        sb.animate([[diagram], [question]])

    elif kind == "outcome_only":
        diagram = svg_outcome_blind_spot(width=9.65, height=3.0)
        note = callout_box(
            "The blind spot",
            "Outcome-only evaluation does not penalise correct answers "
            "reached via spurious or unsupported reasoning paths.",
            "warning",
        )
        sb.layout(VStack(gap=0.18, children=[diagram, note]))
        sb.notes(notes)
        sb.animate([[diagram], [note]])

    elif kind == "forge_pipeline":
        pipeline = svg_forge_pipeline(width=9.5, height=2.5)
        why = callout_box(
            "Why FORGE matters",
            "Backward inference is intractable. "
            "FORGE converts it into a tractable cause-to-effect forward verification.",
            "primary",
        )
        sb.layout(VStack(gap=0.24, children=[pipeline, why]))
        sb.notes(notes)
        sb.animate([[pipeline], [why]])

    elif kind == "stepwise_supervision":
        after_flow = svg_stepwise_supervision_upgrade(width=9.65, height=3.1)
        upgrade = callout_box(
            "OpenRCA 2.0 upgrade",
            "500 instances now annotated with step-wise causal supervision - each propagation hop is independently verifiable.",
            "positive",
        )
        sb.layout(VStack(gap=0.18, children=[after_flow, upgrade]))
        sb.notes(notes)
        sb.animate([[after_flow], [upgrade]])

    elif kind == "metrics_panel":
        diagram = svg_metric_semantics(width=8.2, height=3.2)
        note = Badge(text="Containment view: PR is the faithful subset of correct answers.", color="primary")
        sb.layout(VStack(gap=0.16, children=[diagram, note]))
        sb.notes(notes)
        sb.animate([[diagram], [note]])

    elif kind == "best_gap":
        chart = bar_chart_svg(
            [("Pass@1", 0.76, "primary"), ("Path Reachability", 0.63, "negative")],
            ymax=1.0, width=5.0, height=2.8,
        )
        text = VStack(
            gap=0.12,
            width=3.6,
            children=[
                Badge(text="0.13 hidden gap", color="negative"),
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
        chart = grouped_bar_svg(
            ["Average", "Best", "Weakest"],
            ["Pass@1", "Path Reachability"],
            [[0.52, 0.43], [0.76, 0.63], [0.22, 0.18]],
            ["primary", "negative"],
            ymax=1.0,
            width=6.1,
            height=3.0,
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
        sb.layout(HStack(gap=0.28, children=[chart, summary]))
        sb.notes(notes)
        sb.animate([[chart], [summary]])

    elif kind == "takeaway_c":
        main = callout_box(
            "Takeaway C",
            "Trustworthy RCA requires both correct outcomes and verifiable causal-path faithfulness. "
            "PR <= Pass@1 always - outcome accuracy alone is insufficient.",
            "positive",
        )
        bridge = Badge(text="Synthesizing all three questions  ->", color="primary")
        kicker = TextBlock(
            text="Question C answered: reliable RCA demands auditable process, not just correct final labels.",
            font_size="caption",
            color="text_mid",
            align="center",
        )
        sb.layout(VStack(gap=0.36, children=[Spacer(height=0.42), main, kicker, bridge]))
        sb.notes(notes)
        sb.animate([[main], [kicker, bridge]])

    elif kind == "synthesis":
        ladder = svg_synthesis_ladder(
            evidence=[
                "avg 0.21 vs 0.75\non legacy",
                "best 11.34\nfar from reliable",
                "0.76 Pass@1\nvs 0.63 PR",
            ],
            width=10.2,
            height=2.5,
        )
        thesis = callout_box(
            "One research line",
            "The three studies form one coherent evaluation program: "
            "fix the task first, measure capability honestly, then verify the reasoning chain.",
            "primary",
        )
        sb.layout(VStack(gap=0.20, children=[ladder, thesis]))
        sb.notes(notes)
        sb.animate([[ladder], [thesis]])

    elif kind == "future_agenda":
        grid = svg_future_pillars(width=10.1, height=3.0)
        sb.layout(grid)
        sb.notes(notes)
        sb.animate([[grid]])

    elif kind == "closing":
        close = VStack(
            gap=0.20,
            children=[
                Badge(text="Realism  ->  Capability  ->  Trust", color="primary"),
                TextBlock(
                    text="Build Evaluations That Make Reliable AI Operations Possible",
                    font_size=26, bold=True, color="primary",
                ),
                TextBlock(
                    text="Evaluation quality is both the bottleneck and the lever "
                         "for trustworthy RCA automation.",
                    font_size="body", color="text_mid",
                ),
                Badge(text="Questions?", color="accent"),
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
        # Why SimpleRCA wins
        left = callout_box(
            "Shortcut mechanism",
            "On RE2 and RE3, the injected faults produce large anomaly spikes "
            "in the root-cause service's metrics. "
            "SimpleRCA wins by picking the highest-anomaly node — no topology needed.",
            "negative",
        )
        right = VStack(gap=0.14, children=[
            callout_box("Fair probe?",
                        "Yes — SimpleRCA uses only anomaly scores, no domain knowledge. "
                        "Its parity with SOTA is the evidence of benchmark saturation.",
                        "primary"),
            Badge(text="Conclusion: benchmarks that reward heuristics need redesign", color="warning"),
        ])
        sb.layout(HStack(gap=0.40, children=[left, right]))
        sb.notes(notes)
        sb.animate([[left], [right]])

    elif slide_no == "B2":
        # Benchmark construction details
        pipeline = svg_benchmark_pipeline(width=9.8, height=3.4)
        table = Table(
            headers=["Stage", "Key design choice", "Why it matters"],
            rows=[
                ["Foundation", "TrainTicket + open-telemetry stack",    "Reproducible, real-world topology"],
                ["Workload",   "State-machine dynamic traffic",          "Avoids static injection artifacts"],
                ["Injection",  "31 fault types, stratified sampling",    "Covers diverse failure modes"],
                ["Collection", "Metrics + logs + traces aligned",        "Heterogeneous telemetry fidelity"],
                ["Annotation", "Hierarchical service \u2192 instance labels", "Supports multi-granularity eval"],
                ["Validation", "SLI degradation threshold \u03b8=5%",   "Discards silent / unobservable faults"],
            ],
            header_color="primary",
        )
        sb.layout(VStack(gap=0.20, children=[pipeline, table]))
        sb.notes(notes)
        sb.animate([[pipeline], [table]])

    elif slide_no == "B3":
        # OpenRCA task granularity
        left = VStack(gap=0.16, children=[
            TextBlock(text="Task decomposition (7 sub-tasks)", font_size="body", bold=True, color="primary"),
            BulletList(
                items=[
                    "Anomaly detection (is anything wrong?)",
                    "Root-cause localisation (which service?)",
                    "Root-cause identification (which component?)",
                    "Failure time estimation",
                    "Failure reason classification",
                    "Impact scope estimation",
                    "End-to-end structured report generation",
                ],
                font_size="caption",
            ),
        ])
        right = Table(
            headers=["Metric", "Measures", "Formula"],
            rows=[
                ["Pass@1",             "Correct root cause named",      "top-1 exact match"],
                ["Path Reachability",   "Valid causal path exists",      "PR \u2264 Pass@1"],
                ["Edge Precision",      "Fraction of valid causal edges","TP / (TP+FP)"],
                ["Time Accuracy",       "Failure time within \u00b15 min","binary"],
                ["F1-Composite",        "Weighted combination",          "see paper"],
            ],
            header_color="accent",
        )
        sb.layout(HStack(gap=0.38, children=[left, right]))
        sb.notes(notes)
        sb.animate([[left], [right]])

    elif slide_no == "B4":
        # RCA-agent details
        agent_flow = flow_pipeline(
            ["NL Query\n+ telemetry", "Tool calls:\nfilter / rank", "Hypothesis\nrevision", "Structured\noutput"],
            ["bg_alt", "accent", "accent", "primary"],
        )
        failures = triad_callout([
            ("Context overflow",
             "68 GB telemetry exceeds context window; retrieval errors cascade.", "negative"),
            ("Tool misuse",
             "Agent issues incorrect filter parameters, pruning relevant evidence.", "warning"),
            ("Premature termination",
             "Agent stops after partial hop, missing the real root-cause service.", "secondary"),
        ])
        sb.layout(VStack(gap=0.28, children=[agent_flow, failures]))
        sb.notes(notes)
        sb.animate([[agent_flow], [failures]])

    elif slide_no == "B5":
        # FORGE details + metrics
        pipeline = svg_forge_pipeline(width=9.5, height=2.4)
        table = Table(
            headers=["Metric", "Definition", "Range"],
            rows=[
                ["Pass@1",              "Root cause named correctly (outcome)",          "[0, 1]"],
                ["Path Reachability",   "Claimed cause \u2192 symptom path is valid",    "[0, Pass@1]"],
                ["Edge Precision",      "Fraction of predicted edges that are correct",  "[0, 1]"],
                ["Step Coverage",       "Fraction of ground-truth hops recovered",       "[0, 1]"],
            ],
            header_color="primary",
        )
        note = TextBlock(
            text="PR \u2264 Pass@1 always: a model cannot have a valid causal path "
                 "without first naming the correct root cause.",
            font_size="caption", color="text_mid", italic=True,
        )
        sb.layout(VStack(gap=0.18, children=[pipeline, table, note]))
        sb.notes(notes)
        sb.animate([[pipeline], [table], [note]])

    elif slide_no == "B6":
        # Threats to validity
        table = Table(
            headers=["Threat", "Expected impact", "Mitigation direction"],
            rows=[
                ["Single platform (TrainTicket)",
                 "May not generalise to proprietary stacks",
                 "Add industry partner datasets"],
                ["Oracle telemetry assumption",
                 "Over-estimates agent capability",
                 "Sampled-telemetry track already added"],
                ["Synthetic fault injection",
                 "Miss compound / cascading faults",
                 "Concurrent-fault injection in agenda"],
                ["Ground-truth from injector",
                 "Annotation may not match operator intent",
                 "Human validation on subset"],
                ["English-only task prompts",
                 "Limits multilingual applicability",
                 "Out of scope; language is orthogonal"],
                ["FORGE closed-world assumption",
                 "Validation requires known injection point",
                 "Unknown-fault track planned for v3"],
            ],
            header_color="highlight",
        )
        sb.layout(table)
        sb.notes(notes)
        sb.animate([[table]])

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

    for spec in SLIDE_DATA:
        build_slide(prs, spec)

    for slide_no, title in BACKUP_DATA:
        build_backup_slide(prs, slide_no, title)

    save_path = output_path or OUTPUT_FILE
    prs.save(str(save_path))
    print(f"Saved: {save_path}  ({len(SLIDE_DATA)} main + {len(BACKUP_DATA)} backup slides)")

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
