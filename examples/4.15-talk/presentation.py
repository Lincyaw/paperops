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
        panel = triad_callout([
            ("Injection bias",
             "Injected faults make the root service look obviously abnormal.", "negative"),
            ("Shallow propagation",
             "Call graphs are short — symptoms stay near the source.", "warning"),
            ("Signal dominance",
             "Telemetry patterns expose the answer before reasoning is needed.", "primary"),
        ])
        sb.layout(panel)
        sb.notes(notes)
        sb.animate([[panel]])

    elif kind == "validation_pipeline":
        flow = flow_pipeline(
            ["Inject faults\n(31 types)", "Monitor SLI\nimpact", "Filter:\nimpact > \u03b8?", "Validated\nfailures"],
            ["secondary", "bg_alt", "warning", "primary"],
        )
        discard = Badge(text="Silent faults discarded  \u2191", color="warning")
        principle = callout_box(
            "Key design decision",
            "Only failures that produce measurable user-facing SLI degradation "
            "have operational relevance for benchmarking.",
            "warning",
        )
        sb.layout(VStack(gap=0.22, children=[flow, discard, principle]))
        sb.notes(notes)
        sb.animate([[flow], [discard], [principle]])

    elif kind == "framework":
        pipeline = svg_benchmark_pipeline(width=10.2, height=3.8)
        sb.layout(Padding(child=pipeline, all=0.05))
        sb.notes(notes)
        sb.animate([[pipeline]])

    elif kind == "benchmark_stats":
        cards = Grid(
            cols=3,
            gap=0.26,
            children=[
                metric_card("Validated cases",  "1,430",   "primary"),
                metric_card("Fault injections", "9,152",   "accent"),
                metric_card("Fault types",      "25",      "positive"),
                metric_card("Categories",       "6",       "secondary"),
                metric_card("Services",         "50",      "secondary"),
                metric_card("Workload",         "Dynamic", "warning"),
            ],
        )
        takeaway = callout_box(
            "Interpretation",
            "The new benchmark sits in a structurally different difficulty regime — "
            "not just larger, but operationally harder.",
            "primary",
        )
        sb.layout(VStack(gap=0.28, children=[cards, takeaway]))
        sb.notes(notes)
        sb.animate([[cards], [takeaway]])

    elif kind == "collapse":
        perf_chart = grouped_bar_svg(
            ["Legacy benchmarks", "New benchmark"],
            ["Avg Top@1", "Best Top@1"],
            [[0.75, 0.87], [0.21, 0.37]],
            ["secondary", "primary"],
            ymax=1.0,
            width=5.6,
            height=2.9,
        )
        runtime_box = VStack(gap=0.14, children=[
            callout_box("Runtime escalation",
                        "Methods that ran in seconds on legacy benchmarks now require hours "
                        "on realistic cases — a 12\u00d7 order-of-magnitude increase.",
                        "negative"),
            Badge(text="Leaderboard confidence is fragile", color="negative"),
        ])
        sb.layout(HStack(gap=0.36, children=[perf_chart, runtime_box]))
        sb.notes(notes)
        sb.animate([[perf_chart], [runtime_box]])

    elif kind == "failure_modes":
        panel = triad_callout([
            ("Scalability limits",
             "Runtime grows from seconds to hours; methods cannot finish on large graphs.", "negative"),
            ("Observability blind spots",
             "Signals disappear, conflict, or never surface at the relevant service layer.", "warning"),
            ("Modeling bottlenecks",
             "Method assumptions break under realistic fault propagation patterns.", "primary"),
        ])
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
        bridge = Badge(text="Next: capability under realistic telemetry  \u2192", color="accent")
        sb.layout(VStack(gap=0.5, children=[Spacer(height=0.4), main, Spacer(height=0.3), bridge]))
        sb.notes(notes)
        sb.animate([[main], [bridge]])

    elif kind == "transition_b":
        act_label = Badge(text="Question B: Capability", color="accent")
        question = TextBlock(
            text="Can LLMs Actually Diagnose RCA Under Real Telemetry?",
            font_size=24, bold=True, color="accent", align="center",
        )
        recap = RoundedBox(
            text="\u2713 Realism established: new benchmark reveals realistic challenge",
            color="bg_accent", border="primary",
            text_color="primary", font_size="caption", bold=False,
            height=0.7, size_mode_x="fit",
        )
        sb.layout(Padding(
            child=VStack(gap=0.32, children=[act_label, question, recap]),
            all=0.55,
        ))
        sb.notes(notes)
        sb.animate([[act_label, question], [recap]])

    elif kind == "task_contract":
        pipeline = flow_pipeline(
            ["Natural-language\nQuery", "Telemetry\nAnalysis (68 GB)", "Time \u00b7 Component\n\u00b7 Reason"],
            ["bg_alt", "accent", "primary"],
        )
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
        cards = stat_grid([
            ("Failure cases", "335", "primary"),
            ("Enterprise systems", "3",    "accent"),
            ("Raw telemetry", "68 GB",  "negative"),
        ])
        mods = modality_triptych()
        burden = callout_box(
            "Why scale matters",
            "Real telemetry is noisy, long-context, and heterogeneous — "
            "exactly the reasoning load missing from legacy benchmarks.",
            "primary",
        )
        sb.layout(VStack(gap=0.24, children=[cards, mods, burden]))
        sb.notes(notes)
        sb.animate([[cards], [mods], [burden]])

    elif kind == "llm_scores":
        chart = bar_chart_svg(
            [("Oracle\ntelemetry", 5.37, "secondary"), ("Sampled\ntelemetry", 3.88, "negative")],
            ymax=12.0, width=5.0, height=2.8,
        )
        label = callout_box(
            "Capability gap",
            "Current frontier models solve only the easiest fractions of realistic RCA tasks. "
            "Score of 5.37 out of a possible ~100 (oracle setting) shows a large gap.",
            "negative",
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
        gap = callout_box(
            "Interpretation",
            "Tool-augmented execution improves best result to 11.34, "
            "but the distance to practical reliability remains very large.",
            "accent",
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
        bridge = Badge(text="Next: trustworthiness beyond outcome labels  \u2192", color="positive")
        sb.layout(VStack(gap=0.5, children=[Spacer(height=0.4), main, Spacer(height=0.3), bridge]))
        sb.notes(notes)
        sb.animate([[main], [bridge]])

    elif kind == "label_vs_reasoning":
        diagram = svg_process_comparison(width=9.5, height=2.9)
        question = Badge(text="Question C: can we verify the causal reasoning path?", color="positive")
        sb.layout(VStack(gap=0.18, children=[diagram, question]))
        sb.notes(notes)
        sb.animate([[diagram], [question]])

    elif kind == "outcome_only":
        diagram = svg_process_comparison(width=9.5, height=2.8)
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
        before_row = HStack(gap=0.12, children=[
            Badge(text="Old: outcome label only", color="warning"),
            TextBlock(text="Answer right or wrong, nothing else verified.", font_size="caption", color="text_mid"),
        ])
        after_flow = flow_pipeline(
            ["Root Cause", "Propagation\nStep 1", "Propagation\nStep 2", "Observed\nSymptom"],
            ["secondary", "accent", "accent", "primary"],
        )
        upgrade = callout_box(
            "OpenRCA 2.0 upgrade",
            "500 instances now annotated with step-wise causal supervision — "
            "each propagation hop is independently verifiable.",
            "positive",
        )
        sb.layout(VStack(gap=0.22, children=[before_row, after_flow, upgrade]))
        sb.notes(notes)
        sb.animate([[before_row], [after_flow], [upgrade]])

    elif kind == "metrics_panel":
        diagram = svg_metric_semantics(width=8.2, height=3.2)
        sb.layout(Padding(child=diagram, all=0.05))
        sb.notes(notes)
        sb.animate([[diagram]])

    elif kind == "best_gap":
        chart = bar_chart_svg(
            [("Pass@1", 0.76, "primary"), ("Path Reachability", 0.63, "negative")],
            ymax=1.0, width=5.0, height=2.8,
        )
        text = callout_box(
            "Best-model trust gap",
            "Even the strongest model drops 0.13 from Pass@1 to PR. "
            "One in eight correct diagnoses is built on an unsupported reasoning path.",
            "negative",
        )
        sb.layout(HStack(gap=0.38, children=[chart, text]))
        sb.notes(notes)
        sb.animate([[chart], [text]])

    elif kind == "avg_gap":
        # Per-model data: 7 LLMs showing Pass@1 vs PR
        chart = grouped_bar_svg(
            ["M1", "M2", "M3", "M4", "M5", "M6", "M7"],
            ["Pass@1", "Path Reachability"],
            [[0.76, 0.63], [0.68, 0.56], [0.60, 0.50],
             [0.55, 0.45], [0.45, 0.37], [0.38, 0.31], [0.22, 0.18]],
            ["primary", "negative"],
            ymax=1.0,
            width=6.8,
            height=3.1,
        )
        summary = callout_box(
            "Population trend",
            "Average drops from Pass@1 0.52 to PR 0.43 across 7 LLMs. "
            "The trust gap is systematic, not an isolated failure.",
            "negative",
        )
        sb.layout(HStack(gap=0.28, children=[chart, summary]))
        sb.notes(notes)
        sb.animate([[chart], [summary]])

    elif kind == "takeaway_c":
        main = callout_box(
            "Takeaway C",
            "Trustworthy RCA requires both correct outcomes and verifiable causal-path faithfulness. "
            "PR \u2264 Pass@1 always — outcome accuracy alone is insufficient.",
            "positive",
        )
        bridge = Badge(text="Synthesizing all three questions  \u2192", color="primary")
        sb.layout(VStack(gap=0.5, children=[Spacer(height=0.4), main, Spacer(height=0.3), bridge]))
        sb.notes(notes)
        sb.animate([[main], [bridge]])

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
        sb.layout(VStack(gap=0.28, children=[ladder, thesis]))
        sb.notes(notes)
        sb.animate([[ladder], [thesis]])

    elif kind == "future_agenda":
        grid = triad_callout([
            ("Data", "Harder incidents, concurrent faults, partial observability.", "primary"),
            ("Agents", "Better retrieval, hypothesis revision, and multi-step tool use.", "accent"),
            ("Training", "Process-aware supervision and causal-reasoning diagnostics.", "positive"),
        ])
        sb.layout(grid)
        sb.notes(notes)
        sb.animate([[grid]])

    elif kind == "closing":
        close = VStack(
            gap=0.20,
            children=[
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
