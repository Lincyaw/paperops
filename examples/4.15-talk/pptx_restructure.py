from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import math
import textwrap

from lxml import etree
from PIL import Image, ImageDraw, ImageFont, ImageOps

SLIDE_W = 1920
SLIDE_H = 1080
MARGIN_X = 92
TOP_Y = 72
TITLE_Y = 86
CONTENT_TOP = 190

NAVY = '#23435b'
INK = '#24313d'
MUTED = '#6b7480'
LIGHT = '#eef2f4'
CREAM = '#f7f3eb'
ORANGE = '#c97936'
ORANGE_DARK = '#a85f26'
GREEN = '#4e8b73'
GREEN_DARK = '#2f6e56'
RED = '#b85c5b'
GOLD = '#d8a647'
BLUE = '#5687a6'
PALE_BLUE = '#dfeaf1'
PALE_GREEN = '#dfece6'
PALE_ORANGE = '#f7e7d8'
PALE_RED = '#f4dfde'
WHITE = '#ffffff'
LINE = '#d3d9de'

FONT_REG = '/usr/share/fonts/google-noto/NotoSans-Regular.ttf'
FONT_BOLD = '/usr/share/fonts/google-noto/NotoSans-Bold.ttf'

P_NS = 'http://schemas.openxmlformats.org/presentationml/2006/main'
A_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG_REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'
NSMAP = {'p': P_NS, 'a': A_NS, 'r': R_NS}


@dataclass
class DeckPaths:
    unpacked: Path
    media: Path
    slides: Path
    rels: Path


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def new_slide() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new('RGB', (SLIDE_W, SLIDE_H), WHITE)
    draw = ImageDraw.Draw(img)
    return img, draw


def top_frame(draw: ImageDraw.ImageDraw, title: str, accent: str = ORANGE, kicker: str | None = None) -> None:
    draw.rectangle([0, 0, SLIDE_W, 18], fill=accent)
    draw.text((MARGIN_X, TITLE_Y), title, font=font(34, True), fill=INK)
    draw.line((MARGIN_X, 150, MARGIN_X + 180, 150), fill=accent, width=6)
    if kicker:
        pill(draw, SLIDE_W - 340, 78, 250, 38, kicker, accent, WHITE, 19)


def pill(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, text: str, fill: str, txt: str, size: int = 20) -> None:
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=fill)
    centered_text(draw, (x, y, x + w, y + h), text, font(size, True), txt)


def panel(draw: ImageDraw.ImageDraw, box, fill=WHITE, outline=LINE, width=2) -> None:
    draw.rounded_rectangle(box, radius=20, fill=fill, outline=outline, width=width)


def strong_box(draw: ImageDraw.ImageDraw, box, accent=ORANGE, fill=WHITE) -> None:
    panel(draw, box, fill=fill, outline=accent, width=4)


def band(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, text: str, fill: str, txt: str = WHITE, size: int = 22) -> None:
    draw.rectangle([x, y, x + w, y + h], fill=fill)
    draw.text((x + 18, y + 10), text, font=font(size, True), fill=txt)


def wrap(draw: ImageDraw.ImageDraw, text: str, box_w: int, fnt: ImageFont.FreeTypeFont) -> str:
    words = text.split()
    lines: list[str] = []
    cur = ''
    for word in words:
        test = word if not cur else cur + ' ' + word
        if draw.textlength(test, font=fnt) <= box_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return '\n'.join(lines)


def paragraph(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, text: str, size: int = 23, fill: str = INK, bold: bool = False, leading: int = 10) -> int:
    fnt = font(size, bold)
    wrapped = wrap(draw, text, w, fnt)
    draw.multiline_text((x, y), wrapped, font=fnt, fill=fill, spacing=leading)
    bbox = draw.multiline_textbbox((x, y), wrapped, font=fnt, spacing=leading)
    return bbox[3]


def centered_text(draw: ImageDraw.ImageDraw, box, text: str, fnt, fill: str) -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=6)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x1, y1, x2, y2 = box
    draw.multiline_text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2), text, font=fnt, fill=fill, align='center', spacing=6)


def metric_stack(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, items: list[tuple[str, str]], accent=ORANGE) -> None:
    h = 136
    gap = 20
    for i, (label, value) in enumerate(items):
        yy = y + i * (h + gap)
        panel(draw, [x, yy, x + w, yy + h], fill=CREAM, outline='#e6ddd2')
        draw.rectangle([x, yy, x + 18, yy + h], fill=accent if i == 0 else (GREEN if i == 1 else NAVY))
        draw.text((x + 38, yy + 28), label, font=font(22, True), fill=MUTED)
        draw.text((x + 38, yy + 60), value, font=font(42, True), fill=INK)


def paste_contain(base: Image.Image, img_path: Path, box, pad=0, bg=None):
    x1, y1, x2, y2 = box
    area = (max(1, x2 - x1 - pad * 2), max(1, y2 - y1 - pad * 2))
    src = Image.open(img_path).convert('RGBA')
    fitted = ImageOps.contain(src, area)
    if bg is not None:
        bg_img = Image.new('RGBA', area, bg)
        bg_img.alpha_composite(fitted, ((area[0] - fitted.width) // 2, (area[1] - fitted.height) // 2))
        fitted = bg_img
    paste_x = x1 + pad + (area[0] - fitted.width) // 2
    paste_y = y1 + pad + (area[1] - fitted.height) // 2
    if base.mode != 'RGBA':
        tmp = base.convert('RGBA')
        tmp.alpha_composite(fitted, (paste_x, paste_y))
        base.paste(tmp.convert('RGB'))
    else:
        base.alpha_composite(fitted, (paste_x, paste_y))


def arrow(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, color: str, width: int = 10, head: int = 20) -> None:
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    left = (x2 - head * math.cos(ang) + head * 0.6 * math.sin(ang), y2 - head * math.sin(ang) - head * 0.6 * math.cos(ang))
    right = (x2 - head * math.cos(ang) - head * 0.6 * math.sin(ang), y2 - head * math.sin(ang) + head * 0.6 * math.cos(ang))
    draw.polygon([(x2, y2), left, right], fill=color)


def bullet_list(draw, x, y, w, items, color=INK, size=22, bullet_fill=ORANGE):
    yy = y
    for item in items:
        draw.ellipse([x, yy + 11, x + 14, yy + 25], fill=bullet_fill)
        bottom = paragraph(draw, x + 28, yy, w - 28, item, size=size, fill=color)
        yy = bottom + 14
    return yy


def slide17(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'OpenRCA Forces Reasoning Over Real Telemetry Scale', ORANGE, 'Capability')
    metric_stack(draw, 88, 216, 360, [('FAILURE CASES', '335'), ('RAW TELEMETRY', '68 GB'), ('SYSTEMS', '3')])
    panel(draw, [486, 214, 1832, 540], fill=LIGHT, outline=LINE)
    band(draw, 486, 214, 1346, 54, 'Realistic RCA begins with telemetry that is too large and too heterogeneous to read directly.', NAVY)
    chip_boxes = [
        (540, 302, 880, 444, paths.media / 'image36.png', 'Different metrics expose onset and magnitude.'),
        (988, 302, 1328, 444, paths.media / 'image37.png', 'Logs carry local semantics and failure clues.'),
        (1436, 302, 1776, 444, paths.media / 'image38.png', 'Traces reveal cross-service hop chains.'),
    ]
    for x1, y1, x2, y2, imgp, label in chip_boxes:
        panel(draw, [x1, y1, x2, y2], fill=WHITE, outline=LINE)
        paste_contain(base, imgp, (x1 + 12, y1 + 12, x2 - 12, y1 + 92))
        paragraph(draw, x1 + 18, y1 + 104, x2 - x1 - 36, label, size=20, fill=INK)
    # reload draw after alpha composite workaround later in render phase
    draw = ImageDraw.Draw(base)
    arrow(draw, 710, 498, 954, 498, ORANGE, width=8)
    arrow(draw, 1158, 498, 1402, 498, ORANGE, width=8)
    panel(draw, [486, 592, 1832, 970], fill=WHITE, outline=LINE)
    band(draw, 486, 592, 1346, 54, 'The reasoning burden is sequential, not parallel.', GREEN)
    steps = [
        ('1', 'Collect', 'Mixed evidence across modalities and time ranges.'),
        ('2', 'Fuse', 'Align temporal context before causal search starts.'),
        ('3', 'Trace', 'Walk multi-hop propagation until a defensible origin emerges.'),
    ]
    sx = 548
    for idx, (num, head, body) in enumerate(steps):
        cx = sx + idx * 420
        draw.ellipse([cx, 684, cx + 62, 746], fill=GREEN if idx == 1 else NAVY)
        centered_text(draw, (cx, 684, cx + 62, 746), num, font(28, True), WHITE)
        draw.text((cx + 90, 686), head, font=font(27, True), fill=INK)
        paragraph(draw, cx + 90, 724, 290, body, size=21, fill=MUTED)
        if idx < 2:
            arrow(draw, cx + 320, 714, cx + 386, 714, ORANGE, width=6, head=14)
    paragraph(draw, 92, 934, 1730, 'Scale is part of the task: realistic RCA forces long-context, heterogeneous, multi-hop reasoning before any diagnosis can be trusted.', size=22, fill=MUTED, bold=True)


def slide18(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'OpenRCA: Dataset Construction Pipeline', ORANGE, 'Realism')
    panel(draw, [88, 208, 1180, 902], fill=LIGHT, outline=LINE)
    band(draw, 88, 208, 1092, 54, 'From raw systems to a benchmark where failures must be operationally visible.', NAVY)
    paste_contain(base, paths.media / 'image39.png', (124, 290, 1144, 862), pad=8)
    panel(draw, [1230, 208, 1832, 902], fill=WHITE, outline=LINE)
    band(draw, 1230, 208, 602, 54, 'Why this pipeline feels designed, not assembled', GREEN)
    items = [
        ('System selection', 'Choose systems with real cross-service structure rather than toy graphs.'),
        ('Fault injection', 'Enumerate diverse interventions instead of one dominant failure pattern.'),
        ('Telemetry collection', 'Keep metrics, logs, and traces together so propagation stays observable.'),
        ('Impact validation', 'Retain only cases that create user-facing degradation, not silent injections.'),
    ]
    yy = 294
    colors = [NAVY, ORANGE, BLUE, GREEN]
    for i, (head, body) in enumerate(items):
        draw.rectangle([1260, yy, 1290, yy + 120], fill=colors[i])
        panel(draw, [1290, yy, 1798, yy + 120], fill=CREAM if i % 2 == 0 else WHITE, outline='#e5e8ea')
        draw.text((1320, yy + 18), head, font=font(23, True), fill=INK)
        paragraph(draw, 1320, yy + 50, 452, body, size=19, fill=MUTED)
        yy += 144
    paragraph(draw, 88, 946, 1740, 'The redesign is not just bigger data. It couples intervention diversity with propagation-aware filtering, so benchmark difficulty comes from operational realism.', size=22, fill=MUTED, bold=True)


def slide21(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'OpenRCA: Goal-Driven Task Formulation', ORANGE, 'Task Contract')
    paragraph(draw, 92, 176, 1100, 'A root cause is defined by three elements. Query difficulty comes from asking the agent to recover 1, 2, or all 3 of them.', size=24, fill=MUTED)
    cards = [
        (92, 278, 566, 476, BLUE, 'Originating\nComponent', 'e.g., frontend-2'),
        (724, 278, 1198, 476, GREEN, 'Start\nTime', 'e.g., 2022-03-20 09:05:00'),
        (1356, 278, 1830, 476, ORANGE, 'Failure\nReason', 'e.g., container I/O load'),
    ]
    for x1, y1, x2, y2, color, title, ex in cards:
        strong_box(draw, [x1, y1, x2, y2], accent=color, fill=WHITE)
        draw.rectangle([x1, y1, x2, y1 + 42], fill=color)
        centered_text(draw, (x1 + 20, y1 + 68, x2 - 20, y1 + 170), title, font(28, True), INK)
        centered_text(draw, (x1 + 20, y1 + 176, x2 - 20, y2 - 24), ex, font(22), MUTED)
    panel(draw, [92, 564, 890, 940], fill=LIGHT, outline=LINE)
    band(draw, 92, 564, 798, 52, '7 query types = C(3,1) + C(3,2) + C(3,3)', NAVY)
    bullet_list(draw, 124, 652, 720, [
        'Easy: identify only one element (component, time, or reason).',
        'Medium: recover any two elements jointly, where partial answers no longer suffice.',
        'Hard: recover all three elements as one coherent root-cause tuple.',
    ], size=22, bullet_fill=ORANGE)
    strong_box(draw, [948, 564, 1830, 940], accent=GREEN, fill=CREAM)
    draw.text((984, 628), 'Difficulty ladder', font=font(28, True), fill=INK)
    rows = [
        ('Easy', '3 tasks', 'one element only', PALE_BLUE),
        ('Medium', '3 tasks', 'any two elements', PALE_GREEN),
        ('Hard', '1 task', 'full tuple, no shortcuts', PALE_ORANGE),
    ]
    yy = 690
    for label, count, desc, fill in rows:
        panel(draw, [984, yy, 1794, yy + 78], fill=fill, outline=WHITE)
        draw.text((1014, yy + 20), label, font=font(22, True), fill=INK)
        draw.text((1190, yy + 20), count, font=font(20, True), fill=MUTED)
        draw.text((1390, yy + 20), desc, font=font(20), fill=INK)
        yy += 96


def slide22(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'RCA-agent: Agentic Baseline for OpenRCA', ORANGE, 'Agent Baseline')
    strong_box(draw, [92, 176, 1830, 260], accent=ORANGE, fill=CREAM)
    centered_text(draw, (120, 192, 1802, 244), 'Key insight: telemetry is too large for direct prompting, so the agent writes and runs analysis code instead.', font(27, True), INK)
    panel(draw, [92, 302, 1830, 672], fill=LIGHT, outline=LINE)
    paste_contain(base, paths.media / 'image48.png', (156, 344, 1766, 632), pad=4)
    card_y = 732
    strong_box(draw, [92, card_y, 908, 980], accent=ORANGE, fill=WHITE)
    band(draw, 92, card_y, 816, 46, 'Controller', ORANGE)
    bullet_list(draw, 128, 804, 728, [
        'Orchestrates the analysis sequence: anomaly detection -> fault identification -> root cause localization.',
        'Keeps the investigation goal-driven instead of sampling telemetry blindly.',
    ], size=22, bullet_fill=ORANGE)
    strong_box(draw, [1014, card_y, 1830, 980], accent=GREEN, fill=WHITE)
    band(draw, 1014, card_y, 816, 46, 'Executor', GREEN)
    bullet_list(draw, 1050, 804, 728, [
        'Generates Python code to inspect data programmatically, beyond LLM context limits.',
        'Caches variables inside a stateful kernel so multi-step RCA can build on previous work.',
    ], size=22, bullet_fill=GREEN)


def slide23(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'OpenRCA: Main Evaluation Results', ORANGE, 'Capability Gap')
    panel(draw, [92, 196, 1164, 944], fill=LIGHT, outline=LINE)
    band(draw, 92, 196, 1072, 52, 'The main result is not who wins. It is how far current systems remain from robust RCA.', NAVY)
    paste_contain(base, paths.media / 'image49.png', (122, 286, 1134, 914), pad=6)
    strong_box(draw, [1214, 196, 1830, 944], accent=ORANGE, fill=CREAM)
    draw.text((1250, 266), 'What the table says', font=font(30, True), fill=INK)
    bullet_list(draw, 1252, 340, 520, [
        'Even the strongest agents remain far from saturation on realistic RCA.',
        'Performance collapses fastest on the 2-element and 3-element queries, where reasoning must stay coherent across multiple targets.',
        'The benchmark separates model families by diagnostic competence instead of surface fluency.',
    ], size=22, bullet_fill=ORANGE)
    band(draw, 1242, 748, 558, 44, 'Reading rule', GREEN)
    paragraph(draw, 1252, 812, 522, 'Treat every reported score as evidence about reasoning limits under telemetry pressure, not as a near-solved leaderboard race.', size=23, fill=INK, bold=True)


def slide25(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'Accuracy Across Systems and Difficulty Levels', ORANGE, 'Difficulty')
    panel(draw, [92, 196, 1106, 944], fill=LIGHT, outline=LINE)
    band(draw, 92, 196, 1014, 52, 'Accuracy falls sharply once the query asks for coupled reasoning.', NAVY)
    paste_contain(base, paths.media / 'image50.png', (118, 278, 1080, 652), pad=4)
    panel(draw, [118, 690, 1080, 920], fill=WHITE, outline=LINE)
    paste_contain(base, paths.media / 'image51.png', (138, 710, 1060, 900), pad=4)
    strong_box(draw, [1160, 196, 1830, 944], accent=GREEN, fill=CREAM)
    draw.text((1196, 254), 'Key findings', font=font(30, True), fill=INK)
    bullet_list(draw, 1200, 336, 576, [
        'Performance degrades monotonically from Easy to Hard across systems and model families.',
        'No model solves the Hard 3-element query consistently; full root-cause tuples remain out of reach.',
        'Failure reason is the hardest field because it demands mechanism-level understanding, not just localization.',
    ], size=22, bullet_fill=GREEN)
    band(draw, 1192, 764, 584, 44, ORANGE_DARK, WHITE)
    paragraph(draw, 1200, 824, 576, 'The challenge is compositional: identifying one element is feasible, but keeping multiple elements jointly correct is where agents break.', size=22, fill=INK, bold=True)


def slide27(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'Analysis: Reasoning Length and Error Tolerance', ORANGE, 'Failure Modes')
    panel(draw, [92, 196, 1028, 690], fill=LIGHT, outline=LINE)
    paste_contain(base, paths.media / 'image52.png', (116, 228, 1004, 662), pad=6)
    strong_box(draw, [1080, 196, 1830, 690], accent=ORANGE, fill=CREAM)
    draw.text((1120, 250), 'Interpretation', font=font(30, True), fill=INK)
    bullet_list(draw, 1124, 332, 644, [
        'Short reasoning is common, but longer multi-step analysis tends to correlate with better RCA accuracy.',
        'Execution reliability matters: agents that recover from code errors preserve much more performance.',
    ], size=23, bullet_fill=ORANGE)
    strong_box(draw, [92, 748, 854, 970], accent=GREEN, fill=WHITE)
    band(draw, 92, 748, 762, 44, 'Finding 1', GREEN)
    paragraph(draw, 124, 816, 700, 'OpenRCA rewards persistence. The agent often needs more than 10 steps because each telemetry slice only reveals part of the causal path.', size=22, fill=INK)
    strong_box(draw, [910, 748, 1830, 970], accent=NAVY, fill=WHITE)
    band(draw, 910, 748, 920, 44, 'Finding 2', NAVY)
    paragraph(draw, 942, 816, 850, 'Error tolerance becomes a model capability in its own right: the better agents recover from code mistakes instead of abandoning the investigation.', size=22, fill=INK)


def slide29(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'OpenRCA: Impact and Community Adoption', ORANGE, 'Impact')
    strong_box(draw, [92, 190, 900, 520], accent=ORANGE, fill=CREAM)
    draw.text((132, 238), 'Adopted by Anthropic', font=font(31, True), fill=INK)
    paragraph(draw, 132, 306, 708, 'OpenRCA was used by Anthropic to evaluate Claude Opus 4.6 as a benchmark for real software-diagnosis reasoning.', size=24, fill=INK)
    paste_contain(base, paths.media / 'image53.png', (636, 334, 856, 474), pad=4)
    strong_box(draw, [1020, 190, 1830, 520], accent=GREEN, fill=WHITE)
    draw.text((1060, 238), 'Community contribution', font=font(31, True), fill=INK)
    paragraph(draw, 1060, 306, 708, 'The benchmark is open, published, and already visible as a Microsoft open-source project with strong community pickup.', size=24, fill=INK)
    paste_contain(base, paths.media / 'image54.png', (1540, 336, 1760, 476), pad=4)
    panel(draw, [92, 584, 1830, 960], fill=LIGHT, outline=LINE)
    band(draw, 92, 584, 1738, 48, 'But adoption is not the same as easiness: impact matters because the benchmark stresses realistic propagation patterns.', NAVY)
    bullet_list(draw, 130, 682, 1570, [
        'Earlier AIOps datasets often contain limited propagation diversity, which makes RCA look artificially simple.',
        'OpenRCA matters because it couples visibility, realism, and difficulty in one benchmark rather than optimizing for one axis at a time.',
        'That is why community uptake strengthens the benchmark story instead of replacing the technical argument.',
    ], size=23, bullet_fill=ORANGE)


def slide30(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'Correct Labels Can Still Hide Wrong Reasoning', ORANGE, 'Trust')
    pill(draw, 92, 176, 640, 44, 'Question C: can we verify the causal reasoning path?', NAVY, WHITE, 22)
    # left
    strong_box(draw, [92, 252, 852, 934], accent=RED, fill=PALE_RED)
    band(draw, 92, 252, 760, 48, 'CORRECT LABEL ONLY', RED)
    paragraph(draw, 124, 322, 700, 'The answer lands on the right component by following the loudest symptom, not the real propagation path.', size=23, fill=INK)
    nodes = [('S', 'Symptom', 'largest anomaly'), ('R', 'Rank', 'most visible service'), ('L', 'Label', 'root guessed right')]
    x = 152
    for i, (code, head, sub) in enumerate(nodes):
        cx = x + i * 220
        draw.ellipse([cx, 504, cx + 88, 592], fill=RED if i == 0 else '#d88b8a')
        centered_text(draw, (cx, 504, cx + 88, 592), code, font(34, True), WHITE)
        centered_text(draw, (cx - 28, 628, cx + 116, 708), head, font(23, True), INK)
        centered_text(draw, (cx - 40, 694, cx + 128, 776), sub, font(19), MUTED)
        if i < 2:
            arrow(draw, cx + 100, 548, cx + 186, 548, RED, width=8)
    band(draw, 124, 838, 696, 38, 'Looks correct but earns unearned outcome credit.', '#9f4e4e', size=17)
    # right
    strong_box(draw, [1000, 252, 1830, 934], accent=GREEN, fill=PALE_GREEN)
    band(draw, 1000, 252, 830, 48, 'FAITHFUL REASONING', GREEN)
    paragraph(draw, 1032, 322, 760, 'A trustworthy diagnosis reconstructs the causal chain and checks each hop against real telemetry.', size=23, fill=INK)
    nodes = [('E', 'Evidence', 'metrics / logs / traces'), ('H1', 'Hop 1', 'candidate path'), ('H2', 'Hop 2', 'verified dependency'), ('RC', 'Root', 'causal origin')]
    x = 1044
    xs = [1044, 1260, 1476, 1692]
    for i, (code, head, sub) in enumerate(nodes):
        cx = xs[i]
        draw.ellipse([cx, 504, cx + 88, 592], fill=GREEN if i > 0 else NAVY)
        centered_text(draw, (cx, 504, cx + 88, 592), code, font(24, True), WHITE)
        centered_text(draw, (cx - 28, 628, cx + 116, 708), head, font(22, True), INK)
        centered_text(draw, (cx - 50, 694, cx + 138, 776), sub, font(18), MUTED)
        if i < 3:
            arrow(draw, cx + 100, 548, xs[i + 1] - 16, 548, GREEN, width=8)
    band(draw, 1032, 838, 766, 38, 'Correct label plus auditable path makes the answer usable.', '#386d59', size=17)


def slide31(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'Outcome-Only Evaluation Misses Process Failures', ORANGE, 'Audit Depth')
    steps = [
        ('OUTCOME-ONLY', 'Checks whether the final answer matches the label. Fast and useful, but shallow.', ORANGE, 120, 110),
        ('PROCESS-AWARE', 'Checks whether the explanation survives path-level scrutiny, intervention logic, and causal consistency.', GREEN, 220, 460),
        ('TRUST VERDICT', 'Only the deeper audit layer can distinguish lucky labels from faithful reasoning.', NAVY, 320, 810),
    ]
    for label, body, color, yy, xx in steps:
        strong_box(draw, [xx, yy, 1770, yy + 146], accent=color, fill=CREAM if color != NAVY else LIGHT)
        band(draw, xx, yy, 1770 - xx, 48, label, color)
        paragraph(draw, xx + 34, yy + 74, 1770 - xx - 68, body, size=24, fill=INK)
    strong_box(draw, [92, 818, 1830, 968], accent=RED, fill=PALE_RED)
    draw.text((132, 858), 'Blind spot', font=font(27, True), fill=INK)
    paragraph(draw, 304, 854, 1460, 'If RCA is delegated to an agent, outcome-only evaluation hides the exact process failures that matter most in operations.', size=24, fill=INK, bold=True)


def slide32(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'FORGE Uses Forward Verification from Known Interventions', ORANGE, 'FORGE')
    centers = [(240, 340), (670, 340), (1100, 340), (1530, 340)]
    labels = [('I', 'Intervene', 'known fault', ORANGE), ('F', 'Forward', 'cause -> effect', BLUE), ('V', 'Verify', 'match observed path', GREEN), ('J', 'Judge', 'faithful or not', RED)]
    for i, (cx, cy) in enumerate(centers):
        code, head, sub, color = labels[i]
        draw.ellipse([cx - 74, cy - 74, cx + 74, cy + 74], fill=color)
        centered_text(draw, (cx - 74, cy - 74, cx + 74, cy + 74), code, font(44, True), WHITE)
        centered_text(draw, (cx - 120, cy + 100, cx + 120, cy + 154), head, font(27, True), INK)
        centered_text(draw, (cx - 140, cy + 150, cx + 140, cy + 210), sub, font(21), MUTED)
        if i < 3:
            arrow(draw, cx + 92, cy, centers[i + 1][0] - 92, cy, ORANGE, width=10, head=20)
    strong_box(draw, [146, 560, 1774, 700], accent=GREEN, fill=CREAM)
    centered_text(draw, (188, 590, 1732, 670), 'Intervention turns trust evaluation into an operational test: verify the predicted forward path before granting credit.', font(27, True), INK)
    panel(draw, [92, 756, 1830, 970], fill=LIGHT, outline=LINE)
    band(draw, 92, 756, 1738, 44, 'Why FORGE matters', NAVY)
    bullet_list(draw, 128, 834, 1600, [
        'Backward inference alone is underdetermined in rich telemetry.',
        'Known interventions make verification tractable: start from a cause and test whether the observed downstream effects line up.',
    ], size=24, bullet_fill=ORANGE)


def slide33(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'FORGE Audits Answers; OpenRCA 2.0 Audits Reasoning Steps', ORANGE, 'OpenRCA 2.0')
    strong_box(draw, [92, 194, 680, 482], accent=ORANGE, fill=PALE_ORANGE)
    band(draw, 92, 194, 588, 42, 'ANSWER-ONLY SUPERVISION', ORANGE)
    paragraph(draw, 122, 264, 528, 'A correct root cause can still hide guessed, skipped, or weakly supported intermediate hops.', size=23, fill=INK)
    strong_box(draw, [744, 194, 1830, 482], accent=GREEN, fill=PALE_GREEN)
    band(draw, 744, 194, 1086, 42, 'HOP-LEVEL SUPERVISION', GREEN)
    centers = [832, 1030, 1228, 1426, 1624]
    desc = [('S', 'Symptom'), ('1', 'Hop 1'), ('2', 'Hop 2'), ('3', 'Hop 3'), ('RC', 'Root')]
    for i, (code, label) in enumerate(desc):
        draw.ellipse([centers[i] - 42, 320, centers[i] + 42, 404], fill=NAVY if i == 0 else GREEN)
        centered_text(draw, (centers[i] - 42, 320, centers[i] + 42, 404), code, font(26, True), WHITE)
        centered_text(draw, (centers[i] - 72, 422, centers[i] + 72, 460), label, font(18, True), INK)
        if i < len(desc) - 1:
            arrow(draw, centers[i] + 54, 362, centers[i + 1] - 54, 362, GREEN, width=7, head=15)
    panel(draw, [92, 548, 1830, 954], fill=LIGHT, outline=LINE)
    paste_contain(base, paths.media / 'image55.png', (124, 586, 1798, 916), pad=6)
    paragraph(draw, 92, 974, 1740, 'OpenRCA 2.0 upgrades trust evaluation from answer-level labels to hop-level causal supervision.', size=23, fill=MUTED, bold=True)


def slide34(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'OpenRCA 2.0 Uses a Finite, Fault-Type-Specific Rule Set', ORANGE, 'Rule Set')
    strong_box(draw, [92, 194, 760, 484], accent=ORANGE, fill=PALE_ORANGE)
    draw.text((126, 238), 'Rule design', font=font(30, True), fill=INK)
    bullet_list(draw, 128, 306, 580, [
        'A rule is a finite state transition: (entity type : state) -> (entity type : state).',
        'Rules are fault-type-specific, not system-specific.',
        'Unsupported rules were removed after intervention evidence failed to support them.',
    ], size=22, bullet_fill=ORANGE)
    panel(draw, [92, 536, 760, 950], fill=WHITE, outline=LINE)
    paste_contain(base, paths.media / 'image56.png', (116, 560, 736, 926), pad=4)
    panel(draw, [820, 194, 1830, 950], fill=LIGHT, outline=LINE)
    band(draw, 820, 194, 1010, 46, 'Evidence view', GREEN)
    paste_contain(base, paths.media / 'image57.png', (852, 262, 1798, 640), pad=4)
    strong_box(draw, [852, 684, 1798, 930], accent=GREEN, fill=CREAM)
    paragraph(draw, 888, 734, 874, 'Compact rule sets over discrete states are defensible because they are grounded in known failure mechanisms, then filtered by telemetry evidence.', size=24, fill=INK, bold=True)


def slide35(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'Why Coincidental Chains Rarely Survive Verification', ORANGE, 'Filter Logic')
    draw.ellipse([92, 196, 176, 280], fill=GREEN)
    centered_text(draw, (92, 196, 176, 280), 'C', font(34, True), WHITE)
    paragraph(draw, 212, 196, 720, 'Three filters must agree at every hop', size=31, fill=INK, bold=True)
    paragraph(draw, 212, 246, 980, 'statistical anomaly + rule conformance + temporal order', size=23, fill=MUTED)
    strong_box(draw, [92, 332, 1830, 472], accent=NAVY, fill=LIGHT)
    centered_text(draw, (124, 360, 1798, 440), 'A path survives only when all three checks hold at every hop.  P(false path accepted) <= (p_s * p_r * p_t)^k', font(27, True), INK)
    cards = [
        (92, 556, 612, 836, ORANGE, 'Statistical anomaly', 'Baseline-relative node screen'),
        (700, 556, 1220, 836, GREEN, 'Rule conformance', 'Finite fault-type-specific transition set'),
        (1308, 556, 1828, 836, NAVY, 'Temporal order', 'Downstream effects cannot precede the cause'),
    ]
    for x1, y1, x2, y2, color, head, body in cards:
        strong_box(draw, [x1, y1, x2, y2], accent=color, fill=WHITE)
        band(draw, x1, y1, x2 - x1, 44, head.upper(), color, WHITE, 21)
        centered_text(draw, (x1 + 30, y1 + 120, x2 - 30, y2 - 24), body, font(24, True), INK)
    paragraph(draw, 92, 922, 1740, 'Even permissive per-hop thresholds collapse quickly. Process-aware verification makes lucky coincidence exponentially harder to accept.', size=23, fill=MUTED, bold=True)


def slide36(base: Image.Image, paths: DeckPaths):
    draw = ImageDraw.Draw(base)
    top_frame(draw, 'Takeaway C: OpenRCA 2.0 Makes the Trust Gap Visible', ORANGE, 'Takeaway')
    draw.ellipse([92, 182, 176, 266], fill=GREEN)
    centered_text(draw, (92, 182, 176, 266), 'C', font(34, True), WHITE)
    paragraph(draw, 216, 178, 940, 'Trust now has a measurable criterion', size=34, fill=INK, bold=True)
    paragraph(draw, 216, 234, 1020, 'Reliable RCA needs causal-path faithfulness, not only correct final labels.', size=26, fill=MUTED)
    strong_box(draw, [92, 338, 1830, 480], accent=ORANGE, fill=CREAM)
    centered_text(draw, (124, 366, 1798, 452), 'Outcome accuracy overstates trust: once the path must be verified, many apparently correct diagnoses lose credit.', font(28, True), INK)
    band(draw, 92, 548, 460, 44, 'QUESTION C ANSWERED', NAVY)
    paragraph(draw, 92, 614, 1000, 'Pass@1 and PR expose the process failures hidden behind good labels.', size=26, fill=INK, bold=True)
    cards = [
        (92, 744, 572, 952, ORANGE, 'BEST MODEL', 'Pass@1 0.76 -> PR 0.63'),
        (720, 744, 1200, 952, GREEN, 'AVERAGE OF 7 LLMs', 'Pass@1 0.52 -> PR 0.43'),
        (1348, 744, 1828, 952, NAVY, 'HALLUCINATION SIGNAL', '2.1 fabricated edges / diagnosis'),
    ]
    for x1, y1, x2, y2, color, head, body in cards:
        strong_box(draw, [x1, y1, x2, y2], accent=color, fill=WHITE)
        band(draw, x1, y1, x2 - x1, 42, head, color, WHITE, 20)
        centered_text(draw, (x1 + 20, y1 + 86, x2 - 20, y2 - 24), body, font(28, True), INK)
    paragraph(draw, 92, 990, 1740, 'Next: synthesize realism, capability, and trust as one evaluation story.', size=22, fill=MUTED, bold=True)


SLIDE_BUILDERS: dict[int, Callable[[Image.Image, DeckPaths], None]] = {
    17: slide17,
    18: slide18,
    21: slide21,
    22: slide22,
    23: slide23,
    25: slide25,
    27: slide27,
    29: slide29,
    30: slide30,
    31: slide31,
    32: slide32,
    33: slide33,
    34: slide34,
    35: slide35,
    36: slide36,
}


def build_image(slide_no: int, paths: DeckPaths) -> Path:
    base, _ = new_slide()
    SLIDE_BUILDERS[slide_no](base, paths)
    out = paths.media / f'reworked_slide{slide_no}.png'
    base.save(out)
    return out


def rewrite_slide_xml(slide_xml: Path, slide_rels: Path, img_target: str) -> None:
    rel_root = etree.parse(str(slide_rels)).getroot()
    layout_rel = None
    notes_rel = None
    for rel in rel_root:
        rel_type = rel.get('Type')
        if rel_type.endswith('/slideLayout'):
            layout_rel = rel
        elif rel_type.endswith('/notesSlide'):
            notes_rel = rel
    for child in list(rel_root):
        rel_root.remove(child)
    rel_root.append(etree.Element(f'{{{PKG_REL_NS}}}Relationship', Id='rId1', Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout', Target=layout_rel.get('Target')))
    if notes_rel is not None:
        rel_root.append(etree.Element(f'{{{PKG_REL_NS}}}Relationship', Id='rId2', Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide', Target=notes_rel.get('Target')))
        img_id = 'rId3'
    else:
        img_id = 'rId2'
    rel_root.append(etree.Element(f'{{{PKG_REL_NS}}}Relationship', Id=img_id, Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/image', Target=img_target))
    slide_root = etree.Element(f'{{{P_NS}}}sld', nsmap={'a': A_NS, 'p': P_NS, 'r': R_NS})
    cSld = etree.SubElement(slide_root, f'{{{P_NS}}}cSld')
    spTree = etree.SubElement(cSld, f'{{{P_NS}}}spTree')
    nvGrpSpPr = etree.SubElement(spTree, f'{{{P_NS}}}nvGrpSpPr')
    etree.SubElement(nvGrpSpPr, f'{{{P_NS}}}cNvPr', id='1', name='')
    etree.SubElement(nvGrpSpPr, f'{{{P_NS}}}cNvGrpSpPr')
    etree.SubElement(nvGrpSpPr, f'{{{P_NS}}}nvPr')
    etree.SubElement(spTree, f'{{{P_NS}}}grpSpPr')
    pic = etree.SubElement(spTree, f'{{{P_NS}}}pic')
    nvPicPr = etree.SubElement(pic, f'{{{P_NS}}}nvPicPr')
    etree.SubElement(nvPicPr, f'{{{P_NS}}}cNvPr', id='2', name='Picture 1', descr=Path(img_target).name)
    cNvPicPr = etree.SubElement(nvPicPr, f'{{{P_NS}}}cNvPicPr')
    etree.SubElement(cNvPicPr, f'{{{A_NS}}}picLocks', noChangeAspect='1')
    etree.SubElement(nvPicPr, f'{{{P_NS}}}nvPr')
    blipFill = etree.SubElement(pic, f'{{{P_NS}}}blipFill')
    blip = etree.SubElement(blipFill, f'{{{A_NS}}}blip')
    blip.set(f'{{{R_NS}}}embed', img_id)
    stretch = etree.SubElement(blipFill, f'{{{A_NS}}}stretch')
    etree.SubElement(stretch, f'{{{A_NS}}}fillRect')
    spPr = etree.SubElement(pic, f'{{{P_NS}}}spPr')
    xfrm = etree.SubElement(spPr, f'{{{A_NS}}}xfrm')
    etree.SubElement(xfrm, f'{{{A_NS}}}off', x='0', y='0')
    etree.SubElement(xfrm, f'{{{A_NS}}}ext', cx='12192000', cy='6858000')
    prstGeom = etree.SubElement(spPr, f'{{{A_NS}}}prstGeom', prst='rect')
    etree.SubElement(prstGeom, f'{{{A_NS}}}avLst')
    clr = etree.SubElement(slide_root, f'{{{P_NS}}}clrMapOvr')
    etree.SubElement(clr, f'{{{A_NS}}}masterClrMapping')
    slide_xml.write_bytes(etree.tostring(slide_root, xml_declaration=True, encoding='utf-8', pretty_print=True))
    slide_rels.write_bytes(etree.tostring(rel_root, xml_declaration=True, encoding='utf-8', pretty_print=True))


def main() -> None:
    unpacked = Path('/tmp/reviesed_work')
    paths = DeckPaths(unpacked=unpacked, media=unpacked / 'ppt' / 'media', slides=unpacked / 'ppt' / 'slides', rels=unpacked / 'ppt' / 'slides' / '_rels')
    for slide_no in SLIDE_BUILDERS:
        img = build_image(slide_no, paths)
        rewrite_slide_xml(paths.slides / f'slide{slide_no}.xml', paths.rels / f'slide{slide_no}.xml.rels', f'../media/{img.name}')
    print('Generated and rewired', len(SLIDE_BUILDERS), 'slides')


if __name__ == '__main__':
    main()
