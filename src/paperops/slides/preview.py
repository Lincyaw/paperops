"""Slide preview rendering and presentation checking."""

from __future__ import annotations

import os

from pptx import Presentation as PptxPresentation
from pptx.util import Inches, Pt, Emu

from paperops.slides.core.constants import SLIDE_WIDTH, SLIDE_HEIGHT
from paperops.slides.layout.auto_size import _load_pil_font, measure_text


def _wrap_preview_text(draw, text: str, font, max_width_px: int) -> list[str]:
    """Wrap preview text to fit the available width."""
    if not text:
        return [""]

    lines: list[str] = []
    for raw_line in text.splitlines() or [""]:
        words = raw_line.split()
        if not words:
            lines.append("")
            continue

        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width_px:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)

    return lines


# ──────────────────────────────────────────────────────────────────────
# Preview rendering (PIL-based soft render)
# ──────────────────────────────────────────────────────────────────────

def render_slide_preview(slide_builder, output_path: str, width_px: int = 1920):
    """Render a slide to PNG using PIL.

    Draws rectangles, text, and colors based on the pptx slide's shapes.
    Sufficient for AI to verify layout and proportions, not pixel-perfect.
    """
    from PIL import Image, ImageDraw, ImageFont

    height_px = int(width_px * SLIDE_HEIGHT / SLIDE_WIDTH)
    scale_x = width_px / (SLIDE_WIDTH * 914400)  # EMU to px
    scale_y = height_px / (SLIDE_HEIGHT * 914400)

    img = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(img)

    slide = slide_builder._slide

    # Load font via shared utility (handles caching + fallback)
    font = _load_pil_font("DejaVu Sans", 16) or ImageFont.load_default()

    def _emu_to_px(left, top, width, height):
        x = int((left or 0) * scale_x)
        y = int((top or 0) * scale_y)
        w = int((width or 0) * scale_x)
        h = int((height or 0) * scale_y)
        return x, y, w, h

    def _rgb_from_pptx(color_obj):
        """Try to extract (r, g, b) from a pptx color."""
        try:
            rgb = color_obj.rgb
            return (rgb[0], rgb[1], rgb[2])
        except Exception:
            return (200, 200, 200)

    def _get_text(shape):
        try:
            if shape.has_text_frame:
                return shape.text_frame.text
        except Exception:
            pass
        return ""

    for shape in slide.shapes:
        left = shape.left or 0
        top = shape.top or 0
        w = shape.width or 0
        h = shape.height or 0
        x, y, pw, ph = _emu_to_px(left, top, w, h)

        if pw <= 0 or ph <= 0:
            continue

        # Determine fill color
        fill_color = (245, 245, 245)
        try:
            if shape.fill and shape.fill.type is not None:
                fill_color = _rgb_from_pptx(shape.fill.fore_color)
        except Exception:
            pass

        # Draw rectangle
        draw.rectangle([x, y, x + pw, y + ph], fill=fill_color, outline=(180, 180, 180))

        # Draw text
        text = _get_text(shape)
        if text:
            font_size = max(10, min(ph // 3, 32))
            text_font = _load_pil_font("DejaVu Sans", font_size) or font
            max_text_width = max(pw - 16, 20)
            max_text_height = max(ph - 12, 12)

            lines = _wrap_preview_text(draw, text, text_font, max_text_width)
            bbox = draw.textbbox((0, 0), "Ag", font=text_font)
            line_height = bbox[3] - bbox[1]
            total_height = max(line_height * len(lines), line_height)

            while total_height > max_text_height and font_size > 8:
                font_size -= 1
                text_font = _load_pil_font("DejaVu Sans", font_size) or font
                lines = _wrap_preview_text(draw, text, text_font, max_text_width)
                bbox = draw.textbbox((0, 0), "Ag", font=text_font)
                line_height = bbox[3] - bbox[1]
                total_height = max(line_height * len(lines), line_height)

            widths = []
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=text_font)
                widths.append(bbox[2] - bbox[0])

            ty = y + max((ph - total_height) // 2, 2)
            for line, line_width in zip(lines, widths):
                tx = x + (pw - line_width) // 2
                draw.text((tx, ty), line, fill=(30, 30, 30), font=text_font)
                ty += line_height

    img.save(output_path)


# ──────────────────────────────────────────────────────────────────────
# Presentation checker
# ──────────────────────────────────────────────────────────────────────

EMU_PER_INCH = 914400
EMU_PER_PT = 12700


def _estimate_text_height(text: str, font_size_emu: int, box_width_emu: int,
                          line_spacing: float = 1.2) -> int:
    """Estimate text height in EMU, delegating to the unified measure_text."""
    font_size_pt = font_size_emu / EMU_PER_PT
    box_width_inches = box_width_emu / EMU_PER_INCH if box_width_emu > 0 else None
    _w, h_inches = measure_text(
        text, "Calibri", font_size_pt, max_width_inches=box_width_inches,
    )
    return int(h_inches * EMU_PER_INCH)


def _get_all_text(shape) -> str:
    parts = []
    if shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            for run in para.runs:
                parts.append(run.text)
            if para.runs:
                parts.append('\n')
    return ''.join(parts).strip()


def _get_font_size(shape) -> int:
    if shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            if para.font.size:
                return para.font.size
            for run in para.runs:
                if run.font.size:
                    return run.font.size
    return Pt(16)


def _get_shape_bounds(shape):
    left = shape.left or 0
    top = shape.top or 0
    width = shape.width or 0
    height = shape.height or 0
    return (left, top, left + width, top + height)


def _shapes_overlap(b1, b2, tolerance_emu=0):
    l1, t1, r1, bot1 = b1
    l2, t2, r2, bot2 = b2
    l1 += tolerance_emu
    t1 += tolerance_emu
    r1 -= tolerance_emu
    bot1 -= tolerance_emu
    overlap_left = max(l1, l2)
    overlap_top = max(t1, t2)
    overlap_right = min(r1, r2)
    overlap_bottom = min(bot1, bot2)
    if overlap_left < overlap_right and overlap_top < overlap_bottom:
        return (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
    return 0


def _emu_to_inches(emu):
    return emu / EMU_PER_INCH


def _bounds_to_inches(bounds):
    return tuple(_emu_to_inches(v) for v in bounds)


def _shape_info(shape) -> str:
    text = _get_all_text(shape)
    if text:
        preview = text[:50].replace('\n', ' ')
        if len(text) > 50:
            preview += '...'
        return f'"{preview}"'
    return f'<shape>'


def check_presentation(pptx_path: str) -> list[dict]:
    """Open a .pptx file and check for layout issues.

    Returns list of issue dicts with keys: slide, type, detail, etc.
    """
    prs = PptxPresentation(pptx_path)
    slide_width = prs.slide_width
    slide_height = prs.slide_height

    all_issues = []

    for slide_num, slide in enumerate(prs.slides, 1):
        shapes_with_text = []

        for shape in slide.shapes:
            bounds = _get_shape_bounds(shape)
            info = _shape_info(shape)

            # Check off-slide
            l, t, r, b = bounds
            if (l < -Inches(0.1) or t < -Inches(0.1) or
                    r > slide_width + Inches(0.1) or b > slide_height + Inches(0.1)):
                all_issues.append({
                    'slide': slide_num,
                    'type': 'off_slide',
                    'detail': f'Shape {info} extends beyond slide boundaries',
                    'position': _bounds_to_inches(bounds),
                })

            # Check text overflow
            if shape.has_text_frame:
                text = _get_all_text(shape)
                if text:
                    font_size = _get_font_size(shape)
                    box_w = shape.width or 0
                    box_h = shape.height or 0

                    tf = shape.text_frame
                    margin_l = tf.margin_left or Inches(0.05)
                    margin_r = tf.margin_right or Inches(0.05)
                    margin_t = tf.margin_top or Inches(0.05)
                    margin_b = tf.margin_bottom or Inches(0.05)

                    usable_w = box_w - margin_l - margin_r
                    usable_h = box_h - margin_t - margin_b

                    if usable_w > 0 and usable_h > 0:
                        needed_h = _estimate_text_height(text, font_size, usable_w)
                        if needed_h > usable_h * 1.1:
                            overflow_pct = (needed_h / usable_h - 1) * 100
                            all_issues.append({
                                'slide': slide_num,
                                'type': 'text_overflow',
                                'detail': (f'Text {info} overflows by '
                                           f'{overflow_pct:.0f}%'),
                                'position': _bounds_to_inches(bounds),
                            })

                    shapes_with_text.append((shape, bounds, info))

        # Check overlaps between text shapes
        for i in range(len(shapes_with_text)):
            for j in range(i + 1, len(shapes_with_text)):
                s1, b1, info1 = shapes_with_text[i]
                s2, b2, info2 = shapes_with_text[j]
                overlap = _shapes_overlap(b1, b2, tolerance_emu=Inches(0.05))
                if overlap > 0:
                    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    smaller = min(area1, area2)
                    if smaller > 0:
                        pct = overlap / smaller * 100
                        if pct > 10:
                            all_issues.append({
                                'slide': slide_num,
                                'type': 'overlap',
                                'detail': (f'{info1} and {info2} overlap by '
                                           f'{pct:.0f}%'),
                            })

    return all_issues
