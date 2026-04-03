"""Slide preview rendering, checking, and integrated deck review."""

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


def _estimate_text_height(
    text: str,
    font_size_emu: int,
    box_width_emu: int,
    font_family: str = "Calibri",
    line_spacing: float = 1.2,
) -> int:
    """Estimate text height in EMU, delegating to the unified measure_text."""
    font_size_pt = font_size_emu / EMU_PER_PT
    box_width_inches = box_width_emu / EMU_PER_INCH if box_width_emu > 0 else None
    _w, h_inches = measure_text(
        text, font_family, font_size_pt, max_width_inches=box_width_inches,
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


def _get_font_family(shape) -> str:
    if shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            if para.font.name:
                return para.font.name
            for run in para.runs:
                if run.font.name:
                    return run.font.name
    return "Calibri"


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


def _longest_token_len(text: str) -> int:
    tokens = [token for token in text.replace("\n", " ").split(" ") if token]
    if not tokens:
        return len(text.strip())
    return max(len(token) for token in tokens)


def summarize_slide_shapes(slide) -> dict:
    """Return lightweight, deterministic heuristics about slide density."""
    summary = {
        "shape_count": len(slide.shapes),
        "text_shape_count": 0,
        "small_text_shape_count": 0,
        "narrow_text_shapes": [],
        "crowding_risks": [],
        "density": "normal",
    }

    for idx, shape in enumerate(slide.shapes, 1):
        try:
            has_text = shape.has_text_frame
        except Exception:
            has_text = False
        if not has_text:
            continue

        text = _get_all_text(shape)
        if not text:
            continue

        summary["text_shape_count"] += 1
        width_inches = _emu_to_inches(shape.width or 0)
        height_inches = _emu_to_inches(shape.height or 0)
        longest = _longest_token_len(text)
        ratio = width_inches / max(longest, 1)
        shape_info = {
            "shape_index": idx,
            "text_preview": text[:60],
            "width_inches": round(width_inches, 3),
            "height_inches": round(height_inches, 3),
            "token_width_ratio": round(ratio, 3),
        }

        if width_inches < 1.2 or height_inches < 0.45:
            summary["small_text_shape_count"] += 1
        if ratio < 0.09:
            summary["narrow_text_shapes"].append(shape_info)
        if ratio < 0.07 or (width_inches < 1.5 and longest >= 12):
            summary["crowding_risks"].append(shape_info)

    if summary["text_shape_count"] >= 8 or summary["small_text_shape_count"] >= 4:
        summary["density"] = "high"
    elif summary["text_shape_count"] <= 3 and not summary["crowding_risks"]:
        summary["density"] = "low"

    return summary


def summarize_presentation_issues(issues: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for issue in issues:
        slide_num = issue.get("slide")
        if slide_num is None:
            continue
        grouped.setdefault(slide_num, []).append(issue)
    return grouped


def _score_slide_summary(summary: dict, layout_issues: list[dict], saved_issues: list[dict]) -> int:
    return (
        len(layout_issues) * 4
        + len(saved_issues) * 5
        + len(summary.get("crowding_risks", [])) * 3
        + len(summary.get("narrow_text_shapes", [])) * 2
        + (2 if summary.get("density") == "high" else 0)
    )


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
                        width_slack = Inches(0.08)
                        needed_h = _estimate_text_height(
                            text,
                            font_size,
                            usable_w + width_slack,
                            font_family=_get_font_family(shape),
                        )
                        if needed_h > usable_h * 1.15:
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


def review_deck_artifacts(
    pptx_path: str,
    layout_issues: list[dict] | None = None,
    preview_paths: list[str] | None = None,
    slide_titles: list[str] | None = None,
) -> dict:
    """Return a merged, per-slide review artifact for a generated deck."""
    prs = PptxPresentation(pptx_path)
    layout_issues = layout_issues or []
    saved_issues = check_presentation(pptx_path)
    preview_paths = preview_paths or []
    slide_titles = slide_titles or []

    layout_by_slide = summarize_presentation_issues(layout_issues)
    saved_by_slide = summarize_presentation_issues(saved_issues)
    preview_by_slide = {
        idx + 1: path
        for idx, path in enumerate(preview_paths)
    }

    slides: list[dict] = []
    for slide_num, slide in enumerate(prs.slides, 1):
        summary = summarize_slide_shapes(slide)
        slide_entry = {
            "slide_number": slide_num,
            "title": slide_titles[slide_num - 1] if slide_num - 1 < len(slide_titles) else None,
            "layout_issues": layout_by_slide.get(slide_num, []),
            "saved_file_issues": saved_by_slide.get(slide_num, []),
            "preview_path": preview_by_slide.get(slide_num),
            "summary": summary,
        }
        slide_entry["score"] = _score_slide_summary(
            summary,
            slide_entry["layout_issues"],
            slide_entry["saved_file_issues"],
        )
        slides.append(slide_entry)

    ranked = sorted(
        (
            {
                "slide_number": slide["slide_number"],
                "title": slide["title"],
                "score": slide["score"],
            }
            for slide in slides
            if slide["score"] > 0
        ),
        key=lambda item: (-item["score"], item["slide_number"]),
    )

    return {
        "total_slides": len(prs.slides),
        "layout_issue_count": len(layout_issues),
        "saved_issue_count": len(saved_issues),
        "preview_paths": preview_paths,
        "top_problem_slides": ranked,
        "slides": slides,
    }
