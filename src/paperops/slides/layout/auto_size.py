"""Text measurement utilities for layout calculations."""

from __future__ import annotations

import logging
import os
import subprocess
from functools import lru_cache

logger = logging.getLogger(__name__)

# Font family fallback mappings
_FONT_FALLBACKS: dict[str, list[str]] = {
    "Calibri": ["Liberation Sans", "DejaVu Sans", "Arial"],
    "Consolas": ["DejaVu Sans Mono", "Liberation Mono", "Courier New"],
    "Georgia": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
}

# Cache for resolved font file paths
_font_path_cache: dict[str, str | None] = {}


@lru_cache(maxsize=64)
def _fc_list_lookup(family: str) -> str | None:
    """Use fc-list to find a font file for the given family name."""
    try:
        result = subprocess.run(
            ["fc-list", f":family={family}", "file"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # fc-list returns lines like "/path/to/font.ttf: "
            first_line = result.stdout.strip().splitlines()[0]
            path = first_line.split(":")[0].strip()
            if os.path.isfile(path):
                return path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _scan_font_dirs(family: str) -> str | None:
    """Search common font directories for a matching font file."""
    search_dirs = [
        "/usr/share/fonts/",
        "/usr/local/share/fonts/",
        os.path.expanduser("~/.local/share/fonts/"),
        os.path.expanduser("~/.fonts/"),
    ]
    family_lower = family.lower().replace(" ", "")
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            for fn in filenames:
                if fn.lower().endswith((".ttf", ".otf")):
                    fn_lower = fn.lower().replace(" ", "").replace("-", "")
                    if family_lower in fn_lower:
                        return os.path.join(dirpath, fn)
    return None


def _find_font_path(family: str) -> str | None:
    """Find the font file path for a given family, with fallback chain."""
    if family in _font_path_cache:
        return _font_path_cache[family]

    # Try the exact family first, then fallbacks
    candidates = [family] + _FONT_FALLBACKS.get(family, [])
    for name in candidates:
        path = _fc_list_lookup(name)
        if path:
            _font_path_cache[family] = path
            return path
        path = _scan_font_dirs(name)
        if path:
            _font_path_cache[family] = path
            return path

    _font_path_cache[family] = None
    return None


_pil_font_cache: dict[tuple[str, int], object] = {}


def _load_pil_font(font_family: str, font_size_pt: float):
    """Load a PIL ImageFont, with caching. Falls back to the default font."""
    cache_key = (font_family, int(font_size_pt))
    if cache_key in _pil_font_cache:
        return _pil_font_cache[cache_key]

    result = None
    try:
        from PIL import ImageFont
        path = _find_font_path(font_family)
        if path:
            try:
                result = ImageFont.truetype(path, size=int(font_size_pt))
            except (OSError, IOError):
                pass
        if result is None:
            result = ImageFont.load_default()
    except Exception as exc:
        logger.debug("Failed to load font '%s': %s", font_family, exc)

    _pil_font_cache[cache_key] = result
    return result


# Heuristic constants (inches per point)
_AVG_CHAR_WIDTH_FACTOR = 0.015   # ~0.015 inches per pt per char (proportional)
_CJK_CHAR_WIDTH_FACTOR = 0.026   # CJK characters are roughly full-width
_LINE_HEIGHT_FACTOR = 1.3        # line height = font_size * factor


def _is_cjk(ch: str) -> bool:
    """Return True if the character is a CJK ideograph or symbol."""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF       # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF    # CJK Extension A
        or 0xF900 <= cp <= 0xFAFF    # CJK Compatibility Ideographs
        or 0x3000 <= cp <= 0x303F    # CJK Symbols and Punctuation
    )


def measure_text(
    text: str,
    font_family: str,
    font_size_pt: float,
    max_width_inches: float | None = None,
) -> tuple[float, float]:
    """Measure text dimensions in inches.  Returns (width, height).

    If *max_width_inches* is set, simulate word wrapping and return the
    multi-line height.  Uses PIL.ImageFont.truetype() when available,
    falling back to character-width heuristics.
    """
    if not text:
        return (0.0, 0.0)

    font = _load_pil_font(font_family, font_size_pt)

    # --- helpers to measure a single line ---
    def _measure_line_pil(line: str) -> float:
        """Return line width in pixels using PIL font."""
        try:
            bbox = font.getbbox(line)
            return bbox[2] - bbox[0]
        except Exception:
            return font.getlength(line)

    def _px_to_inches(px: float) -> float:
        """72 DPI mapping: 1 pt = 1 px at 72 DPI, 1 inch = 72 pt."""
        return px / 72.0

    def _measure_line_heuristic(line: str) -> float:
        """Return line width in inches using CJK-aware character-count heuristic."""
        scale = font_size_pt / 12.0
        cjk_count = sum(1 for ch in line if _is_cjk(ch))
        non_cjk_count = len(line) - cjk_count
        return (non_cjk_count * _AVG_CHAR_WIDTH_FACTOR
                + cjk_count * _CJK_CHAR_WIDTH_FACTOR) * scale

    use_pil = font is not None
    line_height_inches = (font_size_pt / 72.0) * _LINE_HEIGHT_FACTOR

    # Split text into hard-break lines
    raw_lines = text.split("\n")

    if max_width_inches is None:
        # No wrapping — measure each hard line, take the widest
        max_w = 0.0
        for line in raw_lines:
            if use_pil:
                w = _px_to_inches(_measure_line_pil(line))
            else:
                w = _measure_line_heuristic(line)
            max_w = max(max_w, w)
        total_h = line_height_inches * max(len(raw_lines), 1)
        return (max_w, total_h)

    # --- word-wrap mode ---
    total_lines = 0
    max_w = 0.0

    for raw_line in raw_lines:
        if not raw_line.strip():
            total_lines += 1
            continue

        words = raw_line.split()
        current_line = ""
        for word in words:
            candidate = f"{current_line} {word}".strip() if current_line else word
            if use_pil:
                cw = _px_to_inches(_measure_line_pil(candidate))
            else:
                cw = _measure_line_heuristic(candidate)

            if cw > max_width_inches and current_line:
                # Emit current_line
                if use_pil:
                    lw = _px_to_inches(_measure_line_pil(current_line))
                else:
                    lw = _measure_line_heuristic(current_line)
                max_w = max(max_w, lw)
                total_lines += 1
                current_line = word
            else:
                current_line = candidate

        # Emit remaining
        if current_line:
            if use_pil:
                lw = _px_to_inches(_measure_line_pil(current_line))
            else:
                lw = _measure_line_heuristic(current_line)
            max_w = max(max_w, lw)
            total_lines += 1

    total_h = line_height_inches * max(total_lines, 1)
    return (min(max_w, max_width_inches), total_h)


def measure_wrapped_text_height(
    text: str,
    font_family: str,
    font_size_pt: float,
    usable_width_inches: float,
) -> float:
    """Return wrapped text height in inches for a given usable width."""
    _w, h = measure_text(
        text,
        font_family,
        font_size_pt,
        max_width_inches=max(usable_width_inches, 0.01),
    )
    return h


def estimate_min_text_width(
    text: str,
    font_family: str,
    font_size_pt: float,
) -> float:
    """Estimate the smallest readable width based on the longest token."""
    if not text:
        return 0.0

    tokens = [token for token in text.replace("\n", " ").split(" ") if token]
    sample = max(tokens, key=len) if tokens else text
    w, _h = measure_text(sample, font_family, font_size_pt)
    return w
