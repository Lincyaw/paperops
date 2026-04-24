"""Pure text measurement utilities for layout, render checks, and preview logic."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache

from paperops.slides.layout.types import Constraints, IntrinsicSize

logger = logging.getLogger(__name__)

_FONT_FALLBACKS: dict[str, list[str]] = {
    "Calibri": ["Liberation Sans", "DejaVu Sans", "Arial"],
    "Consolas": ["DejaVu Sans Mono", "Liberation Mono", "Courier New"],
    "Georgia": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
    "FangSong": ["Noto Serif CJK SC", "Source Han Serif SC", "SimSun"],
}

_font_path_cache: dict[str, str | None] = {}


@dataclass(frozen=True)
class TextStyle:
    font_family: str
    font_size_pt: float
    bold: bool = False
    italic: bool = False
    line_spacing: float = 1.25
    margin_x: float = 0.0
    margin_y: float = 0.0


@dataclass(frozen=True)
class TextMetrics:
    width: float
    height: float
    line_count: int
    lines: tuple[str, ...]
    longest_unbreakable_width: float


@dataclass(frozen=True)
class TextIntrinsic:
    min_width: float
    preferred_width: float
    min_height: float
    preferred_height: float
    metrics: TextMetrics


@lru_cache(maxsize=64)
def _fc_list_lookup(family: str) -> str | None:
    try:
        result = subprocess.run(
            ["fc-list", f":family={family}", "file"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            first_line = result.stdout.strip().splitlines()[0]
            path = first_line.split(":")[0].strip()
            if os.path.isfile(path):
                return path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _scan_font_dirs(family: str) -> str | None:
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
            for filename in filenames:
                if not filename.lower().endswith((".ttf", ".otf")):
                    continue
                normalized = filename.lower().replace(" ", "").replace("-", "")
                if family_lower in normalized:
                    return os.path.join(dirpath, filename)
    return None


def _find_font_path(family: str) -> str | None:
    if family in _font_path_cache:
        return _font_path_cache[family]

    candidates = [family] + _FONT_FALLBACKS.get(family, [])
    for name in candidates:
        path = _fc_list_lookup(name) or _scan_font_dirs(name)
        if path:
            _font_path_cache[family] = path
            return path
    _font_path_cache[family] = None
    return None


_pil_font_cache: dict[tuple[str, int], object] = {}


def _load_pil_font(font_family: str, font_size_pt: float):
    cache_key = (font_family, int(font_size_pt))
    if cache_key in _pil_font_cache:
        return _pil_font_cache[cache_key]

    font = None
    try:
        from PIL import ImageFont

        path = _find_font_path(font_family)
        if path:
            try:
                font = ImageFont.truetype(path, size=max(int(font_size_pt), 1))
            except (OSError, IOError):
                font = None
        if font is None:
            font = ImageFont.load_default()
    except Exception as exc:  # pragma: no cover - PIL missing is acceptable
        logger.debug("Failed to load font '%s': %s", font_family, exc)

    _pil_font_cache[cache_key] = font
    return font


_AVG_CHAR_WIDTH_FACTOR = 0.015
_CJK_CHAR_WIDTH_FACTOR = 0.026
_LINE_HEIGHT_FACTOR = 1.10


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0xF900 <= cp <= 0xFAFF
        or 0x3000 <= cp <= 0x303F
    )


def _is_break_punct(ch: str) -> bool:
    return ch in ",.;:!?)]}，。；：！？、）】》>"


def _tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    buffer = ""
    mode = ""
    for ch in text:
        if ch.isspace():
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append(ch)
            mode = "space"
            continue

        char_mode = "cjk" if _is_cjk(ch) else "punct" if _is_break_punct(ch) else "word"
        if char_mode == "cjk":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append(ch)
            mode = char_mode
            continue
        if char_mode == "punct":
            if buffer and mode == "word":
                buffer += ch
            else:
                if buffer:
                    tokens.append(buffer)
                buffer = ch
            mode = "word"
            continue

        if buffer and mode not in {"word", ""}:
            tokens.append(buffer)
            buffer = ch
        else:
            buffer += ch
        mode = "word"

    if buffer:
        tokens.append(buffer)
    return tokens


def _measure_line_width(line: str, font, font_size_pt: float) -> float:
    if not line:
        return 0.0
    if font is not None:
        try:
            bbox = font.getbbox(line)
            return (bbox[2] - bbox[0]) / 72.0
        except Exception:
            try:
                return font.getlength(line) / 72.0
            except Exception:
                pass

    cjk_count = sum(1 for ch in line if _is_cjk(ch))
    non_cjk_count = len(line) - cjk_count
    scale = font_size_pt / 12.0
    return (non_cjk_count * _AVG_CHAR_WIDTH_FACTOR + cjk_count * _CJK_CHAR_WIDTH_FACTOR) * scale


def _split_long_token(token: str, max_width: float, font, font_size_pt: float) -> list[str]:
    if not token:
        return [token]
    pieces: list[str] = []
    current = ""
    for ch in token:
        candidate = current + ch
        if current and _measure_line_width(candidate, font, font_size_pt) > max_width:
            pieces.append(current)
            current = ch
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [token]


def _wrap_tokens(tokens: list[str], max_width: float | None, font, font_size_pt: float) -> tuple[str, ...]:
    if max_width is None:
        return ("".join(tokens),)

    lines: list[str] = []
    current = ""
    for token in tokens:
        if token == "\n":
            lines.append(current.rstrip())
            current = ""
            continue
        if token.isspace():
            if current:
                current += token
            continue

        candidate = current + token
        if current and _measure_line_width(candidate.rstrip(), font, font_size_pt) > max_width:
            lines.append(current.rstrip())
            current = ""
            if _measure_line_width(token, font, font_size_pt) > max_width:
                token_lines = _split_long_token(token, max_width, font, font_size_pt)
                lines.extend(token_lines[:-1])
                current = token_lines[-1]
            else:
                current = token
        else:
            if not current and _measure_line_width(token, font, font_size_pt) > max_width:
                token_lines = _split_long_token(token, max_width, font, font_size_pt)
                lines.extend(token_lines[:-1])
                current = token_lines[-1]
            else:
                current = candidate
    if current or not lines:
        lines.append(current.rstrip())
    return tuple(lines)


def measure_text_metrics(text: str, style: TextStyle, max_width_inches: float | None = None) -> TextMetrics:
    if not text:
        return TextMetrics(0.0, 0.0, 0, tuple(), 0.0)

    font = _load_pil_font(style.font_family, style.font_size_pt)
    effective_line_spacing = max(style.line_spacing, _LINE_HEIGHT_FACTOR)
    line_height = (style.font_size_pt / 72.0) * effective_line_spacing

    wrapped_lines: list[str] = []
    longest_unbreakable = 0.0
    for paragraph in text.split("\n"):
        if not paragraph:
            wrapped_lines.append("")
            continue
        tokens = _tokenize_text(paragraph)
        for token in tokens:
            if token.strip():
                longest_unbreakable = max(longest_unbreakable, _measure_line_width(token, font, style.font_size_pt))
        wrapped_lines.extend(_wrap_tokens(tokens, max_width_inches, font, style.font_size_pt))

    widths = [_measure_line_width(line, font, style.font_size_pt) for line in wrapped_lines] or [0.0]
    width = min(max(widths), max_width_inches) if max_width_inches is not None and widths else max(widths)
    height = line_height * max(len(wrapped_lines), 1)
    return TextMetrics(
        width=width,
        height=height,
        line_count=len(wrapped_lines),
        lines=tuple(wrapped_lines),
        longest_unbreakable_width=longest_unbreakable,
    )


def measure_text_intrinsic(text: str, style: TextStyle, max_width_inches: float | None = None) -> TextIntrinsic:
    unconstrained = measure_text_metrics(text, style, None)
    wrapped = measure_text_metrics(text, style, max_width_inches)
    min_width = unconstrained.longest_unbreakable_width + style.margin_x
    preferred_width = (max_width_inches if max_width_inches is not None else unconstrained.width) if text else 0.0
    if max_width_inches is not None:
        preferred_width = min(unconstrained.width, max_width_inches)
    preferred_width += style.margin_x
    preferred_height = wrapped.height + style.margin_y
    effective_line_spacing = max(style.line_spacing, _LINE_HEIGHT_FACTOR)
    min_height = max((style.font_size_pt / 72.0) * effective_line_spacing + style.margin_y, 0.0)
    return TextIntrinsic(
        min_width=max(min_width, 0.0),
        preferred_width=max(preferred_width, min_width),
        min_height=min_height,
        preferred_height=max(preferred_height, min_height),
        metrics=wrapped,
    )


def build_intrinsic_size(intrinsic: TextIntrinsic) -> IntrinsicSize:
    return IntrinsicSize(
        min_width=intrinsic.min_width,
        preferred_width=intrinsic.preferred_width,
        min_height=intrinsic.min_height,
        preferred_height=intrinsic.preferred_height,
    )


def preferred_size_from_intrinsic(intrinsic: TextIntrinsic, constraints: Constraints | None = None) -> tuple[float, float]:
    if constraints is None:
        return intrinsic.preferred_width, intrinsic.preferred_height
    boxed = build_intrinsic_size(intrinsic).clamp(constraints)
    return boxed.preferred_width, boxed.preferred_height


def constraint_value(constraints: Constraints | None, axis: str) -> float | None:
    if constraints is None:
        return None
    return constraints.max_width if axis == "x" else constraints.max_height


def measure_text(
    text: str,
    font_family: str,
    font_size_pt: float,
    max_width_inches: float | None = None,
) -> tuple[float, float]:
    style = TextStyle(font_family=font_family, font_size_pt=font_size_pt)
    metrics = measure_text_metrics(text, style, max_width_inches=max_width_inches)
    return metrics.width, metrics.height


def measure_wrapped_text_height(
    text: str,
    font_family: str,
    font_size_pt: float,
    usable_width_inches: float,
) -> float:
    style = TextStyle(font_family=font_family, font_size_pt=font_size_pt)
    metrics = measure_text_metrics(text, style, max_width_inches=max(usable_width_inches, 0.01))
    return metrics.height


def estimate_min_text_width(
    text: str,
    font_family: str,
    font_size_pt: float,
) -> float:
    style = TextStyle(font_family=font_family, font_size_pt=font_size_pt)
    intrinsic = measure_text_intrinsic(text, style)
    return intrinsic.min_width
