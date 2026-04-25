"""Pure text measurement utilities for layout, render checks, and preview logic."""

from __future__ import annotations

import logging
import os
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from paperops.slides.ir.validator import (
    StructuredValidationError,
    ValidationMessage,
    ValidationReport,
)
from paperops.slides.layout.containers import (
    Absolute,
    AbsoluteItem,
    Grid,
    GridItem,
    LayoutNode,
    Padding,
)
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
    return (
        non_cjk_count * _AVG_CHAR_WIDTH_FACTOR + cjk_count * _CJK_CHAR_WIDTH_FACTOR
    ) * scale


def _split_long_token(
    token: str, max_width: float, font, font_size_pt: float
) -> list[str]:
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


def _wrap_tokens(
    tokens: list[str], max_width: float | None, font, font_size_pt: float
) -> tuple[str, ...]:
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
        if (
            current
            and _measure_line_width(candidate.rstrip(), font, font_size_pt) > max_width
        ):
            lines.append(current.rstrip())
            current = ""
            if _measure_line_width(token, font, font_size_pt) > max_width:
                token_lines = _split_long_token(token, max_width, font, font_size_pt)
                lines.extend(token_lines[:-1])
                current = token_lines[-1]
            else:
                current = token
        else:
            if (
                not current
                and _measure_line_width(token, font, font_size_pt) > max_width
            ):
                token_lines = _split_long_token(token, max_width, font, font_size_pt)
                lines.extend(token_lines[:-1])
                current = token_lines[-1]
            else:
                current = candidate
    if current or not lines:
        lines.append(current.rstrip())
    return tuple(lines)


def measure_text_metrics(
    text: str, style: TextStyle, max_width_inches: float | None = None
) -> TextMetrics:
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
                longest_unbreakable = max(
                    longest_unbreakable,
                    _measure_line_width(token, font, style.font_size_pt),
                )
        wrapped_lines.extend(
            _wrap_tokens(tokens, max_width_inches, font, style.font_size_pt)
        )

    widths = [
        _measure_line_width(line, font, style.font_size_pt) for line in wrapped_lines
    ] or [0.0]
    width = (
        min(max(widths), max_width_inches)
        if max_width_inches is not None and widths
        else max(widths)
    )
    height = line_height * max(len(wrapped_lines), 1)
    return TextMetrics(
        width=width,
        height=height,
        line_count=len(wrapped_lines),
        lines=tuple(wrapped_lines),
        longest_unbreakable_width=longest_unbreakable,
    )


def measure_text_intrinsic(
    text: str, style: TextStyle, max_width_inches: float | None = None
) -> TextIntrinsic:
    unconstrained = measure_text_metrics(text, style, None)
    wrapped = measure_text_metrics(text, style, max_width_inches)
    min_width = unconstrained.longest_unbreakable_width + style.margin_x
    preferred_width = (
        (max_width_inches if max_width_inches is not None else unconstrained.width)
        if text
        else 0.0
    )
    if max_width_inches is not None:
        preferred_width = min(unconstrained.width, max_width_inches)
    preferred_width += style.margin_x
    preferred_height = wrapped.height + style.margin_y
    effective_line_spacing = max(style.line_spacing, _LINE_HEIGHT_FACTOR)
    min_height = max(
        (style.font_size_pt / 72.0) * effective_line_spacing + style.margin_y, 0.0
    )
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


def preferred_size_from_intrinsic(
    intrinsic: TextIntrinsic, constraints: Constraints | None = None
) -> tuple[float, float]:
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
    metrics = measure_text_metrics(
        text, style, max_width_inches=max(usable_width_inches, 0.01)
    )
    return metrics.height


def estimate_min_text_width(
    text: str,
    font_family: str,
    font_size_pt: float,
) -> float:
    style = TextStyle(font_family=font_family, font_size_pt=font_size_pt)
    intrinsic = measure_text_intrinsic(text, style)
    return intrinsic.min_width


_DEFAULT_OVERFLOW = {
    "title": "shrink",
    "heading": "shrink",
    "subtitle": "shrink",
    "prose": "reflow",
    "code": "clip",
}


@dataclass(frozen=True)
class OverflowInfo:
    policy: str
    overflows: bool
    needed_height: float
    available_height: float


def _layout_children(node: LayoutNode) -> list[LayoutNode]:
    if isinstance(node, Padding):
        return [node.child] if node.child is not None else []
    if isinstance(node, Absolute):
        return [
            item.child
            for item in node.children
            if item is not None and item.child is not None
        ]
    if isinstance(node, Grid):
        return [
            item.child
            for item in node.iter_items()
            if item is not None and item.child is not None
        ]
    if isinstance(node, GridItem):
        return [node.child] if node.child is not None else []
    if isinstance(node, AbsoluteItem):
        return [node.child] if node.child is not None else []
    children = getattr(node, "children", None)
    if children is None:
        return []
    return [child for child in children if child is not None]


def _extract_text(node: Any) -> str:
    if node is None:
        return ""
    if getattr(node, "text", None):
        return str(node.text)
    if getattr(node, "props", None) and isinstance(node.props.get("text"), str):
        return str(node.props["text"])
    chunks: list[str] = []
    for child in getattr(node, "children", None) or []:
        if isinstance(child, str):
            chunks.append(child)
        else:
            chunks.append(_extract_text(child))
    return "".join(chunks)


def _style_snapshot(source_node: Any) -> dict[str, Any]:
    if source_node is None:
        return {}
    computed = getattr(source_node, "computed_style", None)
    snapshot = computed.snapshot() if computed is not None else {}
    for key, value in (getattr(source_node, "style", None) or {}).items():
        snapshot.setdefault(key, value)
    return snapshot


def overflow_policy(source_node: Any) -> str:
    style = _style_snapshot(source_node)
    if style.get("overflow") is not None:
        return str(style["overflow"])
    node_type = getattr(source_node, "type", None)
    return _DEFAULT_OVERFLOW.get(node_type, "shrink")


def measure_overflow(layout_node: LayoutNode, theme) -> OverflowInfo | None:
    source_node = getattr(layout_node, "_ir_node", None)
    region = getattr(layout_node, "_region", None)
    if source_node is None or region is None:
        return None
    text = _extract_text(source_node).strip()
    if not text:
        return None

    available_width = max(float(region.width), 0.1)
    available_height = max(float(region.height), 0.1)
    style = _style_snapshot(source_node)
    font_token = style.get("font")
    font_pt = theme.resolve_font_size(
        font_token or _DEFAULT_FONT_FOR_TYPE(getattr(source_node, "type", None))
    )
    metrics = measure_text_intrinsic(
        text,
        TextStyle(
            font_family=getattr(theme, "font_family", "Calibri"),
            font_size_pt=float(font_pt),
            line_spacing=float(style.get("line-height", 1.25) or 1.25),
            margin_x=0.0,
            margin_y=0.0,
        ),
        max_width_inches=available_width,
    )
    return OverflowInfo(
        policy=overflow_policy(source_node),
        overflows=metrics.preferred_height > available_height + 1e-6,
        needed_height=metrics.preferred_height,
        available_height=available_height,
    )


def _DEFAULT_FONT_FOR_TYPE(node_type: str | None) -> str:
    if node_type == "title":
        return "title"
    if node_type == "subtitle":
        return "subtitle"
    if node_type == "heading":
        return "heading"
    return "body"


def _collect_textual_nodes(
    layout_node: LayoutNode,
) -> list[tuple[Any, LayoutNode, OverflowInfo]]:
    matches: list[tuple[Any, LayoutNode, OverflowInfo]] = []
    for child in _layout_children(layout_node):
        matches.extend(_collect_textual_nodes(child))
    return matches


def _iter_layout_nodes(layout_node: LayoutNode):
    yield layout_node
    for child in _layout_children(layout_node):
        yield from _iter_layout_nodes(child)


def _find_reflow_target(layout_root: LayoutNode, theme):
    for node in _iter_layout_nodes(layout_root):
        info = measure_overflow(node, theme)
        if info is None or not info.overflows:
            continue
        source_node = getattr(node, "_ir_node", None)
        if source_node is None:
            continue
        if info.policy in {"shrink", "clip"}:
            continue
        return source_node, node, info
    return None


def _clone_node(node):
    from paperops.slides.ir.node import Node

    if isinstance(node, str):
        return node
    if isinstance(node, Node):
        return Node.from_dict(node.to_dict())
    return deepcopy(node)


def _replace_child_path(root, path: list[int], replacement):
    from paperops.slides.ir.node import Node

    if not path:
        return replacement
    children = list(root.children or [])
    index = path[0]
    child = children[index]
    if not isinstance(child, Node):
        raise TypeError("Expected node child while cloning reflow slide")
    children[index] = _replace_child_path(child, path[1:], replacement)
    return Node(
        type=root.type,
        class_=root.class_,
        id=root.id,
        style=dict(root.style) if root.style is not None else None,
        text=root.text,
        props=dict(root.props) if root.props is not None else None,
        children=children,
    )


def _find_child_path(root, target, prefix: list[int] | None = None):
    from paperops.slides.ir.node import Node

    if prefix is None:
        prefix = []
    if root is target:
        return prefix
    for index, child in enumerate(root.children or []):
        if isinstance(child, Node):
            path = _find_child_path(child, target, [*prefix, index])
            if path is not None:
                return path
    return None


def _continuation_suffix(meta: dict[str, Any] | None) -> str:
    lang = str((meta or {}).get("lang", "")).lower()
    if lang.startswith("zh"):
        return "(续)"
    return "(cont.)"


def _apply_title_suffix(slide_node, suffix: str):
    from paperops.slides.ir.node import Node

    children = list(slide_node.children or [])
    updated = False
    new_children: list[Any] = []
    for child in children:
        if not isinstance(child, Node):
            new_children.append(child)
            continue
        if not updated and child.type == "title":
            title_text = _extract_text(child).strip()
            if suffix not in title_text:
                child = Node(
                    type=child.type,
                    class_=child.class_,
                    id=child.id,
                    style=dict(child.style) if child.style is not None else None,
                    text=(title_text + f" {suffix}").strip(),
                    props=dict(child.props) if child.props is not None else None,
                    children=child.children,
                )
            updated = True
        new_children.append(child)
    return Node(
        type=slide_node.type,
        class_=slide_node.class_,
        id=slide_node.id,
        style=dict(slide_node.style) if slide_node.style is not None else None,
        text=slide_node.text,
        props=dict(slide_node.props) if slide_node.props is not None else None,
        children=new_children,
    )


def _split_reflow_content(
    node, layout_node: LayoutNode, theme
) -> tuple[Any, Any, bool]:
    from paperops.slides.ir.node import Node

    segments: list[Any]
    if node.children:
        segments = list(node.children)
    else:
        text = _extract_text(node)
        parts = [part.strip() for part in text.split("\n\n") if part.strip()]
        segments = parts if len(parts) > 1 else [text]

    if len(segments) <= 1:
        return node, None, True

    region = getattr(layout_node, "_region", None)
    style = _style_snapshot(node)
    available_width = max(float(region.width), 0.1)
    available_height = max(float(region.height), 0.1)
    font_pt = theme.resolve_font_size(
        style.get("font") or _DEFAULT_FONT_FOR_TYPE(getattr(node, "type", None))
    )
    text_style = TextStyle(
        font_family=getattr(theme, "font_family", "Calibri"),
        font_size_pt=float(font_pt),
        line_spacing=float(style.get("line-height", 1.25) or 1.25),
    )

    def fits(count: int) -> bool:
        content = "".join(
            _extract_text(item) if not isinstance(item, str) else item
            for item in segments[:count]
        )
        intrinsic = measure_text_intrinsic(
            content, text_style, max_width_inches=available_width
        )
        return intrinsic.preferred_height <= available_height + 1e-6

    lo, hi = 1, len(segments)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best <= 0:
        return node, None, True
    if best >= len(segments):
        return node, None, False

    first = segments[:best]
    rest = segments[best:]
    front = Node(
        type=node.type,
        class_=node.class_,
        id=node.id,
        style=dict(node.style) if node.style is not None else None,
        text=(
            node.text
            if node.children is not None
            else ("\n\n".join(str(item) for item in first))
        ),
        props=dict(node.props) if node.props is not None else None,
        children=first if node.children is not None else None,
    )
    back = Node(
        type=node.type,
        class_=node.class_,
        id=node.id,
        style=dict(node.style) if node.style is not None else None,
        text=(
            node.text
            if node.children is not None
            else ("\n\n".join(str(item) for item in rest))
        ),
        props=dict(node.props) if node.props is not None else None,
        children=rest if node.children is not None else None,
    )
    return front, back, False


def resolve_overflow(
    slide_layouts: list[tuple[Any, LayoutNode]],
    *,
    theme,
    meta: dict[str, Any] | None = None,
) -> list[tuple[Any, LayoutNode]]:
    resolved: list[tuple[Any, LayoutNode]] = []
    queue = list(slide_layouts)

    while queue:
        slide_node, layout_root = queue.pop(0)
        reflow_target = _find_reflow_target(layout_root, theme)
        if reflow_target is None:
            resolved.append((slide_node, layout_root))
            continue

        source_node, target_layout, overflow = reflow_target
        policy = overflow.policy
        if policy == "shrink" or policy == "clip":
            resolved.append((slide_node, layout_root))
            continue
        if policy == "error":
            report = ValidationReport(
                errors=[
                    ValidationMessage(
                        code="OVERFLOW_UNRECOVERABLE",
                        message=(
                            f"Text overflows its region: needs {overflow.needed_height:.2f}in, "
                            f"has {overflow.available_height:.2f}in"
                        ),
                        path=getattr(target_layout, "_node_path", "slide"),
                        severity="error",
                        meta={
                            "needed_height": round(overflow.needed_height, 4),
                            "available_height": round(overflow.available_height, 4),
                        },
                    )
                ]
            )
            raise StructuredValidationError(report)

        front, back, fallback_shrink = _split_reflow_content(
            source_node, target_layout, theme
        )
        if fallback_shrink or back is None:
            source_style = dict(getattr(source_node, "style", None) or {})
            source_style.setdefault("overflow", "shrink")
            source_style.setdefault("overflow-warning", "reflow-fallback-shrink")
            from paperops.slides.ir.node import Node

            replacement = Node(
                type=source_node.type,
                class_=source_node.class_,
                id=source_node.id,
                style=source_style,
                text=source_node.text,
                props=(
                    dict(source_node.props) if source_node.props is not None else None
                ),
                children=source_node.children,
            )
            path = _find_child_path(slide_node, source_node)
            if path is None:
                resolved.append((slide_node, layout_root))
            else:
                resolved.append(
                    (_replace_child_path(slide_node, path, replacement), None)
                )
            continue

        path = _find_child_path(slide_node, source_node)
        if path is None:
            resolved.append((slide_node, layout_root))
            continue

        first_slide = _replace_child_path(slide_node, path, front)
        continued_slide = _replace_child_path(_clone_node(slide_node), path, back)
        continued_slide = _apply_title_suffix(
            continued_slide, _continuation_suffix(meta)
        )
        resolved.append((first_slide, None))
        queue.insert(0, (continued_slide, None))

    return resolved
