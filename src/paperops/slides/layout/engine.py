"""Layout engine — resolve a layout tree into absolute slide regions."""

from __future__ import annotations

from typing import Any

from paperops.slides.core.constants import CONTENT_REGION, Region, SLIDE_HEIGHT, SLIDE_WIDTH
from paperops.slides.ir.node import Node
from paperops.slides.components.table import Table as LayoutTable
from paperops.slides.components.text import TextBlock
from paperops.slides.components.shapes import Box as ShapeBox
from paperops.slides.components.shapes import RoundedBox
from paperops.slides.layout.containers import Absolute, AbsoluteItem, Flex, Grid, GridItem, HStack, Layer, LayoutNode, Padding
from paperops.slides.layout.types import Constraints, IntrinsicSize, LayoutIssue, TrackSpec


def build_layout_tree(
    node: Node,
    theme,
    *,
    region: Region = CONTENT_REGION,
) -> LayoutNode:
    """Convert an IR node tree into a layout node tree."""
    layout_root = _ir_node_to_layout(node, theme)
    return layout_root


def compute_layout(
    root: Node | LayoutNode,
    region: Region,
    theme,
    *,
    slide: int | None = None,
    root_path: str = "root",
) -> list[dict]:
    """Resolve a component tree into absolute regions and return layout issues."""
    layout_root = root
    if isinstance(root, Node):
        layout_root = _ir_node_to_layout(root, theme)
    if not isinstance(layout_root, LayoutNode):
        raise TypeError("compute_layout expects a Node or LayoutNode")

    issues: list[LayoutIssue] = []
    _layout(layout_root, region, theme, issues, root_path, slide)
    return [issue.to_dict() for issue in issues]


def _resolve_layout_metrics_from_style(node: Node) -> dict[str, Any]:
    style = {}
    if getattr(node, "computed_style", None) is not None:
        style = dict(node.computed_style.snapshot())
    if style is None:
        style = {}
    for key, value in (node.style or {}).items():
        style.setdefault(key, value)
    return style


def _apply_style_to_layout_node(
    layout_node: LayoutNode,
    source_node: Node | None,
) -> None:
    if source_node is None:
        return
    style = _resolve_layout_metrics_from_style(source_node)

    if "width" in style:
        _as_float(layout_node, "width", style.get("width"))
    if "height" in style:
        _as_float(layout_node, "height", style.get("height"))
    if "min-width" in style:
        _as_float(layout_node, "min_width", style.get("min-width"))
    if "min-height" in style:
        _as_float(layout_node, "min_height", style.get("min-height"))
    if "max-width" in style:
        _as_float(layout_node, "max_width", style.get("max-width"))
    if "max-height" in style:
        _as_float(layout_node, "max_height", style.get("max-height"))

    if "grow" in style:
        _as_float(layout_node, "grow", style.get("grow"))
    if "shrink" in style:
        _as_float(layout_node, "shrink", style.get("shrink"))
    if "basis" in style:
        _as_float(layout_node, "basis", style.get("basis"))
    if "wrap" in style:
        layout_node.wrap = bool(style["wrap"])

    if isinstance(layout_node, Flex):
        if "justify" in style:
            layout_node.justify = str(style["justify"])
        if "align" in style:
            layout_node.align = str(style["align"])
        layout_node.gap = _as_float(layout_node, "gap", style.get("gap"), fallback=layout_node.gap)
        if layout_node.gap is not None:
            layout_node.gap = float(layout_node.gap)
        row_gap = style.get("row-gap")
        col_gap = style.get("column-gap")
        if row_gap is not None:
            layout_node.row_gap = float(row_gap) if isinstance(row_gap, (int, float)) else None
        if col_gap is not None:
            layout_node.column_gap = float(col_gap) if isinstance(col_gap, (int, float)) else None

    if isinstance(layout_node, Grid):
        layout_node.columns = _parse_grid_tracks(style.get("cols"))
        layout_node.rows = _parse_grid_tracks(style.get("rows"))
        if isinstance(style.get("cols"), int):
            layout_node.cols = int(style["cols"])


def _as_float(layout_node: LayoutNode, attr: str, value: Any, *, fallback: float | None = None) -> float | None:
    if value is None:
        if fallback is not None:
            setattr(layout_node, attr, fallback)
        return None
    if value in {"auto", "inherit", "none", None}:
        return None
    if isinstance(value, (int, float)):
        setattr(layout_node, attr, float(value))
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    setattr(layout_node, attr, number)
    return number


def _as_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return str(value)


def _parse_grid_tracks(raw: Any) -> list[TrackSpec] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, tuple):
        values = list(raw)
    elif isinstance(raw, (int, float)):
        count = int(raw)
        if count <= 0:
            return None
        return [TrackSpec("fr", 1.0) for _ in range(count)]
    elif isinstance(raw, str):
        values = raw.split()
    else:
        return None

    tracks: list[TrackSpec] = []
    for token in values:
        if token in {"auto", "inherit", "none"}:
            tracks.append(TrackSpec("auto", 0.0))
            continue
        text = str(token).strip()
        if not text:
            continue
        if text.endswith("fr"):
            number = text[:-2]
            try:
                tracks.append(TrackSpec("fr", float(number) if number else 1.0))
            except ValueError:
                tracks.append(TrackSpec("auto", 0.0))
            continue
        try:
            tracks.append(TrackSpec("fixed", float(text)))
            continue
        except ValueError:
            tracks.append(TrackSpec("auto", 0.0))
    return tracks or None


def _extract_node_text(node: Node) -> str:
    if node.props and isinstance(node.props.get("text"), str):
        return str(node.props["text"])
    if node.text is not None:
        return str(node.text)
    if not node.children:
        return ""
    chunks: list[str] = []
    for child in node.children:
        if isinstance(child, str):
            chunks.append(child)
        elif isinstance(child, Node):
            chunks.append(_extract_node_text(child))
    return "".join(chunks)


def _build_style_font_size(style: dict[str, Any], theme) -> float:
    value = style.get("font")
    if value is None:
        return float(theme.fonts.get("body", 18))
    if isinstance(value, (int, float)):
        return float(value)
    return float(theme.resolve_font_size(value))


def _build_style_text_align(style: dict[str, Any]) -> str:
    value = style.get("text-align", "left")
    if value in {"left", "center", "right"}:
        return str(value)
    return "left"


def _build_layout_leaf(node: Node, theme) -> LayoutNode:
    style = _resolve_layout_metrics_from_style(node)
    text = _extract_node_text(node)
    node_type = node.type
    props = dict(node.props or {})

    if node_type in {"box", "kpi"}:
        props = dict(node.props or {})
        if text:
            props.setdefault("text", text)
        if node_type == "kpi":
            label = props.get("label", "")
            value = props.get("value", "")
            delta = props.get("delta", "")
            if delta:
                text = f"{label}: {value} ({delta})"
            else:
                text = f"{label}: {value}"
            props["text"] = text
        layout = ShapeBox(
            text=str(props.get("text", "")),
            color=str(style.get("bg", "bg_alt")),
            border=str(style.get("border", "border")),
            text_color=str(style.get("color", "text")),
            font_size=_build_style_font_size(style, theme),
            bold=_is_bold(style.get("font-weight")),
            align=_build_style_text_align(style),
        )
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    if node_type == "roundedbox":
        text = text or _as_text(props.get("text"), "")
        raw_radius = style.get("radius")
        if isinstance(raw_radius, (int, float)):
            radius = float(raw_radius)
        else:
            radius = _safe_float(raw_radius)
        if radius is None:
            radius = 0.08
        layout = RoundedBox(
            text=text,
            color=str(style.get("bg", "bg_alt")),
            text_color=str(style.get("color", "text")),
            font_size=_build_style_font_size(style, theme),
            bold=_is_bold(style.get("font-weight")),
            align=_build_style_text_align(style),
            radius=radius,
        )
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    if node_type in {"image", "svg", "icon"}:
        text = (
            text
            or _as_text(props.get("name"), "")
            or _as_text(props.get("body"), "")
            or _as_text(props.get("src"), "")
            or f"{node_type}"
        )
        layout = TextBlock(
            text=_as_text(f"[{node_type}] ") + text,
            font_size=_build_style_font_size(style, theme),
            color=str(style.get("color", "text")),
            align=_build_style_text_align(style),
            bold=False,
            italic=False,
            line_spacing=float(style.get("line-height", 1.25)),
            margin_x=float(style.get("padding-left", 0.0) or 0.0),
            margin_y=float(style.get("padding-top", 0.0) or 0.0),
        )
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    if node_type == "chart":
        title = _as_text(props.get("title"), "")
        chart_type = _as_text(props.get("chart_type"), "chart")
        text = title or f"{chart_type} chart"
        layout = TextBlock(
            text=text,
            font_size=_build_style_font_size(style, theme),
            color=str(style.get("color", "text")),
            align=_build_style_text_align(style),
            bold=True,
            italic=False,
            line_spacing=float(style.get("line-height", 1.25)),
            margin_x=float(style.get("padding-left", 0.0) or 0.0),
            margin_y=float(style.get("padding-top", 0.0) or 0.0),
        )
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    if node_type == "table":
        rows = props.get("rows", [])
        headers = props.get("headers", [])
        layout = LayoutTable(
            headers=headers if isinstance(headers, list) else [],
            rows=rows if isinstance(rows, list) else [],
        )
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    if node_type in {"line", "arrow", "divider"}:
        text = "—" if text == "" else text
        layout = TextBlock(
            text=_as_text(text),
            font_size=_build_style_font_size(style, theme),
            color=str(style.get("color", "text")),
            align=_build_style_text_align(style),
            bold=False,
            italic=False,
            line_spacing=float(style.get("line-height", 1.25)),
            margin_x=float(style.get("padding-left", 0.0) or 0.0),
            margin_y=float(style.get("padding-top", 0.0) or 0.0),
        )
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    if node_type == "spacer":
        layout = LayoutNode()
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    if node_type == "note":
        layout = LayoutNode()
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    # Default text-like leaf
    layout = TextBlock(
        text=text,
        font_size=_build_style_font_size(style, theme),
        color=str(style.get("color", "text")),
        align=_build_style_text_align(style),
        bold=_is_bold(style.get("font-weight")),
        italic=bool(str(style.get("font-style", "")).lower() == "italic"),
        line_spacing=float(style.get("line-height", 1.25)),
        margin_x=float(style.get("padding-left", 0.0) or 0.0),
        margin_y=float(style.get("padding-top", 0.0) or 0.0),
    )
    _apply_style_to_layout_node(layout, node)
    _attach_source(layout, node)
    return layout


def _build_layout_children(node: Node, theme) -> list[LayoutNode]:
    children: list[LayoutNode] = []
    for child in node.children or []:
        if isinstance(child, str):
            children.append(
                TextBlock(
                    text=child,
                    font_size=float(theme.fonts.get("body", 18)),
                    align="left",
                    color=str((node.computed_style.snapshot() if getattr(node, "computed_style", None) else {}).get("color", "text")),
                )
            )
            continue
        children.append(_ir_node_to_layout(child, theme))
    return children


def _ir_node_to_layout(node: Node, theme) -> LayoutNode:
    node_type = node.type.lower()
    if node_type == "slide":
        layout = Layer(children=_build_layout_children(node, theme))
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    if node_type in {"flex", "hstack", "stack"}:
        children = _build_layout_children(node, theme)
        gap = _resolve_layout_metrics_from_style(node).get("gap")
        layout = Flex(
            direction="row" if node_type != "stack" else "column",
            children=children,
            gap=float(gap) if isinstance(gap, (int, float)) else 0.3,
        )
        if node_type == "stack":
            layout.direction = "column"
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    if node_type in {"vstack"}:
        children = _build_layout_children(node, theme)
        layout = Flex(direction="column", children=children)
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    if node_type == "grid":
        children = _build_layout_children(node, theme)
        # Convert explicit textual children into generic positioned entries if needed.
        grid_children = []
        for child in children:
            if isinstance(child, LayoutNode):
                grid_children.append(child)
            else:
                grid_children.append(child)
        layout = Grid(children=grid_children)
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    if node_type == "layer":
        layout = Layer(children=_build_layout_children(node, theme))
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    if node_type == "absolute":
        children: list[AbsoluteItem] = []
        for child in node.children or []:
            if isinstance(child, str):
                item_style = _resolve_layout_metrics_from_style(node)
                item_child = TextBlock(text=child, font_size=float(theme.fonts.get("body", 18)))
            elif isinstance(child, Node):
                item_style = _resolve_layout_metrics_from_style(child)
                item_child = _ir_node_to_layout(child, theme)
            else:
                continue
            left = _to_float(item_style.get("left"), fallback=0.0)
            top = _to_float(item_style.get("top"), fallback=0.0)
            width = item_style.get("width")
            height = item_style.get("height")
            child_layout = AbsoluteItem(
                child=item_child,
                left=left,
                top=top,
                width=float(width) if isinstance(width, (int, float)) else None,
                height=float(height) if isinstance(height, (int, float)) else None,
            )
            _attach_source(child_layout, child if isinstance(child, Node) else node)
            children.append(child_layout)
        layout = Absolute(children=children)
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    if node_type == "padding":
        padding_value = _resolve_layout_metrics_from_style(node)
        style = padding_value
        all_padding = _float_or_none(style.get("padding"))
        child_layout: LayoutNode | None = None
        if node.children:
            first = node.children[0]
            if isinstance(first, Node):
                child_layout = _ir_node_to_layout(first, theme)
        layout = Padding(
            child=child_layout,
            all=all_padding,
            left=float(style.get("padding-left") or 0.0) if _is_numeric(style.get("padding-left")) else None,
            right=float(style.get("padding-right") or 0.0) if _is_numeric(style.get("padding-right")) else None,
            top=float(style.get("padding-top") or 0.0) if _is_numeric(style.get("padding-top")) else None,
            bottom=float(style.get("padding-bottom") or 0.0) if _is_numeric(style.get("padding-bottom")) else None,
        )
        _apply_style_to_layout_node(layout, node)
        _attach_source(layout, node)
        return layout

    if node_type in {"text", "title", "subtitle", "heading"}:
        layout = _build_layout_leaf(node, theme)
        return layout

    if node_type == "box" or node_type == "kpi":
        return _build_layout_leaf(node, theme)

    layout = _build_layout_leaf(node, theme)
    _attach_source(layout, node)
    return layout


def _attach_source(layout_node: LayoutNode, source_node: Node | None) -> None:
    setattr(layout_node, "_ir_node", source_node)
    return None


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float))


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_bold(font_weight: Any) -> bool:
    if font_weight in {None, "inherit", "auto"}:
        return False
    if isinstance(font_weight, bool):
        return font_weight
    if isinstance(font_weight, (int, float)):
        return float(font_weight) >= 600
    if isinstance(font_weight, str):
        return font_weight.lower() in {"bold", "bolder"}
    return False


def _to_float(value: Any, *, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    if value in {"auto", "inherit", "none"}:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Any) -> float | None:
    if value in {"auto", "inherit", "none", None}:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def _layout(
    node: LayoutNode,
    region: Region,
    theme,
    issues: list[LayoutIssue],
    node_path: str,
    slide: int | None,
) -> None:
    node._region = region
    node._node_path = node_path

    if isinstance(node, Padding):
        _layout_padding(node, region, theme, issues, node_path, slide)
        return
    if isinstance(node, Absolute):
        _layout_absolute(node, region, theme, issues, node_path, slide)
        return
    if isinstance(node, Layer):
        _layout_layer(node, region, theme, issues, node_path, slide)
        return
    if isinstance(node, Grid):
        _layout_grid(node, region, theme, issues, node_path, slide)
        return
    if isinstance(node, Flex):
        _layout_flex(node, region, theme, issues, node_path, slide)
        return

    _validate_region(node, issues, node_path, slide)


def _layout_padding(node: Padding, region: Region, theme, issues: list[LayoutIssue], node_path: str, slide: int | None) -> None:
    _validate_region(node, issues, node_path, slide)
    if node.child is None:
        return
    inner = Region(
        left=region.left + node._left,
        top=region.top + node._top,
        width=max(region.width - node._left - node._right, 0.0),
        height=max(region.height - node._top - node._bottom, 0.0),
    )
    _layout(node.child, inner, theme, issues, f"{node_path}.child", slide)


def _layout_layer(node: Layer, region: Region, theme, issues: list[LayoutIssue], node_path: str, slide: int | None) -> None:
    _validate_region(node, issues, node_path, slide)
    for index, child in enumerate(node.children):
        _layout(child, region, theme, issues, f"{node_path}.children[{index}]", slide)


def _layout_absolute(node: Absolute, region: Region, theme, issues: list[LayoutIssue], node_path: str, slide: int | None) -> None:
    _validate_region(node, issues, node_path, slide)
    for index, item in enumerate(node.children):
        child_path = f"{node_path}.children[{index}]"
        item._region = Region(left=region.left + item.left, top=region.top + item.top, width=0.0, height=0.0)
        item._node_path = child_path
        if item.child is None:
            _validate_region(item, issues, child_path, slide)
            continue
        intrinsic = item.child.measure(
            Constraints(
                max_width=item.width if item.width is not None else max(region.width - item.left, 0.0),
                max_height=item.height if item.height is not None else max(region.height - item.top, 0.0),
            ),
            theme,
        )
        child_region = Region(
            left=region.left + item.left,
            top=region.top + item.top,
            width=item.width if item.width is not None else intrinsic.preferred_width,
            height=item.height if item.height is not None else intrinsic.preferred_height,
        )
        _layout(item.child, child_region, theme, issues, f"{child_path}.child", slide)


def _layout_flex(node: Flex, region: Region, theme, issues: list[LayoutIssue], node_path: str, slide: int | None) -> None:
    _validate_region(node, issues, node_path, slide)
    if not node.children:
        return

    if node.direction == "row" and (node.wrap or node.wrap_mode == "wrap"):
        _layout_wrapped_row(node, region, theme, issues, node_path, slide)
        return

    gap = node.main_gap() or 0.0
    is_row = node.direction == "row"
    available_main = (region.width if is_row else region.height) - gap * max(len(node.children) - 1, 0)
    resolved = _resolve_flex_main_sizes(node, theme, available_main, region, issues, node_path, slide)
    used_main = sum(resolved)
    remaining_main = max(available_main - used_main, 0.0)

    offset = 0.0
    distribute_gap = gap
    if node.justify == "center":
        offset = remaining_main / 2
    elif node.justify == "end":
        offset = remaining_main
    elif node.justify == "space-between" and len(node.children) > 1:
        distribute_gap = gap + remaining_main / (len(node.children) - 1)

    cursor = (region.left if is_row else region.top) + offset
    for index, child in enumerate(node.children):
        child_path = f"{node_path}.children[{index}]"
        child_main = resolved[index]
        child_cross = _resolve_cross_size(child, node, region, theme, child_main)
        child_pos_cross = _resolve_cross_position(child, node, region, child_cross)
        if is_row:
            child_region = Region(left=cursor, top=child_pos_cross, width=child_main, height=child_cross)
        else:
            child_region = Region(left=child_pos_cross, top=cursor, width=child_cross, height=child_main)
        _layout(child, child_region, theme, issues, child_path, slide)
        cursor += child_main + distribute_gap


def _layout_wrapped_row(node: Flex, region: Region, theme, issues: list[LayoutIssue], node_path: str, slide: int | None) -> None:
    lines: list[list[tuple[int, LayoutNode, IntrinsicSize, float]]] = []
    current_line: list[tuple[int, LayoutNode, IntrinsicSize, float]] = []
    current_width = 0.0
    gap = node.main_gap() or 0.0

    for index, child in enumerate(node.children):
        intrinsic = child.measure(_main_probe_constraints(child, node, region), theme)
        child_width = _preferred_main_size(child, intrinsic, axis="x")
        next_width = child_width if not current_line else current_width + gap + child_width
        if current_line and next_width > region.width + 1e-6:
            lines.append(current_line)
            current_line = [(index, child, intrinsic, child_width)]
            current_width = child_width
        else:
            current_line.append((index, child, intrinsic, child_width))
            current_width = next_width
    if current_line:
        lines.append(current_line)

    y = region.top
    row_gap = node.cross_gap()
    for line_index, line in enumerate(lines):
        line_children = [child for _, child, _, _ in line]
        line_node = Flex(
            direction="row",
            justify=node.justify,
            align=node.align,
            gap=node.main_gap(),
            children=line_children,
        )
        line_height = max(
            (_measured_cross_size(child, "y", child_width, region.height, theme) for _, child, _intrinsic, child_width in line),
            default=0.0,
        )
        if y + line_height > region.bottom + 1e-6:
            issues.append(LayoutIssue(
                code="row_wrap_overflow",
                message="Wrapped flex rows exceed the available container height.",
                severity="warning",
                slide=slide,
                node_path=node_path,
                region=region,
            ))
            break
        line_region = Region(left=region.left, top=y, width=region.width, height=max(line_height, 0.0))
        _layout_flex(line_node, line_region, theme, issues, f"{node_path}.line[{line_index}]", slide)
        for child_index, (original_index, child, _intrinsic, _child_width) in enumerate(line):
            source_child = line_children[child_index]
            child._region = source_child._region
            child._node_path = f"{node_path}.children[{original_index}]"
            _validate_region(child, issues, child._node_path, slide)
        y += line_height + row_gap


def _layout_grid(node: Grid, region: Region, theme, issues: list[LayoutIssue], node_path: str, slide: int | None) -> None:
    _validate_region(node, issues, node_path, slide)
    items = node.iter_items()
    if not items:
        return

    columns = node.resolved_columns()
    rows = node.resolved_rows()
    column_gap = node._column_gap()
    row_gap = node._row_gap()
    available_width = max(region.width - column_gap * max(len(columns) - 1, 0), 0.0)
    column_sizes = _resolve_grid_tracks(columns, available_width, items, axis="x", theme=theme)
    row_sizes = _resolve_grid_rows(rows, items, column_sizes, column_gap, theme)

    content_height = sum(row_sizes) + row_gap * max(len(row_sizes) - 1, 0)
    if content_height > region.height + 1e-6:
        issues.append(LayoutIssue(
            code="grid_overconstrained",
            message="Grid content exceeds the available container height.",
            severity="warning",
            slide=slide,
            node_path=node_path,
            region=region,
        ))

    column_offsets = _accumulate_offsets(region.left, column_sizes, column_gap)
    row_offsets = _accumulate_offsets(region.top, row_sizes, row_gap)

    for index, item in enumerate(items):
        child = item.child
        if child is None:
            continue
        left = column_offsets[item.col]
        top = row_offsets[item.row]
        width = sum(column_sizes[item.col:item.col + item.col_span]) + column_gap * max(item.col_span - 1, 0)
        height = sum(row_sizes[item.row:item.row + item.row_span]) + row_gap * max(item.row_span - 1, 0)
        _layout(child, Region(left=left, top=top, width=width, height=height), theme, issues, f"{node_path}.items[{index}]", slide)


def _resolve_flex_main_sizes(
    node: Flex,
    theme,
    available_main: float,
    region: Region,
    issues: list[LayoutIssue],
    node_path: str,
    slide: int | None,
) -> list[float]:
    is_row = node.direction == "row"
    sizes = [0.0 for _ in node.children]
    min_sizes = [0.0 for _ in node.children]
    fill_indices: list[int] = []
    used = 0.0

    for index, child in enumerate(node.children):
        intrinsic = child.measure(_main_probe_constraints(child, node, region), theme)
        preferred = _preferred_main_size(child, intrinsic, "x" if is_row else "y")
        minimum = _minimum_main_size(child, intrinsic, "x" if is_row else "y")
        min_sizes[index] = minimum
        mode = _size_mode(child, "x" if is_row else "y")
        if mode == "fill":
            fill_indices.append(index)
            base = child.basis if child.basis is not None else minimum
            sizes[index] = base
            used += base
        else:
            sizes[index] = preferred
            used += preferred

    remaining = available_main - used
    if remaining > 0 and fill_indices:
        grow_total = sum(max(node.children[index].grow, 1.0) for index in fill_indices)
        for index in fill_indices:
            share = max(node.children[index].grow, 1.0) / grow_total
            sizes[index] += remaining * share
    elif remaining < 0:
        overflow = -remaining
        overflow = _shrink_to_fit(node, sizes, min_sizes, overflow)
        if sum(sizes) > available_main + 1e-6:
            issues.append(LayoutIssue(
                code="overconstrained",
                message="Flex children reached min size but still exceed available space.",
                severity="warning",
                slide=slide,
                node_path=node_path,
                region=region,
            ))

    return [max(size, 0.0) for size in sizes]


def _resolve_cross_size(child: LayoutNode, node: Flex, region: Region, theme, main_size: float) -> float:
    axis = "y" if node.direction == "row" else "x"
    explicit = child.height if axis == "y" else child.width
    available_cross = region.height if axis == "y" else region.width
    if explicit is not None:
        return min(explicit, available_cross)
    if node.align == "stretch" or _size_mode(child, axis) == "fill":
        return available_cross
    return _measured_cross_size(child, axis, main_size, available_cross, theme)


def _resolve_cross_position(child: LayoutNode, node: Flex, region: Region, child_cross: float) -> float:
    if node.direction == "row":
        available = region.height
        base = region.top
    else:
        available = region.width
        base = region.left

    align = node.align
    if align == "center":
        return base + (available - child_cross) / 2
    if align == "end":
        return base + available - child_cross
    return base


def _size_mode(node: LayoutNode, axis: str) -> str:
    explicit = node.width if axis == "x" else node.height
    if explicit is not None:
        return "fixed"
    mode = node.size_mode_x if axis == "x" else node.size_mode_y
    if mode == "stretch":
        return "fill"
    return "fit" if mode == "auto" else mode


def _preferred_main_size(node: LayoutNode, intrinsic: IntrinsicSize, axis: str) -> float:
    if node.basis is not None:
        return node.basis
    if axis == "x":
        return node.width if node.width is not None else intrinsic.preferred_width
    return node.height if node.height is not None else intrinsic.preferred_height


def _minimum_main_size(node: LayoutNode, intrinsic: IntrinsicSize, axis: str) -> float:
    if axis == "x":
        if node.width is not None:
            return node.width
        if node.min_width is not None:
            return node.min_width
        return intrinsic.min_width
    if node.height is not None:
        return node.height
    if node.min_height is not None:
        return node.min_height
    return intrinsic.min_height


def _resolve_grid_tracks(tracks: list[TrackSpec], available: float, items: list[GridItem], axis: str, theme) -> list[float]:
    sizes = [0.0 for _ in tracks]
    fr_indices: list[int] = []
    fr_total = 0.0
    for index, track in enumerate(tracks):
        if track.kind == "fixed":
            sizes[index] = track.value
        elif track.kind == "fr":
            fr_indices.append(index)
            fr_total += max(track.value, 0.0)

    for item in items:
        child = item.child
        if child is None:
            continue
        intrinsic = child.measure(Constraints(max_width=max(available, 0.0)), theme)
        preferred = intrinsic.preferred_width if axis == "x" else intrinsic.preferred_height
        start = item.col if axis == "x" else item.row
        span = item.col_span if axis == "x" else item.row_span
        per_track = preferred / max(span, 1)
        for track_index in range(start, min(start + span, len(tracks))):
            if tracks[track_index].kind == "auto":
                sizes[track_index] = max(sizes[track_index], per_track)

    remaining = max(available - sum(sizes), 0.0)
    if fr_indices and fr_total > 0:
        for index in fr_indices:
            sizes[index] = remaining * (tracks[index].value / fr_total)

    return sizes


def _resolve_grid_rows(
    rows: list[TrackSpec],
    items: list[GridItem],
    column_sizes: list[float],
    column_gap: float,
    theme,
) -> list[float]:
    sizes = [0.0 for _ in rows]
    full_width = sum(column_sizes)
    for row_index, row in enumerate(rows):
        if row.kind == "fixed":
            sizes[row_index] = row.value

    for item in items:
        child = item.child
        if child is None:
            continue
        available_width = (
            sum(column_sizes[item.col:item.col + item.col_span])
            + column_gap * max(item.col_span - 1, 0)
        ) or full_width
        intrinsic = child.measure(Constraints(max_width=available_width), theme)
        per_row = intrinsic.preferred_height / max(item.row_span, 1)
        for row_index in range(item.row, min(item.row + item.row_span, len(rows))):
            if rows[row_index].kind != "fixed":
                sizes[row_index] = max(sizes[row_index], per_row)
    return sizes


def _main_probe_constraints(child: LayoutNode, node: Flex, region: Region) -> Constraints:
    if node.direction == "row":
        return Constraints(
            max_width=child.width,
            max_height=child.height if child.height is not None else region.height,
        )
    return Constraints(
        max_width=child.width if child.width is not None else region.width,
        max_height=child.height,
    )


def _cross_probe_constraints(axis: str, main_size: float, available_cross: float, child: LayoutNode) -> Constraints:
    if axis == "y":
        return Constraints(
            max_width=child.width if child.width is not None else max(main_size, 0.0),
            max_height=child.height if child.height is not None else available_cross,
        )
    return Constraints(
        max_width=child.width if child.width is not None else available_cross,
        max_height=child.height if child.height is not None else max(main_size, 0.0),
    )


def _measured_cross_size(child: LayoutNode, axis: str, main_size: float, available_cross: float, theme) -> float:
    intrinsic = child.measure(_cross_probe_constraints(axis, main_size, available_cross, child), theme)
    if axis == "y":
        return min(intrinsic.preferred_height, available_cross)
    return min(intrinsic.preferred_width, available_cross)


def _shrink_to_fit(node: Flex, sizes: list[float], min_sizes: list[float], overflow: float) -> float:
    remaining = overflow
    while remaining > 1e-6:
        shrinkable = [
            index
            for index, child in enumerate(node.children)
            if child.shrink > 0 and sizes[index] > min_sizes[index] + 1e-6
        ]
        if not shrinkable:
            break
        shrink_total = sum(max(node.children[index].shrink, 0.0) for index in shrinkable)
        if shrink_total <= 0:
            break

        reduced = 0.0
        for index in shrinkable:
            share = max(node.children[index].shrink, 0.0) / shrink_total
            capacity = sizes[index] - min_sizes[index]
            delta = min(remaining * share, capacity)
            sizes[index] -= delta
            reduced += delta

        if reduced <= 1e-6:
            break
        remaining = max(remaining - reduced, 0.0)
    return remaining


def _accumulate_offsets(start: float, sizes: list[float], gap: float) -> list[float]:
    offsets: list[float] = []
    cursor = start
    for size in sizes:
        offsets.append(cursor)
        cursor += size + gap
    return offsets


def _validate_region(node: LayoutNode, issues: list[LayoutIssue], node_path: str, slide: int | None) -> None:
    region = node._region
    if region is None:
        return
    if region.width < -1e-6 or region.height < -1e-6:
        issues.append(LayoutIssue(
            code="negative_size",
            message="Layout region has negative width or height.",
            severity="error",
            slide=slide,
            node_path=node_path,
            region=region,
        ))
    if region.left < -0.01:
        issues.append(LayoutIssue(
            code="overflow_x",
            message="Layout region extends beyond the slide's left edge.",
            severity="error",
            slide=slide,
            node_path=node_path,
            region=region,
        ))
    if region.top < -0.01:
        issues.append(LayoutIssue(
            code="overflow_y",
            message="Layout region extends beyond the slide's top edge.",
            severity="error",
            slide=slide,
            node_path=node_path,
            region=region,
        ))
    if region.right > SLIDE_WIDTH + 0.01:
        issues.append(LayoutIssue(
            code="overflow_x",
            message="Layout region extends beyond the slide's right edge.",
            severity="error",
            slide=slide,
            node_path=node_path,
            region=region,
        ))
    if region.bottom > SLIDE_HEIGHT + 0.01:
        issues.append(LayoutIssue(
            code="overflow_y",
            message="Layout region extends beyond the slide's bottom edge.",
            severity="error",
            slide=slide,
            node_path=node_path,
            region=region,
        ))


if __name__ == "__main__":
    from paperops.slides.core.theme import themes

    theme = themes.professional
    demo = HStack(children=[LayoutNode(width=2.0), LayoutNode(grow=1.0, size_mode_x="fill"), LayoutNode(width=1.5)])
    for issue in compute_layout(demo, CONTENT_REGION, theme):
        print(issue)
