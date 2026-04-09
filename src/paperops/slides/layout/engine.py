"""Layout engine — resolve a layout tree into absolute slide regions."""

from __future__ import annotations

from paperops.slides.core.constants import CONTENT_REGION, Region, SLIDE_HEIGHT, SLIDE_WIDTH
from paperops.slides.layout.containers import Flex, Grid, GridItem, HStack, LayoutNode, Padding
from paperops.slides.layout.types import Constraints, IntrinsicSize, LayoutIssue, TrackSpec


def compute_layout(
    root: LayoutNode,
    region: Region,
    theme,
    *,
    slide: int | None = None,
    root_path: str = "root",
) -> list[dict]:
    """Resolve a component tree into absolute regions and return layout issues."""
    issues: list[LayoutIssue] = []
    _layout(root, region, theme, issues, root_path, slide)
    return [issue.to_dict() for issue in issues]


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


def _layout_flex(node: Flex, region: Region, theme, issues: list[LayoutIssue], node_path: str, slide: int | None) -> None:
    _validate_region(node, issues, node_path, slide)
    if not node.children:
        return

    if node.direction == "row" and (node.wrap or node.wrap_mode == "wrap"):
        _layout_wrapped_row(node, region, theme, issues, node_path, slide)
        return

    gap = node.main_gap()
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
    gap = node.main_gap()

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
