"""Layout containers and primitives for SlideCraft."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from paperops.slides.layout.types import Constraints, IntrinsicSize, TrackSpec, auto, fr, fixed


@dataclass
class LayoutNode:
    """Base node for all layout participants."""

    _region: Any = field(default=None, init=False, repr=False)
    _node_path: str | None = field(default=None, init=False, repr=False)

    width: float | None = field(default=None, kw_only=True)
    height: float | None = field(default=None, kw_only=True)
    min_width: float | None = field(default=None, kw_only=True)
    min_height: float | None = field(default=None, kw_only=True)
    max_width: float | None = field(default=None, kw_only=True)
    max_height: float | None = field(default=None, kw_only=True)
    size_mode_x: str = field(default="auto", kw_only=True)
    size_mode_y: str = field(default="auto", kw_only=True)
    grow: float = field(default=0.0, kw_only=True)
    shrink: float = field(default=1.0, kw_only=True)
    basis: float | None = field(default=None, kw_only=True)
    wrap: bool = field(default=False, kw_only=True)
    node_id: str | None = field(default=None, kw_only=True)

    def display_name(self) -> str:
        return self.node_id or self.__class__.__name__.lower()

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        """Return a side-effect-free intrinsic size request."""
        if type(self).preferred_size is not LayoutNode.preferred_size:
            available_width = constraints.max_width
            if available_width is None:
                available_width = self.width if self.width is not None else 0.0
            preferred_width, preferred_height = type(self).preferred_size(self, theme, available_width)
            min_width = self.min_width if self.min_width is not None else preferred_width
            min_height = self.min_height if self.min_height is not None else preferred_height
            return IntrinsicSize(
                min_width=min_width,
                preferred_width=preferred_width,
                min_height=min_height,
                preferred_height=preferred_height,
            ).clamp(constraints)
        width = self.width if self.width is not None else 1.0
        height = self.height if self.height is not None else 1.0
        min_width = self.min_width if self.min_width is not None else width
        min_height = self.min_height if self.min_height is not None else height
        intrinsic = IntrinsicSize(
            min_width=min_width,
            preferred_width=width,
            min_height=min_height,
            preferred_height=height,
        )
        return intrinsic.clamp(constraints)

    def preferred_size(self, theme, available_width: float) -> tuple[float, float]:
        intrinsic = self.measure(Constraints(max_width=max(available_width, 0.0)), theme)
        return intrinsic.preferred_width, intrinsic.preferred_height


@dataclass
class Flex(LayoutNode):
    """Frontend-style linear layout primitive."""

    direction: str = "row"
    justify: str = "start"
    align: str = "stretch"
    gap: float = 0.3
    row_gap: float | None = None
    column_gap: float | None = None
    wrap_mode: str = "nowrap"
    children: list[LayoutNode] = field(default_factory=list)

    def main_gap(self) -> float:
        if self.direction == "column":
            return self.row_gap if self.row_gap is not None else self.gap
        return self.column_gap if self.column_gap is not None else self.gap

    def cross_gap(self) -> float:
        if self.direction == "column":
            return self.column_gap if self.column_gap is not None else self.gap
        return self.row_gap if self.row_gap is not None else self.gap

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        if not self.children:
            width = self.width or constraints.max_width or 0.0
            height = self.height or 0.0
            return IntrinsicSize(width, width, height, height).clamp(constraints)

        available_width = self.width if self.width is not None else constraints.max_width
        available_height = self.height if self.height is not None else constraints.max_height
        child_sizes: list[IntrinsicSize] = []
        for child in self.children:
            if self.direction == "row":
                child_constraints = Constraints(
                    max_width=child.width,
                    max_height=child.height if child.height is not None else available_height,
                )
            else:
                child_constraints = Constraints(
                    max_width=child.width if child.width is not None else available_width,
                    max_height=child.height,
                )
            child_sizes.append(child.measure(child_constraints, theme))
        gap = self.main_gap()

        if self.direction == "row":
            preferred_width = sum(size.preferred_width for size in child_sizes) + gap * max(len(child_sizes) - 1, 0)
            min_width = sum(size.min_width for size in child_sizes) + gap * max(len(child_sizes) - 1, 0)
            preferred_height = max((size.preferred_height for size in child_sizes), default=0.0)
            min_height = max((size.min_height for size in child_sizes), default=0.0)
            if self.wrap or self.wrap_mode == "wrap":
                available_width = constraints.max_width or preferred_width
                preferred_width = min(preferred_width, available_width)
                min_width = min(min_width, available_width)
                preferred_height = _estimate_wrapped_row_height(self.children, available_width, theme, gap)
            elif available_width is not None:
                available_main = max(available_width - gap * max(len(child_sizes) - 1, 0), 0.0)
                assigned_widths = _resolve_measure_main_sizes(self, child_sizes, available_main)
                cross_sizes = [
                    child.measure(
                        Constraints(
                            max_width=child.width if child.width is not None else max(child_width, 0.0),
                            max_height=child.height if child.height is not None else available_height,
                        ),
                        theme,
                    )
                    for child, child_width in zip(self.children, assigned_widths)
                ]
                preferred_height = max((size.preferred_height for size in cross_sizes), default=preferred_height)
                min_height = max((size.min_height for size in cross_sizes), default=min_height)
        else:
            preferred_width = max((size.preferred_width for size in child_sizes), default=0.0)
            min_width = max((size.min_width for size in child_sizes), default=0.0)
            preferred_height = sum(size.preferred_height for size in child_sizes) + gap * max(len(child_sizes) - 1, 0)
            min_height = sum(size.min_height for size in child_sizes) + gap * max(len(child_sizes) - 1, 0)

        if self.width is not None:
            preferred_width = min_width = self.width
        else:
            if self.min_width is not None:
                min_width = max(min_width, self.min_width)
            if self.max_width is not None:
                preferred_width = min(preferred_width, self.max_width)

        if self.height is not None:
            preferred_height = min_height = self.height
        else:
            if self.min_height is not None:
                min_height = max(min_height, self.min_height)
            if self.max_height is not None:
                preferred_height = min(preferred_height, self.max_height)

        intrinsic = IntrinsicSize(
            min_width=min_width,
            preferred_width=max(preferred_width, min_width),
            min_height=min_height,
            preferred_height=max(preferred_height, min_height),
        )
        return intrinsic.clamp(constraints)

    def __iter__(self):
        return iter(self.children)


@dataclass
class HStack(Flex):
    """Compatibility wrapper for row-based flex."""

    direction: str = field(default="row", init=False)


@dataclass
class VStack(Flex):
    """Compatibility wrapper for column-based flex."""

    direction: str = field(default="column", init=False)


Row = HStack
Column = VStack


@dataclass
class GridItem(LayoutNode):
    """Explicitly placed grid item."""

    child: LayoutNode | None = None
    row: int = 0
    col: int = 0
    row_span: int = 1
    col_span: int = 1

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        if self.child is None:
            return IntrinsicSize(0.0, 0.0, 0.0, 0.0)
        return self.child.measure(constraints, theme)


@dataclass
class Grid(LayoutNode):
    """Track-based grid layout with legacy equal-column fallback."""

    columns: list[TrackSpec] | None = None
    rows: list[TrackSpec] | None = None
    cols: int | None = None
    gap: float = 0.3
    column_gap: float | None = None
    row_gap: float | None = None
    center_last_row: bool = False
    children: list[Any] = field(default_factory=list)

    def _column_gap(self) -> float:
        return self.column_gap if self.column_gap is not None else self.gap

    def _row_gap(self) -> float:
        return self.row_gap if self.row_gap is not None else self.gap

    def iter_items(self) -> list[GridItem]:
        items: list[GridItem] = []
        if any(isinstance(child, GridItem) for child in self.children):
            for child in self.children:
                if isinstance(child, GridItem):
                    items.append(child)
                else:
                    raise TypeError("Grid children must all be GridItem when explicit placement is used")
            return items

        col_count = self.cols or len(self.columns or []) or 2
        for index, child in enumerate(self.children):
            items.append(GridItem(child=child, row=index // col_count, col=index % col_count))
        return items

    def resolved_columns(self) -> list[TrackSpec]:
        if self.columns:
            return list(self.columns)
        count = self.cols or 2
        return [fr(1.0) for _ in range(count)]

    def resolved_rows(self) -> list[TrackSpec]:
        if self.rows:
            return list(self.rows)
        items = self.iter_items()
        if not items:
            return [auto()]
        max_row = max(item.row + item.row_span for item in items)
        return [auto() for _ in range(max_row)]

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        items = self.iter_items()
        if not items:
            width = self.width or constraints.max_width or 0.0
            height = self.height or 0.0
            return IntrinsicSize(width, width, height, height).clamp(constraints)

        columns = self.resolved_columns()
        rows = self.resolved_rows()
        column_gap = self._column_gap()
        row_gap = self._row_gap()

        preferred_width = self.width if self.width is not None else constraints.max_width or _estimate_grid_width(items, columns, theme)
        if self.min_width is not None:
            preferred_width = max(preferred_width, self.min_width)

        available_width = max(preferred_width - column_gap * max(len(columns) - 1, 0), 0.0)
        widths = _estimate_track_sizes(columns, available_width, items, axis="x", theme=theme)
        preferred_height = self.height if self.height is not None else _estimate_grid_height(
            items,
            rows,
            widths,
            row_gap,
            column_gap,
            theme,
        )
        min_width = self.min_width if self.min_width is not None else min(preferred_width, sum(max(width, 0.0) for width in widths) + column_gap * max(len(widths) - 1, 0))
        min_height = self.min_height if self.min_height is not None else preferred_height

        intrinsic = IntrinsicSize(
            min_width=min_width,
            preferred_width=preferred_width,
            min_height=min_height,
            preferred_height=preferred_height,
        )
        return intrinsic.clamp(constraints)

    def __iter__(self):
        return iter(self.children)


@dataclass
class Padding(LayoutNode):
    """Single-child padding wrapper."""

    child: Any = None
    all: float | None = None
    left: float | None = None
    right: float | None = None
    top: float | None = None
    bottom: float | None = None

    @property
    def _left(self) -> float:
        return self.left if self.left is not None else (self.all or 0.0)

    @property
    def _right(self) -> float:
        return self.right if self.right is not None else (self.all or 0.0)

    @property
    def _top(self) -> float:
        return self.top if self.top is not None else (self.all or 0.0)

    @property
    def _bottom(self) -> float:
        return self.bottom if self.bottom is not None else (self.all or 0.0)

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        pad_h = self._left + self._right
        pad_v = self._top + self._bottom
        inner_max_width = None if constraints.max_width is None else max(constraints.max_width - pad_h, 0.0)
        inner_max_height = None if constraints.max_height is None else max(constraints.max_height - pad_v, 0.0)
        inner_constraints = Constraints(max_width=inner_max_width, max_height=inner_max_height)

        if self.child is None:
            intrinsic = IntrinsicSize(0.0, 0.0, 0.0, 0.0)
        else:
            intrinsic = self.child.measure(inner_constraints, theme)

        result = IntrinsicSize(
            min_width=intrinsic.min_width + pad_h,
            preferred_width=intrinsic.preferred_width + pad_h,
            min_height=intrinsic.min_height + pad_v,
            preferred_height=intrinsic.preferred_height + pad_v,
        )
        return result.clamp(constraints)


def _estimate_wrapped_row_height(children: Iterable[LayoutNode], width: float, theme, gap: float) -> float:
    line_width = 0.0
    line_height = 0.0
    total_height = 0.0
    for child in children:
        intrinsic = child.measure(Constraints(max_width=max(width, 0.0)), theme)
        child_width = intrinsic.preferred_width
        child_height = intrinsic.preferred_height
        next_width = child_width if line_width == 0 else line_width + gap + child_width
        if line_width > 0 and next_width > width:
            total_height += line_height + gap
            line_width = child_width
            line_height = child_height
        else:
            line_width = next_width
            line_height = max(line_height, child_height)
    return total_height + line_height


def _resolve_measure_main_sizes(node: Flex, child_sizes: list[IntrinsicSize], available_main: float) -> list[float]:
    sizes = [0.0 for _ in node.children]
    min_sizes = [0.0 for _ in node.children]
    fill_indices: list[int] = []
    used = 0.0

    for index, (child, intrinsic) in enumerate(zip(node.children, child_sizes)):
        preferred = child.basis if child.basis is not None else (
            child.width if node.direction == "row" and child.width is not None
            else child.height if node.direction == "column" and child.height is not None
            else intrinsic.preferred_width if node.direction == "row"
            else intrinsic.preferred_height
        )
        minimum = (
            child.width if node.direction == "row" and child.width is not None
            else child.height if node.direction == "column" and child.height is not None
            else child.min_width if node.direction == "row" and child.min_width is not None
            else child.min_height if node.direction == "column" and child.min_height is not None
            else intrinsic.min_width if node.direction == "row"
            else intrinsic.min_height
        )
        min_sizes[index] = minimum
        if _size_mode_for_measure(child, "x" if node.direction == "row" else "y") == "fill":
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
        while overflow > 1e-6:
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
                delta = min(overflow * share, capacity)
                sizes[index] -= delta
                reduced += delta

            if reduced <= 1e-6:
                break
            overflow = max(overflow - reduced, 0.0)

    return [max(size, 0.0) for size in sizes]


def _size_mode_for_measure(node: LayoutNode, axis: str) -> str:
    explicit = node.width if axis == "x" else node.height
    if explicit is not None:
        return "fixed"
    mode = node.size_mode_x if axis == "x" else node.size_mode_y
    if mode == "stretch":
        return "fill"
    return "fit" if mode == "auto" else mode


def _estimate_grid_width(items: list[GridItem], columns: list[TrackSpec], theme) -> float:
    intrinsic_width = 0.0
    for item in items:
        child = item.child
        if child is None:
            continue
        intrinsic = child.measure(Constraints(), theme)
        span = max(item.col_span, 1)
        intrinsic_width = max(intrinsic_width, intrinsic.preferred_width / span)
    return intrinsic_width * max(len(columns), 1)


def _estimate_track_sizes(
    tracks: list[TrackSpec],
    available: float,
    items: list[GridItem],
    axis: str,
    theme,
) -> list[float]:
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
        demand = intrinsic.preferred_width if axis == "x" else intrinsic.preferred_height
        start = item.col if axis == "x" else item.row
        span = item.col_span if axis == "x" else item.row_span
        if span <= 0:
            continue
        unit = demand / span
        for idx in range(start, min(start + span, len(tracks))):
            if tracks[idx].kind == "auto":
                sizes[idx] = max(sizes[idx], unit)

    used = sum(sizes)
    remaining = max(available - used, 0.0)
    if fr_indices and fr_total > 0:
        for idx in fr_indices:
            sizes[idx] = remaining * (tracks[idx].value / fr_total)

    return sizes


def _estimate_grid_height(
    items: list[GridItem],
    rows: list[TrackSpec],
    widths: list[float],
    row_gap: float,
    column_gap: float,
    theme,
) -> float:
    row_sizes = [0.0 for _ in rows]
    full_width = sum(widths)
    for item in items:
        child = item.child
        if child is None:
            continue
        span_width = sum(widths[item.col:item.col + item.col_span]) + column_gap * max(item.col_span - 1, 0)
        intrinsic = child.measure(Constraints(max_width=span_width or full_width), theme)
        per_row = intrinsic.preferred_height / max(item.row_span, 1)
        for idx in range(item.row, min(item.row + item.row_span, len(rows))):
            if rows[idx].kind == "fixed":
                row_sizes[idx] = max(row_sizes[idx], rows[idx].value)
            else:
                row_sizes[idx] = max(row_sizes[idx], per_row)
    return sum(row_sizes) + row_gap * max(len(rows) - 1, 0)
