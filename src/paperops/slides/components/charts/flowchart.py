"""Flowchart component — node-edge diagram rendered as native PPT shapes."""

from __future__ import annotations

from dataclasses import dataclass, field

from paperops.slides.core.constants import Direction
from paperops.slides.layout.containers import Column, LayoutNode, Row
from paperops.slides.layout.types import Constraints, IntrinsicSize
from paperops.slides.components.shapes import RoundedBox, Arrow


def _container_kwargs(node: LayoutNode) -> dict:
    return {
        "width": node.width,
        "height": node.height,
        "min_width": node.min_width,
        "min_height": node.min_height,
        "max_width": node.max_width,
        "max_height": node.max_height,
        "size_mode_x": node.size_mode_x,
        "size_mode_y": node.size_mode_y,
        "grow": node.grow,
        "shrink": node.shrink,
        "basis": node.basis,
        "wrap": node.wrap,
    }


def _is_rightward(direction: Direction | str) -> bool:
    value = str(direction)
    return value in {"right", "horizontal"}


@dataclass
class Flowchart(LayoutNode):
    """Node-edge diagram rendered with native PPT shapes.

    nodes: dict of {id: label} or {id: (label, color_name)}
    edges: list of (from_id, to_id) or (from_id, to_id, label)
    direction: "right" or "down"
    Example:
        Flowchart(
            nodes={"a": "Input", "b": ("Process", "primary"), "c": ("Output", "positive")},
            edges=[("a", "b"), ("b", "c")],
        )
    """

    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    direction: Direction | str = "right"
    node_widths: dict[str, float] = field(default_factory=dict)

    def measure(self, constraints: Constraints, theme) -> IntrinsicSize:
        return self.to_layout().measure(constraints, theme)

    def preferred_size(self, theme, available_width):
        layout = self.to_layout()
        return layout.preferred_size(theme, available_width)

    def to_layout(self) -> LayoutNode:
        """Expand into native shapes (RoundedBox + Arrow) layout tree."""
        if not self.nodes:
            return Row(children=[])

        node_list = list(self.nodes.keys())

        def _parse_node(val, nid: str):
            if isinstance(val, tuple):
                label = val[0]
                color = val[1]
                # Check for explicit width in node_widths dict
                if nid in self.node_widths:
                    return label, color, self.node_widths[nid]
                return label, color, None
            if nid in self.node_widths:
                return val, "bg_alt", self.node_widths[nid]
            return val, "bg_alt", None

        box_height = self.height if _is_rightward(self.direction) and self.height is not None else 0.9

        # Build RoundedBox for each node
        node_boxes: dict[str, RoundedBox] = {}
        for nid in node_list:
            label, color, explicit_width = _parse_node(self.nodes[nid], nid)
            text_clr = "white" if color not in ("bg_alt", "bg", "bg_accent") else "text"
            node_boxes[nid] = RoundedBox(
                text=label,
                color=color,
                border="border" if color in ("bg_alt", "bg", "bg_accent") else color,
                text_color=text_clr,
                bold=True,
                font_size="caption",
                height=box_height,
                width=explicit_width,
                size_mode_x="fit" if explicit_width is None else "fixed",
            )

        # Build edge-ordered sequence following the edges
        ordered_ids = self._edge_order(node_list)

        # Build children: box, arrow, box, arrow, ...
        arrow_dir = "horizontal" if _is_rightward(self.direction) else "vertical"
        children: list[LayoutNode] = []
        for i, nid in enumerate(ordered_ids):
            box = node_boxes[nid]
            children.append(box)
            if i < len(ordered_ids) - 1:
                next_nid = ordered_ids[i + 1]
                # Find edge label if any
                edge_label = None
                for edge in self.edges:
                    src, dst = edge[0], edge[1]
                    lbl = edge[2] if len(edge) == 3 else None
                    if src == nid and dst == next_nid:
                        edge_label = lbl
                        break

                arr = Arrow(
                    from_component=box,
                    to_component=node_boxes[next_nid],
                    label=edge_label,
                    color="primary",
                    width=0.22,
                    direction=arrow_dir,
                )
                children.append(arr)

        Container = Row if _is_rightward(self.direction) else Column
        return Container(
            gap=0.15,
            children=children,
            **_container_kwargs(self),
        )

    def _edge_order(self, node_list: list[str]) -> list[str]:
        """Determine node order from edges (topological sort for chain graphs)."""
        if not self.edges:
            return node_list

        # Build adjacency
        out_edges: dict[str, str] = {}
        in_edges: dict[str, str] = {}
        for edge in self.edges:
            src, dst = edge[0], edge[1]
            out_edges[src] = dst
            in_edges[dst] = src

        # Find the start node (no incoming edge)
        starts = [nid for nid in node_list if nid not in in_edges]
        if not starts:
            return node_list  # cycle or complex graph, fall back

        ordered = []
        current = starts[0]
        visited = set()
        while current and current not in visited:
            visited.add(current)
            ordered.append(current)
            current = out_edges.get(current)

        # Add any remaining nodes not in the chain
        for nid in node_list:
            if nid not in visited:
                ordered.append(nid)

        return ordered
