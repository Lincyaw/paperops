"""Flowchart component — node-edge diagram rendered as native PPT shapes."""

from __future__ import annotations

from dataclasses import dataclass, field

from paperops.slides.core.constants import Direction
from paperops.slides.layout.containers import LayoutNode, HStack, VStack
from paperops.slides.components.shapes import RoundedBox, Arrow
from paperops.slides.components.text import TextBlock


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

    def preferred_size(self, theme, available_width):
        layout = self.to_layout()
        return layout.preferred_size(theme, available_width)

    def to_layout(self) -> LayoutNode:
        """Expand into native shapes (RoundedBox + Arrow) layout tree."""
        if not self.nodes:
            return HStack(children=[])

        node_list = list(self.nodes.keys())

        def _parse_node(val):
            if isinstance(val, tuple):
                return val[0], val[1]
            return val, "bg_alt"

        # Determine box dimensions based on direction and node count
        n = len(node_list)
        if self.direction == "right":
            box_height = self.height or 1.2
        else:
            box_height = 0.9

        # Build RoundedBox for each node
        node_boxes: dict[str, RoundedBox] = {}
        for nid in node_list:
            label, color = _parse_node(self.nodes[nid])
            text_clr = "white" if color not in ("bg_alt", "bg", "bg_accent") else "text"
            node_boxes[nid] = RoundedBox(
                text=label,
                color=color,
                border="border" if color in ("bg_alt", "bg", "bg_accent") else color,
                text_color=text_clr,
                bold=True,
                font_size="caption",
                height=box_height,
            )

        # Build edge-ordered sequence following the edges
        ordered_ids = self._edge_order(node_list)

        # Build children: box, arrow, box, arrow, ...
        arrow_dir = "vertical" if self.direction == "down" else "horizontal"
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
                    direction=arrow_dir,
                )
                children.append(arr)

        Container = HStack if self.direction == "right" else VStack
        return Container(
            gap=0.15,
            children=children,
            width=self.width,
            height=self.height,
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
