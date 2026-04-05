"""Flowchart component — node-edge diagram rendered as native PPT shapes."""

from __future__ import annotations

from dataclasses import dataclass, field

from paperops.slides.core.constants import Direction
from paperops.slides.layout.containers import LayoutNode, HStack, VStack
from paperops.slides.layout.auto_size import measure_text
from paperops.slides.components.shapes import RoundedBox, Arrow
from paperops.slides.components.text import TextBlock


def _estimate_full_text_width(text: str, font_size_pt: float, font_family: str = "Calibri") -> float:
    """Estimate the full width needed for text, accounting for margins.

    Unlike estimate_min_text_width which only measures the longest token,
    this measures the entire text string.
    """
    if not text:
        return 0.5
    content_w, _ = measure_text(text, font_family, font_size_pt)
    # Add horizontal margin padding (0.6 inches total for comfortable fit)
    # Increased from 0.3 to prevent text wrapping in PowerPoint
    return content_w + 0.6


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

    def preferred_size(self, theme, available_width):
        layout = self.to_layout()
        return layout.preferred_size(theme, available_width)

    def to_layout(self) -> LayoutNode:
        """Expand into native shapes (RoundedBox + Arrow) layout tree."""
        if not self.nodes:
            return HStack(children=[])

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

        # Determine box dimensions based on direction and node count
        n = len(node_list)
        if self.direction == "right":
            box_height = self.height or 1.2
        else:
            box_height = 0.9

        # Pre-calculate text widths to ensure boxes fit content
        caption_pt = 10.0
        font_family = "Liberation Sans"

        # Build RoundedBox for each node
        node_boxes: dict[str, RoundedBox] = {}
        for nid in node_list:
            label, color, explicit_width = _parse_node(self.nodes[nid], nid)
            text_clr = "white" if color not in ("bg_alt", "bg", "bg_accent") else "text"
            box_width = explicit_width if explicit_width is not None else _estimate_full_text_width(label, caption_pt, font_family)
            node_boxes[nid] = RoundedBox(
                text=label,
                color=color,
                border="border" if color in ("bg_alt", "bg", "bg_accent") else color,
                text_color=text_clr,
                bold=True,
                font_size="caption",
                height=box_height,
                width=box_width,
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
                    width=0.22,
                    direction=arrow_dir,
                )
                children.append(arr)

        Container = HStack if self.direction == "right" else VStack
        return Container(
            gap=0.15,
            children=children,
            width=self.width,
            height=self.height,
            size_mode_x=self.size_mode_x,
            size_mode_y=self.size_mode_y,
            grow=self.grow,
            shrink=self.shrink,
            basis=self.basis,
            wrap=self.wrap,
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
