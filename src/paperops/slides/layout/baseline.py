"""Baseline grid snapping and sibling alignment helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from paperops.slides.layout.containers import (
    Absolute,
    AbsoluteItem,
    Grid,
    GridItem,
    LayoutNode,
    Padding,
)


def apply_baseline_alignment(
    slide_node, layout_root: LayoutNode, *, baseline: float
) -> None:
    style = getattr(slide_node, "computed_style", None)
    enabled = False
    if style is not None:
        enabled = bool(style.get("baseline-snap", False))
    elif getattr(slide_node, "style", None):
        enabled = bool(slide_node.style.get("baseline-snap", False))
    if not enabled:
        return

    _apply_align_to(layout_root)
    _snap_tree(layout_root, baseline)


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


def _ir_style(node: LayoutNode) -> dict[str, Any]:
    source = getattr(node, "_ir_node", None)
    if source is None:
        return {}
    computed = getattr(source, "computed_style", None)
    if computed is not None:
        return computed.snapshot()
    return dict(source.style or {})


def _matches_token(node: LayoutNode, token: str) -> bool:
    source = getattr(node, "_ir_node", None)
    if source is None:
        return False
    if getattr(source, "id", None) == token:
        return True
    if getattr(source, "type", None) == token:
        return True
    classes = (getattr(source, "class_", None) or "").split()
    return token in classes


def _edge_value(node: LayoutNode, edge: str) -> float | None:
    region = getattr(node, "_region", None)
    if region is None:
        return None
    if edge == "top":
        return float(region.top)
    if edge == "bottom":
        return float(region.top + region.height)
    if edge == "center":
        return float(region.top + region.height / 2.0)
    return float(region.top)


def _apply_align_to(node: LayoutNode) -> None:
    children = _layout_children(node)
    for child in children:
        style = _ir_style(child)
        align_to = style.get("align-to")
        if isinstance(align_to, str) and align_to.startswith("sibling."):
            target_spec = align_to[len("sibling.") :]
            target_name, _, edge = target_spec.partition(":")
            edge = edge or "top"
            for sibling in children:
                if sibling is child or not _matches_token(sibling, target_name):
                    continue
                region = getattr(child, "_region", None)
                target_value = _edge_value(sibling, edge)
                if region is None or target_value is None:
                    continue
                setattr(child, "_region", replace(region, top=target_value))
                break
        _apply_align_to(child)


def _snap_value(value: float, baseline: float) -> float:
    return round(value / baseline) * baseline


def _snap_tree(node: LayoutNode, baseline: float) -> None:
    region = getattr(node, "_region", None)
    if region is not None and region.top is not None:
        setattr(
            node,
            "_region",
            replace(region, top=_snap_value(float(region.top), baseline)),
        )
    for child in _layout_children(node):
        _snap_tree(child, baseline)


__all__ = ["apply_baseline_alignment"]
