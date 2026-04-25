from __future__ import annotations

import re

import pytest

from paperops.slides.components.registry import ComponentError, expand_node, expand_nodes
from paperops.slides.ir.node import Node
from paperops.slides.style import get_sheet


def _text_nodes(node: Node) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if node.text:
        out.append((node.type, node.text))
    for child in node.children or []:
        if isinstance(child, Node):
            out.extend(_text_nodes(child))
    return out


def test_kpi_expands_to_card_with_label_value_delta_lines():
    node = Node(type="kpi", props={"label": "DAU", "value": "125k", "delta": "+56%"})
    expanded = expand_nodes([node])[0]

    assert expanded.type == "box"
    assert expanded.props is not None
    assert expanded.props["text"].splitlines() == ["DAU", "125k", "+56%"]


def test_missing_required_kpi_prop_raises_mapped_error():
    node = Node(type="kpi", props={"label": "DAU"})
    with pytest.raises(ComponentError, match="MISSING_REQUIRED_PROP"):
        expand_node(node)


def test_kpi_node_merges_default_and_user_classes_without_duplicates():
    node = Node(
        type="kpi",
        class_="value kpi",
        props={"label": "DAU", "value": "125k"},
    )
    expanded = expand_node(node)
    classes = [token for token in (expanded.class_ or "").split() if token]
    assert classes == ["box", "card", "kpi", "value"], classes


def test_unknown_kpi_prop_raises_mapped_error():
    node = Node(
        type="kpi",
        props={"label": "DAU", "value": "125k", "delta": "+56%", "unknown": "x"},
    )
    with pytest.raises(ComponentError, match="UNKNOWN_PROP"):
        expand_node(node)


def test_default_class_merges_before_user_class_and_dedupes():
    expanded = expand_nodes([Node(type="text", class_="body text", props={"text": "Hello"})])[0]
    assert expanded.class_ == "text body"
    assert re.fullmatch(r"text\s+body", expanded.class_)


def test_builtin_sheet_registry_exposes_default_variants():
    sheet = get_sheet("minimal")
    assert sheet is not None
    assert len(list(sheet)) > 0


def test_all_semantic_components_expand():
    cards = {
        "card": {"type": "card", "children": [{"type": "text", "text": "body"}]},
        "kpi": {"type": "kpi", "props": {"label": "DAU", "value": "125k", "delta": "+56%"}},
        "callout": {"type": "callout", "props": {"kind": "Tip", "text": "Focus on signal"}},
        "quote": {"type": "quote", "props": {"text": "Stay close to data", "author": "Ops Team"}},
        "pullquote": {"type": "pullquote", "props": {"text": "This deck is consistent.", "author": "Lead"}},
        "keypoint": {"type": "keypoint", "props": {"number": "01", "title": "Start", "body": "Collect traces"}},
        "stepper": {"type": "stepper", "props": {"steps": [{"label": "a"}, {"label": "b"}]}},
        "timeline": {"type": "timeline", "props": {"items": [{"date": "2026-01", "title": "Pilot"}]}},
        "figure": {"type": "figure", "props": {"chart_type": "bar", "caption": "Trend"}},
        "caption": {"type": "caption", "props": {"text": "A tiny caption"}},
        "spacer": {"type": "spacer", "props": {"size": "sm"}},
        "note": {"type": "note", "children": ["speaker note line"]},
    }

    for name, raw in cards.items():
        expanded = expand_node(Node.from_dict(raw))
        if name in {"note", "spacer"}:
            assert expanded.type == name
        else:
            assert expanded.type != name, f"{name} should expand into a different node type"

        class_tokens = set((expanded.class_ or "").split())
        expected = {
            "card": "card",
            "kpi": "kpi",
            "callout": "callout",
            "quote": "quote",
            "pullquote": "pullquote",
            "keypoint": "keypoint",
            "stepper": "stepper",
            "timeline": "timeline",
            "figure": "figure",
            "caption": "caption",
            "spacer": "spacer",
            "note": "note",
        }.get(name)
        if expected:
            assert expected in class_tokens

    # card-like nodes should resolve directly into a styled box leaf.
    card_tree = expand_node(Node.from_dict({"type": "card", "children": [{"type": "text", "text": "x"}]}))
    assert card_tree.type == "box"
    assert card_tree.props == {"text": "x"}
