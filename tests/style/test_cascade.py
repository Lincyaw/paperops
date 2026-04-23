from __future__ import annotations

import time

from paperops.slides.core import themes
from paperops.slides.ir.node import Node
from paperops.slides.style import CascadeResult, StyleSheet, resolve_computed_styles


def _style(result: CascadeResult, node: Node):
    return result.style_for(node)


def test_sheet_selector_precedence_resolves_card_kpi_bg():
    sheet = StyleSheet(
        {
            ".card": {"bg": "bg_alt"},
            ".card.kpi": {"bg": "bg_accent"},
        }
    )
    node = Node(type="card", class_="card kpi")
    result = resolve_computed_styles(
        node,
        theme=themes.minimal,
        style_sheet=sheet,
    )
    assert _style(result, node).get("bg") == themes.minimal.colors["bg_accent"]


def test_descendant_selector_only_targets_descendants():
    deck = Node(
        type="deck",
        class_="deck",
        children=[
            Node(type="card", class_="card", children=[Node(type="value", class_="value")]),
            Node(type="value", class_="value"),
        ],
    )
    sheet = StyleSheet({".card .value": {"color": "primary"}})
    result = resolve_computed_styles(deck, theme=themes.minimal, style_sheet=sheet, strict=False)
    direct_child_value = deck.children[0].children[0]
    root_value = deck.children[1]
    assert direct_child_value is not None and isinstance(direct_child_value, Node)
    assert root_value is not None and isinstance(root_value, Node)
    assert _style(result, direct_child_value).get("color") == themes.minimal.colors["primary"]
    assert _style(result, root_value).get("color") is None


def test_child_selector_only_targets_direct_children():
    deck = Node(
        type="deck",
        children=[
            Node(
                type="card",
                class_="card",
                children=[Node(type="value", class_="value")],
            ),
            Node(type="value", class_="value"),
            Node(
                type="card",
                children=[
                    Node(type="value", class_="value", children=[]),
                ],
            ),
        ],
    )
    sheet = StyleSheet({".card > .value": {"bg": "bg_accent"}})
    result = resolve_computed_styles(deck, theme=themes.minimal, style_sheet=sheet, strict=False)
    direct = deck.children[0].children[0]
    grandchild = deck.children[2].children[0]
    loose = deck.children[1]
    assert _style(result, direct).get("bg") == themes.minimal.colors["bg_accent"]
    assert _style(result, grandchild).get("bg") is None
    assert _style(result, loose).get("bg") is None


def test_pseudo_first_matches_only_first_same_type_node():
    container = Node(
        type="stack",
        children=[
            Node(type="value", class_="value"),
            Node(type="text"),
            Node(type="value", class_="value"),
            Node(type="value", class_="value"),
        ],
    )
    sheet = StyleSheet({".value:first": {"color": "negative"}})
    result = resolve_computed_styles(container, theme=themes.academic, style_sheet=sheet, strict=False)
    first = container.children[0]
    second = container.children[2]
    third = container.children[3]
    assert isinstance(first, Node)
    assert isinstance(second, Node)
    assert isinstance(third, Node)
    assert _style(result, first).get("color") == themes.academic.colors["negative"]
    assert _style(result, second).get("color") is None
    assert _style(result, third).get("color") is None


def test_same_specificity_later_rule_wins():
    sheet = StyleSheet(
        {
            ".kpi": {"bg": "bg_alt"},
            ".highlight": {"bg": "bg_accent"},
        }
    )
    node = Node(type="value", class_="kpi highlight")
    result = resolve_computed_styles(node, theme=themes.minimal, style_sheet=sheet)
    assert _style(result, node).get("bg") == themes.minimal.colors["bg_accent"]


def test_inline_style_overrides_sheet_even_high_specificity():
    sheet = StyleSheet({".value": {"bg": "bg_alt", "font": "title"}})
    node = Node(type="value", class_="value", style={"bg": "bg_accent", "font": "body"})
    result = resolve_computed_styles(node, theme=themes.minimal, style_sheet=sheet)
    assert _style(result, node).get("bg") == themes.minimal.colors["bg_accent"]
    assert _style(result, node).get("font") == themes.minimal.fonts["body"]


def test_inherited_style_key_flows_down_until_overridden():
    tree = Node(
        type="deck",
        children=[
            Node(type="card", style={"color": "primary"}, children=[
                Node(type="value", children=[])
            ])
        ],
    )
    result = resolve_computed_styles(tree, theme=themes.academic, strict=False)
    card = tree.children[0]
    value = card.children[0]
    assert isinstance(card, Node)
    assert isinstance(value, Node)
    assert _style(result, card).get("color") == themes.academic.colors["primary"]
    assert _style(result, value).get("color") == themes.academic.colors["primary"]

    override = Node(
        type="deck",
        children=[
            Node(type="card", style={"color": "positive"}, children=[
                Node(type="value", class_="value")
            ]),
        ],
    )
    resolved_override = resolve_computed_styles(override, theme=themes.academic, strict=False)
    overridden_value = override.children[0].children[0]
    assert _style(resolved_override, overridden_value).get("color") == themes.academic.colors["positive"]


def test_non_inheritable_style_key_does_not_inherit():
    tree = Node(type="deck", style={"bg": "bg_alt"}, children=[
        Node(type="card")
    ])
    result = resolve_computed_styles(tree, theme=themes.minimal, strict=False)
    child = tree.children[0]
    assert isinstance(child, Node)
    assert _style(result, child).get("bg") is None


def test_blacklist_style_key_reports_structured_error():
    node = Node(type="card", style={"display": "flex"})
    result = resolve_computed_styles(node, theme=themes.minimal, strict=False)
    assert result.errors
    error = result.errors[0]
    assert error.code == "UNKNOWN_STYLE_KEY"
    assert error.key == "display"
    assert error.suggestion and "<Flex>" in error.suggestion


def test_pseudo_nth_matches_position():
    container = Node(
        type="stack",
        children=[
            Node(type="item", class_="value"),
            Node(type="item", class_="value"),
            Node(type="item", class_="value"),
        ],
    )
    sheet = StyleSheet({".value:nth(2)": {"color": "positive"}})
    result = resolve_computed_styles(container, theme=themes.academic, style_sheet=sheet, strict=False)
    middle = container.children[1]
    assert isinstance(middle, Node)
    assert _style(result, middle).get("color") == themes.academic.colors["positive"]


def test_computed_style_lookup_follows_fallback_chain():
    tree = Node(type="deck", style={"color": "primary"}, children=[
        Node(type="card", children=[Node(type="value", style={})])
    ])
    result = resolve_computed_styles(tree, theme=themes.academic, strict=False)
    value = tree.children[0].children[0]
    assert isinstance(value, Node)
    assert _style(result, value).get("color") == themes.academic.colors["primary"]


def test_style_resolution_is_fast_for_thousands_of_nodes():
    sheet = StyleSheet({f".c{i}": {"font": "body", "padding": "md"} for i in range(20)})
    slides = []
    for _ in range(20):
        children = [
            Node(type="item", class_=f"c{i % 20}") for i in range(150)
        ]
        slides.append(Node(type="slide", children=children))
    deck = Node(type="deck", children=slides)

    start = time.perf_counter()
    result = resolve_computed_styles(
        deck,
        theme=themes.minimal,
        style_sheet=sheet,
        strict=False,
    )
    elapsed = time.perf_counter() - start

    assert isinstance(result, CascadeResult)
    assert len(result.computed) == 1 + 20 + 3000
    assert elapsed < 0.2


def test_theme_default_sheet_deck_local_and_inline_precedence():
    node = Node(type="value", class_="value", style={"font": "title"})
    result = resolve_computed_styles(
        node,
        theme=themes.minimal,
        theme_defaults={"value": {"font": "body"}},
        style_sheet=StyleSheet({".value": {"font": "caption"}},
        ),
        deck_style=StyleSheet({".value": {"font": "primary"}}),
        strict=False,
    )
    assert _style(result, node).get("font") == themes.minimal.fonts["title"]


def test_inherit_value_for_inheritable_key_keeps_chain():
    tree = Node(type="deck", style={"color": "primary"}, children=[Node(type="value", style={"color": "inherit"})])
    result = resolve_computed_styles(tree, theme=themes.academic, strict=False)
    child = tree.children[0]
    assert isinstance(child, Node)
    assert _style(result, child).get("color") == themes.academic.colors["primary"]


def test_auto_and_none_values_are_preserved():
    node = Node(type="value", style={"color": "auto", "bg": "none"})
    result = resolve_computed_styles(node, theme=themes.minimal, strict=False)
    assert _style(result, node).get("color") == "auto"
    assert _style(result, node).get("bg") == "none"
