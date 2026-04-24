from __future__ import annotations

import pytest

from paperops.slides.ir.defines import (
    CircularMacroError,
    UnresolvedMacroVarError,
    expand_document,
)


def test_macro_expansion_replaces_variables_and_children():
    raw = {
        "theme": "minimal",
        "slides": [
            {"$use": "CoverCard", "title": "Phase 2", "children": ["deployed to deck"]},
        ],
        "defines": {
            "CoverCard": {
                "type": "card",
                "class": "cover",
                "children": [
                    {"type": "title", "text": "$title"},
                    {"type": "box", "children": "$CHILDREN"},
                ],
            }
        },
    }

    expanded = expand_document(raw, strict=True)
    slide = expanded["slides"][0]
    assert slide["type"] == "card"
    assert slide["class"] == "cover"
    assert len(slide["children"]) == 2
    assert slide["children"][0] == {"type": "title", "text": "Phase 2"}
    assert slide["children"][1] == {"type": "box", "children": ["deployed to deck"]}


def test_nested_macro_expansion_is_recursive():
    raw = {
        "theme": "minimal",
        "slides": [{"$use": "Wrapper", "title": "Q1", "children": []}],
        "defines": {
            "Wrapper": {
                "type": "stack",
                "children": [
                    {"$use": "Heading", "title": "$title"},
                ]
            },
            "Heading": {
                "type": "title",
                "text": "$title",
            },
        },
    }

    expanded = expand_document(raw, strict=True)
    slide = expanded["slides"][0]
    assert slide["children"] == [{"type": "title", "text": "Q1"}]


def test_circular_macro_reference_raises_structured_error():
    raw = {
        "theme": "minimal",
        "slides": [{"$use": "A"}],
        "defines": {
            "A": {"type": "stack", "children": [{"$use": "B"}]},
            "B": {"type": "stack", "children": [{"$use": "A"}]},
        },
    }
    with pytest.raises(CircularMacroError) as exc:
        expand_document(raw, strict=True)
    assert exc.value.code == "CIRCULAR_MACRO"


def test_unresolved_macro_variable_raises_only_in_strict_mode():
    raw = {
        "theme": "minimal",
        "slides": [{"$use": "Panel", "children": []}],
        "defines": {
            "Panel": {
                "type": "text",
                "text": "$missing",
            }
        },
    }

    relaxed = expand_document(raw, strict=False)
    assert relaxed["slides"][0]["text"] == "$missing"

    with pytest.raises(UnresolvedMacroVarError) as exc:
        expand_document(raw, strict=True)
    assert exc.value.code == "UNRESOLVED_MACRO_VAR"
