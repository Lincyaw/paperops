from __future__ import annotations

import pytest

from paperops.slides.ir import IR_DOCUMENT_SCHEMA, validate_document, validate_node


def test_validate_node_accepts_valid_node():
    node = {
        "type": "card",
        "class": "kpi",
        "id": "node-1",
        "style": {"padding": "md", "font": "body", "radius": 0.12},
        "props": {"value": 10},
        "children": ["text", {"type": "text", "children": ["Hello"]}],
    }

    validate_node(node)


def test_validate_node_rejects_missing_type():
    with pytest.raises(ValueError, match="missing required"):
        validate_node({"class": "kpi"})


def test_validate_node_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown fields"):
        validate_node({"type": "kpi", "class": "a", "unknown": 1})


def test_validate_node_rejects_non_string_class():
    with pytest.raises(TypeError, match="class must be string"):
        validate_node({"type": "kpi", "class": 1})


def test_validate_document_round_trip_shape_is_accepted():
    document = {
        "$schema": "paperops-slide-1.0",
        "meta": {"title": "phase 0"},
        "theme": "minimal",
        "slides": [
            {
                "type": "slide",
                "children": [{"type": "text", "children": ["hello"]}],
            }
        ],
    }

    validate_document(document)


def test_validate_document_rejects_unknown_top_key():
    with pytest.raises(ValueError, match="unknown fields"):
        validate_document({"slides": [{"type": "slide"}], "oops": 1})
