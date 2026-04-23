from __future__ import annotations

from paperops.slides.ir import Node


def test_node_round_trip_with_children_and_strings():
    node = Node(
        type="slide",
        class_="cover hero",
        id="hero-slide",
        style={"bg": "bg_alt", "padding": "lg"},
        props={"align": "center"},
        children=[
            "intro",
            Node(type="text", props={"value": "hello"}),
        ],
    )

    roundtrip = Node.from_dict(node.to_dict())

    assert roundtrip == node
    assert roundtrip.children is not None
    assert roundtrip.children[0] == "intro"
    assert isinstance(roundtrip.children[1], Node)


def test_node_rejects_invalid_type_and_children_in_schema_validation():
    base = {
        "class": "test",
        "children": ["ok"],
    }
    try:
        Node.from_dict(base)
        raise AssertionError("Expected ValueError")
    except ValueError as err:
        assert "missing required field 'type'" in str(err)


def test_node_invalid_child_type_is_rejected_by_from_dict():
    invalid = {
        "type": "slide",
        "children": [123],
    }
    try:
        Node.from_dict(invalid)
        raise AssertionError("Expected TypeError")
    except TypeError as err:
        assert "must be either string or node object" in str(err)
