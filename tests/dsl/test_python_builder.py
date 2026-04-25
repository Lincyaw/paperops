from __future__ import annotations
from pathlib import Path

from paperops.slides.dsl import (
    Deck,
    Flex,
    Box,
    Callout,
    Card,
    Stepper,
    KPI,
    Layer,
    Padding,
    Stack,
    Text,
    Title,
    load_json_document,
)


def test_python_builder_and_json_loader_emit_same_dict():
    source_json = {
        "$schema": "paperops-slide-1.0",
        "theme": "minimal",
        "styles": {
            ".cover": {"bg": "bg_alt"},
        },
        "slides": [
            {
                "type": "slide",
                "children": [
                    {"type": "title", "text": "Parity Check"},
                    {
                        "type": "flex",
                        "children": [
                            {"type": "box", "class": "card", "text": "alpha"},
                            {"type": "text", "text": "beta"},
                        ],
                    },
                    {
                        "type": "stack",
                        "children": [{"type": "text", "text": "stacked"}],
                    },
                    {
                        "type": "padding",
                        "style": {"padding": "sm"},
                        "children": [{"type": "text", "text": "padded"}],
                    },
                    {
                        "type": "layer",
                        "children": [
                            {"type": "title", "text": "Layered"},
                        ],
                    },
                ],
            },
        ],
    }
    json_deck_dict = load_json_document(source_json).to_dict()

    parity_deck = Deck(
        theme="minimal",
        styles={".cover": {"bg": "bg_alt"}},
    )
    parity_deck.slide(
        Title("Parity Check"),
        Flex(
            Box(text="alpha", class_="card"),
            Text(text="beta"),
        ),
        Stack(Text(text="stacked")),
        Padding(Text(text="padded"), style={"padding": "sm"}),
        Layer(Title("Layered")),
    )

    assert parity_deck.to_dict() == json_deck_dict


def test_deck_render_shortcut_writes_pptx(tmp_path: Path):
    deck = Deck(theme="minimal")
    deck.slide(Title("From Builder"), Text("renders to pptx"))
    out = tmp_path / "builder_render.pptx"

    rendered = deck.render(out)
    assert rendered == out
    assert out.exists()


def test_python_builder_exports_semantic_components():
    deck = Deck(theme="minimal", sheet="seminar")
    deck.slide(
        Card(Text("plain card")),
        Callout(kind="Insight", text="Style lives in sheets."),
        Stepper(steps=[{"label": "Plan"}, {"label": "Render"}]),
    )

    slide_children = deck.to_dict()["slides"][0]["children"]
    assert [node["type"] for node in slide_children] == ["card", "callout", "stepper"]
    assert slide_children[1]["props"] == {
        "kind": "Insight",
        "text": "Style lives in sheets.",
    }
    assert slide_children[2]["props"]["steps"][0]["label"] == "Plan"


def test_kpi_builder_omits_empty_optional_props():
    deck = Deck(theme="minimal")
    deck.slide(KPI(label="Cases", value="9,152"))

    kpi = deck.to_dict()["slides"][0]["children"][0]
    assert kpi["props"] == {"label": "Cases", "value": "9,152"}
