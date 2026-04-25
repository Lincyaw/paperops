from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from paperops.slides.dsl.inline_html import InlineHtmlError
from paperops.slides.dsl.json_loader import load_json_document
from paperops.slides.dsl.markdown_parser import (
    load_markdown_document,
    parse_markdown_fragment,
    parse_markdown_text,
)
from paperops.slides.dsl.markdown_parser import (
    to_canonical_ir as markdown_to_canonical_ir,
)
from paperops.slides.dsl.mdx_parser import (
    MDXParseError,
    load_mdx_document,
    parse_mdx_text,
)
from paperops.slides.dsl.mdx_parser import to_canonical_ir as mdx_to_canonical_ir

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "issue5"


def _load_python_fixture() -> dict:
    path = FIXTURE_DIR / "gallery.py"
    spec: ModuleSpec = importlib.util.spec_from_file_location("issue5_gallery_py", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DECK.to_dict()


def test_gallery_four_frontends_emit_same_canonical_ir():
    json_doc = load_json_document(FIXTURE_DIR / "gallery.json").to_dict()

    markdown_doc = load_markdown_document(FIXTURE_DIR / "gallery.md").to_dict()
    assert markdown_doc == json_doc

    mdx_doc = load_mdx_document(FIXTURE_DIR / "gallery.mdx").to_dict()
    assert mdx_doc == json_doc

    python_doc = _load_python_fixture()
    assert python_doc == json_doc


def test_markdown_frontmatter_and_fenced_blocks_with_attributes_and_slide_separator():
    source = """---\ntheme: minimal\nsheet: seminar\nmeta:\n  title: Gallery\n  author: PaperOps\nstyles:\n  .cover:\n    bg: bg_alt\ndefines:\n  KPI:\n    type: card\n    class: kpi\n---\n# Title {.cover #hero}\n\n> Subtitle\n{.subtitle}\n\n::: grid {cols=\"1fr 1fr\" gap=\"md\"}\n::: card.kpi\n**DAU** 125k\n:::\n::: card.kpi\n**Retention** 42%\n:::\n::::\n---\n## Outro\n"""

    document = parse_markdown_text(source)[0]
    assert document.theme == "minimal"
    assert document.sheet == "seminar"
    assert document.meta == {"title": "Gallery", "author": "PaperOps"}
    assert document.styles == {".cover": {"bg": "bg_alt"}}
    assert document.defines == {"KPI": {"type": "card", "class": "kpi"}}
    assert len(document.slides) == 2
    assert document.slides[0].children[0].type == "title"
    assert document.slides[0].children[0].class_ == "cover"
    assert document.slides[0].children[0].id == "hero"
    assert document.slides[0].children[1].type == "prose"
    assert document.slides[0].children[1].class_ == "subtitle"
    assert document.slides[0].children[2].type == "grid"
    assert document.slides[0].children[2].style == {"cols": "1fr 1fr", "gap": "md"}
    assert len(document.slides[0].children[2].children or []) == 2
    assert len(document.slides[1].children) == 1
    assert document.slides[1].children[0].type == "heading"


def test_markdown_paragraph_attribute_and_quote_line():
    doc = parse_markdown_text("> Subtitle here.\n{.subtitle}")[0]
    assert len(doc.slides) == 1
    title = doc.slides[0].children[0]
    assert title.type == "prose"
    assert title.class_ == "subtitle"


def test_markdown_fenced_div_parsing_preserves_nested_blocks():
    text = """::: grid {cols=\"1fr 1fr\" gap=\"md\"}
::: card.kpi
A
:::
::: card.kpi
B
:::
::::
"""
    nodes = parse_markdown_fragment(text)
    assert len(nodes) == 1
    assert nodes[0].type == "grid"
    assert nodes[0].style == {"cols": "1fr 1fr", "gap": "md"}
    assert len(nodes[0].children or []) == 2


def test_markdown_inline_escape_less_than_with_space_or_digit_is_literal():
    nodes = parse_markdown_fragment("Use < 3 and 4 < 5")
    assert nodes[0].text == "Use < 3 and 4 < 5"


def test_markdown_span_style_color_parsed_and_invalid_display_blocked():
    good = parse_markdown_fragment('<span style="color:#ff0000">red</span>')[0]
    assert good.children and good.children[0].type == "span"
    assert good.children[0].props == {"style": {"color": "#ff0000"}}

    with pytest.raises(InlineHtmlError) as exc_info:
        parse_markdown_fragment('<span style="display:flex">bad</span>')
    assert exc_info.value.code == "UNSUPPORTED_INLINE_HTML"
    assert exc_info.value.suggestion is not None


def test_markdown_unsupported_inline_html_reports_diagnostic():
    with pytest.raises(InlineHtmlError) as exc_info:
        parse_markdown_fragment("hello <div>bad</div> world")
    assert exc_info.value.code == "UNSUPPORTED_INLINE_HTML"
    assert "fenced div" in " ".join(exc_info.value.suggestion or []).lower()


def test_mdx_component_properties_style_and_unknown_component_error():
    source = """<KPI label=\"DAU\" value=\"125k\" />\n<Grid cols=\"1fr 1fr\" gap=\"lg\">\n</Grid>"""
    document = parse_mdx_text(source)[0]
    kpi, grid = document.slides[0].children
    assert kpi.type == "kpi"
    assert kpi.props == {"label": "DAU", "value": "125k"}
    assert grid.type == "grid"
    assert grid.style == {"cols": "1fr 1fr", "gap": "lg"}
    assert not grid.props

    with pytest.raises(MDXParseError) as exc_info:
        parse_mdx_text("<UnknownTag />")
    assert exc_info.value.code == "UNKNOWN_TYPE"
    assert exc_info.value.suggestion is not None


def test_mdx_style_object_and_markdown_children_are_parsed():
    source = """<Card>\n\n- one\n- two\n</Card>\n\n<Grid style={{ cols: \"1fr 1fr\", gap: \"md\" }}>\n  <Heading>Cell A</Heading>\n  <Heading>Cell B</Heading>\n</Grid>"""
    document = parse_mdx_text(source)[0]
    card = document.slides[0].children[0]
    grid = document.slides[0].children[1]
    assert card.type == "card"
    assert card.children and card.children[0].type == "list"
    assert grid.type == "grid"
    assert grid.style == {"cols": "1fr 1fr", "gap": "md"}
    assert len(grid.children or []) == 2
    assert (grid.children or [])[0].type == "heading"


def test_to_canonical_ir_handles_files_by_path_string_for_markdown_and_mdx():
    markdown_path = str(FIXTURE_DIR / "gallery.md")
    mdx_path = str(FIXTURE_DIR / "gallery.mdx")

    assert (
        markdown_to_canonical_ir(markdown_path).to_dict()
        == parse_markdown_text(Path(markdown_path).read_text(encoding="utf-8"))[
            0
        ].to_dict()
    )
    assert (
        mdx_to_canonical_ir(mdx_path).to_dict()
        == parse_mdx_text(Path(mdx_path).read_text(encoding="utf-8"))[0].to_dict()
    )
