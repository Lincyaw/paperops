from __future__ import annotations

from paperops.slides.core import (
    UnknownTokenCategoryError,
    UnknownTokenError,
    resolve,
    themes,
)


def test_resolve_theme_spacing_token_must_be_float():
    executive = themes.executive

    assert executive.resolve_token("spacing", "md") == 0.32


def test_resolve_theme_color_token_returns_hex():
    professional = themes.professional

    assert isinstance(professional.resolve_token("colors", "primary"), str)
    assert professional.resolve_token("colors", "primary") == "#3B6B9D"


def test_resolve_with_unit_string_is_preserved_as_inches():
    executive = themes.executive

    assert executive.resolve_token("spacing", "0.5in") == 0.5
    assert isinstance(executive.resolve_token("fonts", "12pt"), float)


def test_unknown_token_raises_with_available_list():
    minimal = themes.minimal

    try:
        minimal.resolve_token("spacing", "absurd")
        raise AssertionError("Expected UnknownTokenError")
    except UnknownTokenError as err:
        assert err.token == "absurd"
        assert err.category == "spacing"
        assert "Available" in err.args[0]
        assert "md" in err.available


def test_resolve_color_supports_hex_and_auto_inherit_values():
    academic = themes.academic

    assert academic.resolve_token("colors", "#123ABC") == "#123ABC"
    assert academic.resolve_token("spacing", "auto") == "auto"
    assert academic.resolve_token("spacing", "inherit") == "inherit"


def test_resolve_shadow_resolves_named_shadow_spec():
    executive = themes.executive

    shadow = executive.resolve_token("shadow", "sm")
    assert shadow.dx > 0
    assert shadow.color == "#C5CBD3"


def test_resolve_raises_for_unknown_category():
    base = themes.academic

    try:
        resolve(base, "unsupported", "md")
        raise AssertionError("Expected UnknownTokenCategoryError")
    except UnknownTokenCategoryError:
        pass
