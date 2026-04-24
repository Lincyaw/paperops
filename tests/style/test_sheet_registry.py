from __future__ import annotations

from paperops.slides.style import list_sheets


def test_builtin_sheets_are_registered():
    sheets = list_sheets()
    expected = {"default", "minimal", "academic", "seminar", "keynote", "whitepaper", "pitch"}
    assert expected.issubset(set(sheets))
    assert len(sheets) >= len(expected)
