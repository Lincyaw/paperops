from __future__ import annotations

import pytest

from paperops.slides.ir.node import Node
from paperops.slides.style.selector import parse_selector, specificity


def test_parse_and_specificity_supports_combinations():
    parsed = parse_selector(".card.kpi.primary")
    assert len(parsed.steps) == 1
    assert specificity(parsed) == 30


def test_parse_accepts_descendant_and_child_operators():
    descendant = parse_selector(".card .value")
    child = parse_selector(".card > .value")
    assert descendant.combinators == ("descendant",)
    assert child.combinators == ("child",)


def test_specificity_weights_id_class_tag():
    assert specificity(parse_selector("#hero")) == 100
    assert specificity(parse_selector(".a")) == 10
    assert specificity(parse_selector("title")) == 1


def test_selector_rejects_unsupported_token():
    with pytest.raises(ValueError):
        parse_selector("a[role=btn]")
