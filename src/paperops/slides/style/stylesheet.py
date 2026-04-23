from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping

from paperops.slides.style.selector import ParsedSelector, parse_selector, specificity


@dataclass(frozen=True)
class StyleRule:
    selector_text: str
    selector: ParsedSelector
    declarations: Mapping[str, Any]
    source_index: int

    @property
    def specificity(self) -> int:
        return specificity(self.selector)


class StyleSheet:
    """Ordered selector -> declaration map."""

    def __init__(self, rules: Mapping[str, Mapping[str, Any]] | None = None):
        self._rules: list[StyleRule] = []
        if rules:
            for index, (selector_text, declarations) in enumerate(rules.items()):
                self._append_rule(selector_text, declarations, source_index=index)

    def _append_rule(
        self,
        selector_text: str,
        declarations: Mapping[str, Any],
        source_index: int,
    ) -> None:
        if not isinstance(declarations, Mapping):
            raise TypeError(
                f"Style declarations for {selector_text!r} must be a mapping"
            )
        parsed = parse_selector(selector_text)
        self._rules.append(
            StyleRule(
                selector_text=selector_text,
                selector=parsed,
                declarations=dict(declarations),
                source_index=source_index,
            )
        )

    def __len__(self) -> int:
        return len(self._rules)

    def __iter__(self) -> Iterator[StyleRule]:
        return iter(self._rules)

    def __getitem__(self, selector_text: str) -> dict[str, Any]:
        for rule in self._rules:
            if rule.selector_text == selector_text:
                return dict(rule.declarations)
        raise KeyError(selector_text)

    def items(self) -> Iterator[tuple[str, Mapping[str, Any]]]:
        for rule in self._rules:
            yield rule.selector_text, rule.declarations
