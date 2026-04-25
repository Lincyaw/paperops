from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


_INHERITABLE_KEYS = {
    "color",
    "font",
    "font-family",
    "font-weight",
    "font-style",
    "line-height",
    "text-align",
    "lang",
}


@dataclass(frozen=True)
class ComputedStyle:
    """Resolved style values for a single IR node.

    Values are resolved and merged according to style cascade and inheritance rules.
    `get()` follows a fallback chain to parent nodes for inheritable keys.
    """

    values: Mapping[str, Any] = field(default_factory=dict)
    parent: "ComputedStyle | None" = None

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.values:
            value = self.values[key]
            if value == "inherit":
                if self.parent is None:
                    return default
                return self.parent.get(key, default)
            return value
        if self.parent is None or key not in _INHERITABLE_KEYS:
            return default
        return self.parent.get(key, default)

    def __getattr__(self, key: str) -> Any:
        if key in self.values:
            return self.values[key]
        raise AttributeError(key)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)

    def snapshot(self) -> dict[str, Any]:
        merged = dict(self.values)
        if self.parent is None:
            return {key: value for key, value in merged.items() if value is not None}

        for key in _INHERITABLE_KEYS:
            if key in merged and merged[key] == "inherit":
                inherited = self.parent.get(key)
                if inherited is None:
                    merged.pop(key, None)
                else:
                    merged[key] = inherited
            elif key not in merged:
                inherited = self.parent.get(key)
                if inherited is not None:
                    merged[key] = inherited
        return {key: value for key, value in merged.items() if value is not None}
