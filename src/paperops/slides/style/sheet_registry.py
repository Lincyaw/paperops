"""Sheet registry for named style presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from paperops.slides.style.stylesheet import StyleSheet


_REGISTRY: dict[str, StyleSheet] = {}


@dataclass(frozen=True)
class UnknownSheetError(ValueError):
    """Raised when a sheet name is not registered."""


def register_sheet(name: str, sheet: StyleSheet | Mapping[str, Mapping[str, Any]]) -> None:
    """Register a named sheet.

    Parameters:
        name: public sheet name
        sheet: stylesheet mapping or StyleSheet instance
    """
    if not isinstance(name, str) or not name.strip():
        raise TypeError("sheet name must be a non-empty string")
    sheet_name = name.strip()

    compiled = sheet if isinstance(sheet, StyleSheet) else StyleSheet(sheet)
    if sheet_name in _REGISTRY:
        raise ValueError(f"Sheet {sheet_name!r} is already registered")

    _REGISTRY[sheet_name] = compiled


def get_sheet(name: str) -> StyleSheet:
    """Resolve registered sheet by name."""
    if not isinstance(name, str) or not name.strip():
        raise TypeError("sheet name must be a non-empty string")
    if name not in _REGISTRY:
        raise UnknownSheetError(f"Unknown style sheet {name!r}")
    return _REGISTRY[name]


def list_sheets() -> dict[str, StyleSheet]:
    """Return a snapshot of the sheet registry."""
    return dict(_REGISTRY)


def clear_sheets() -> None:
    """Clear all registered sheets (test-only helper)."""
    _REGISTRY.clear()


__all__ = [
    "UnknownSheetError",
    "clear_sheets",
    "get_sheet",
    "list_sheets",
    "register_sheet",
]
