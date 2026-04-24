"""JSON loader for the IR DSL."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from paperops.slides.ir import Node, validate_document
from paperops.slides.ir.defines import expand_document


DEFAULT_SCHEMA = "paperops-slide-1.0"


def load_json_document(source: str | Path | Mapping[str, Any], *, strict: bool = False) -> "Document":
    """Load a JSON IR document and return a validated deck representation."""
    raw = _load_source(source)
    if not isinstance(raw, dict):
        raise TypeError("Document payload must be an object")

    # Parse macros first; this keeps callers free to validate expanded trees.
    expanded = expand_document(raw, strict=strict)
    validate_document(expanded)

    return Document.from_dict(expanded)


def _load_source(source: str | Path | Mapping[str, Any]) -> Any:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    try:
        return json.loads(str(source))
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON source: {source!r}") from None


@dataclass(frozen=True)
class Document:
    """IR document container used as pipeline input/output contracts."""

    schema: str = DEFAULT_SCHEMA
    meta: dict[str, Any] | None = None
    theme: str = "minimal"
    sheet: str | None = None
    styles: dict[str, dict[str, Any]] | None = None
    defines: dict[str, dict[str, Any]] | None = None
    slides: list[Node] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Document":
        if not isinstance(raw, Mapping):
            raise TypeError("Document payload must be a mapping")
        raw_meta = raw.get("meta")
        raw_styles = raw.get("styles")
        raw_defines = raw.get("defines")

        if raw_styles is not None and not isinstance(raw_styles, dict):
            raise TypeError("Document.styles must be a mapping")
        if raw_defines is not None and not isinstance(raw_defines, dict):
            raise TypeError("Document.defines must be a mapping")

        if not isinstance(raw.get("slides", []), list):
            raise TypeError("Document.slides must be a list")
        slides = [Node.from_dict(item) for item in raw["slides"]]

        return cls(
            schema=str(raw.get("$schema", DEFAULT_SCHEMA)),
            meta=dict(raw_meta) if isinstance(raw_meta, dict) else None,
            theme=str(raw.get("theme", "minimal")) if raw.get("theme") is not None else "minimal",
            sheet=str(raw.get("sheet")) if raw.get("sheet") is not None else None,
            styles={key: dict(value) for key, value in raw_styles.items()}
            if isinstance(raw_styles, dict)
            else None,
            defines={key: dict(value) for key, value in raw_defines.items()}
            if isinstance(raw_defines, dict)
            else None,
            slides=slides,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"$schema": self.schema}
        if self.meta is not None:
            payload["meta"] = dict(self.meta)
        payload["theme"] = self.theme
        if self.sheet is not None:
            payload["sheet"] = self.sheet
        if self.styles is not None:
            payload["styles"] = self.styles
        if self.defines is not None:
            payload["defines"] = self.defines
        payload["slides"] = [slide.to_dict() for slide in self.slides]
        return payload
