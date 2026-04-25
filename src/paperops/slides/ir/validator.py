"""Structured validation helpers for LLM-friendly slide DSL errors."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from paperops.slides.ir.defines import MacroExpansionError, expand_document
from paperops.slides.ir.schema import STYLE_KEY_SCHEMAS, validate_document


@dataclass(frozen=True)
class ValidationMessage:
    code: str
    message: str
    path: str
    hint: str | None = None
    suggestion: list[str] | None = None
    severity: str = "error"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }
        if self.hint is not None:
            payload["hint"] = self.hint
        if self.suggestion is not None:
            payload["suggestion"] = list(self.suggestion)
        payload.update(self.meta)
        return payload


@dataclass
class ValidationReport:
    document: dict[str, Any] | None = None
    errors: list[ValidationMessage] = field(default_factory=list)
    warnings: list[ValidationMessage] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "error" if self.errors else "ok"

    def has_errors(self) -> bool:
        return bool(self.errors)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "errors": [item.to_dict() for item in self.errors],
            "warnings": [item.to_dict() for item in self.warnings],
        }
        if self.traces:
            payload["trace"] = list(self.traces)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    def to_pretty_text(self) -> str:
        lines: list[str] = []
        for group_name, items in (("ERROR", self.errors), ("WARN", self.warnings)):
            for item in items:
                line = f"[{group_name}] {item.code} @ {item.path}: {item.message}"
                if item.hint:
                    line += f" | hint: {item.hint}"
                if item.suggestion:
                    line += f" | suggestion: {', '.join(item.suggestion)}"
                lines.append(line)
        return "\n".join(lines)


class StructuredValidationError(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(report.to_pretty_text() or report.to_json())

    def to_dict(self) -> dict[str, Any]:
        return self.report.to_dict()


_STYLE_HINTS: dict[str, tuple[str, list[str]]] = {
    "display": (
        "Layout is selected by component type, not CSS display.",
        ["<Flex>", "<Grid>", "<Stack>", "<Absolute>"],
    ),
    "position": (
        "Absolute positioning is modeled by the <Absolute> component.",
        ["<Absolute>"],
    ),
    "transition": (
        "PPT transitions are modeled with animation style keys.",
        ["animate", "duration"],
    ),
    "transform": (
        "Use semantic components or preset animations instead of CSS transform.",
        ["animate", "<Rotate>", "<Scale>"],
    ),
}

_KNOWN_STRUCTURAL_TYPES = {
    "slide",
    "flex",
    "hstack",
    "vstack",
    "stack",
    "grid",
    "layer",
    "absolute",
    "padding",
}


def _load_source(source: str | Path | Mapping[str, Any]) -> Any:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(str(source))


def _message(
    code: str,
    message: str,
    path: str,
    *,
    severity: str,
    hint: str | None = None,
    suggestion: list[str] | None = None,
    **meta: Any,
) -> ValidationMessage:
    return ValidationMessage(
        code=code,
        message=message,
        path=path,
        hint=hint,
        suggestion=suggestion,
        severity=severity,
        meta=meta,
    )


def _record(report: ValidationReport, item: ValidationMessage) -> None:
    if item.severity == "error":
        report.errors.append(item)
    else:
        report.warnings.append(item)


def _walk_style(
    style: Mapping[str, Any], path: str, report: ValidationReport, *, severity: str
) -> None:
    for key in style:
        if key in STYLE_KEY_SCHEMAS:
            continue
        hint, suggestion = _STYLE_HINTS.get(key, (None, None))
        _record(
            report,
            _message(
                "UNKNOWN_STYLE_KEY",
                f"Unknown style key: {key!r}",
                f"{path}.{key}",
                severity=severity,
                hint=hint,
                suggestion=suggestion,
            ),
        )


def _scan_node(
    node: Any,
    path: str,
    report: ValidationReport,
    *,
    severity: str,
    seen_ids: dict[str, str],
) -> None:
    from paperops.slides import components  # noqa: F401
    from paperops.slides.components.registry import registry

    if not isinstance(node, dict):
        return

    node_type = node.get("type")
    if (
        isinstance(node_type, str)
        and node_type not in _KNOWN_STRUCTURAL_TYPES
        and not registry.has(node_type)
    ):
        _record(
            report,
            _message(
                "UNKNOWN_TYPE",
                f"Unknown component type: {node_type!r}",
                f"{path}.type",
                severity=severity,
                hint="Use a registered atomic or semantic component.",
            ),
        )

    node_id = node.get("id")
    if isinstance(node_id, str):
        if node_id in seen_ids:
            _record(
                report,
                _message(
                    "DUPLICATE_ID",
                    f"Duplicate id {node_id!r}; first declared at {seen_ids[node_id]}",
                    f"{path}.id",
                    severity="error",
                ),
            )
        else:
            seen_ids[node_id] = path

    style = node.get("style")
    if isinstance(style, Mapping):
        _walk_style(style, f"{path}.style", report, severity=severity)

    text = node.get("text")
    if isinstance(text, str) and text.startswith("$"):
        _record(
            report,
            _message(
                "UNRESOLVED_MACRO_VAR",
                f"Macro variable {text!r} not provided",
                path,
                severity=severity,
            ),
        )

    children = node.get("children") or []
    if isinstance(children, list):
        for index, child in enumerate(children):
            child_path = f"{path}.children[{index}]"
            if isinstance(child, str) and child.startswith("$"):
                _record(
                    report,
                    _message(
                        "UNRESOLVED_MACRO_VAR",
                        f"Macro variable {child!r} not provided",
                        child_path,
                        severity=severity,
                    ),
                )
                continue
            _scan_node(child, child_path, report, severity=severity, seen_ids=seen_ids)


def validate_ir_document(
    source: str | Path | Mapping[str, Any],
    *,
    strict: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    severity = "error" if strict else "warning"

    try:
        raw = _load_source(source)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        _record(
            report,
            _message("INVALID_SCHEMA", str(exc), "document", severity="error"),
        )
        return report

    if not isinstance(raw, dict):
        _record(
            report,
            _message(
                "INVALID_SCHEMA",
                "Document payload must be an object",
                "document",
                severity="error",
            ),
        )
        return report

    try:
        expanded = expand_document(raw, strict=False)
    except MacroExpansionError as exc:
        _record(report, _message(exc.code, str(exc), exc.path, severity="error"))
        return report
    except Exception as exc:
        _record(
            report, _message("INVALID_SCHEMA", str(exc), "document", severity="error")
        )
        return report

    try:
        validate_document(expanded)
    except Exception as exc:
        _record(
            report, _message("INVALID_SCHEMA", str(exc), "document", severity="error")
        )
        return report

    if isinstance(expanded.get("styles"), Mapping):
        for selector, declarations in expanded["styles"].items():
            if isinstance(declarations, Mapping):
                _walk_style(
                    declarations, f"styles.{selector}", report, severity=severity
                )

    seen_ids: dict[str, str] = {}
    for index, slide in enumerate(expanded.get("slides", [])):
        _scan_node(
            slide, f"slides[{index}]", report, severity=severity, seen_ids=seen_ids
        )

    report.document = expanded
    return report


__all__ = [
    "StructuredValidationError",
    "ValidationMessage",
    "ValidationReport",
    "validate_ir_document",
]
