"""JSON Schema definitions for the slide DSL canonical IR.

This module provides both the schema dictionaries (draft 2020-12 shape) and small
lightweight runtime validators for project-local test assertions.
"""

from __future__ import annotations

from typing import Any

from paperops.slides.core.tokens import ShadowSpec

STYLE_KEY_SCHEMAS: dict[str, dict[str, Any]] = {
    "color": {
        "type": ["string", "number", "null"],
        "description": "Color token or hex value",
    },
    "bg": {
        "type": ["string", "number", "null"],
        "description": "Background color token",
    },
    "border": {
        "type": ["string", "number", "null"],
        "description": "Border color or style spec",
    },
    "border-left": {
        "type": ["string", "number", "array", "null"],
        "description": "Border-left shorthand",
    },
    "padding": {
        "type": ["string", "number", "null"],
        "description": "Shorthand spacing",
    },
    "padding-x": {
        "type": ["string", "number", "null"],
        "description": "Horizontal padding",
    },
    "padding-y": {
        "type": ["string", "number", "null"],
        "description": "Vertical padding",
    },
    "padding-top": {"type": ["string", "number", "null"]},
    "padding-right": {"type": ["string", "number", "null"]},
    "padding-bottom": {"type": ["string", "number", "null"]},
    "padding-left": {"type": ["string", "number", "null"]},
    "margin": {"type": ["string", "number", "null"]},
    "margin-x": {"type": ["string", "number", "null"]},
    "margin-y": {"type": ["string", "number", "null"]},
    "margin-top": {"type": ["string", "number", "null"]},
    "margin-right": {"type": ["string", "number", "null"]},
    "margin-bottom": {"type": ["string", "number", "null"]},
    "margin-left": {"type": ["string", "number", "null"]},
    "radius": {"type": ["string", "number", "null"]},
    "shadow": {"type": ["string", "array", "null", "object"]},
    "opacity": {"type": ["number", "null"]},
    "width": {"type": ["string", "number", "null"]},
    "height": {"type": ["string", "number", "null"]},
    "gap": {"type": ["string", "number", "null"]},
    "row-gap": {"type": ["string", "number", "null"]},
    "column-gap": {"type": ["string", "number", "null"]},
    "cols": {"type": ["string", "number", "null"]},
    "rows": {"type": ["string", "number", "null"]},
    "justify": {"type": ["string", "null"]},
    "align": {"type": ["string", "null"]},
    "align-items": {"type": ["string", "null"]},
    "align-self": {"type": ["string", "null"]},
    "overflow": {"type": ["string", "null"]},
    "max-lines": {"type": ["integer", "string", "null"]},
    "font": {"type": ["string", "number", "null"]},
    "font-family": {"type": ["string", "null"]},
    "font-weight": {"type": ["string", "number", "null"]},
    "font-style": {"type": ["string", "null"]},
    "line-height": {"type": ["number", "string", "null"]},
    "letter-spacing": {"type": ["number", "string", "null"]},
    "text-align": {"type": ["string", "null"]},
    "text-transform": {"type": ["string", "null"]},
    "animate": {"type": ["string", "null"]},
    "delay": {"type": ["string", "number", "null"]},
    "duration": {"type": ["string", "number", "null"]},
    "stagger": {"type": ["string", "number", "null"]},
    "animate-trigger": {"type": ["string", "null"]},
    "animate-group": {"type": ["string", "null"]},
    "grow": {"type": ["number", "null"]},
    "shrink": {"type": ["number", "null"]},
    "basis": {"type": ["string", "number", "null"]},
    "wrap": {"type": ["string", "boolean", "null"]},
    "aspect-ratio": {"type": ["string", "number", "null"]},
}

STYLE_VALUE_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "integer"},
        {"type": "boolean"},
        {"type": "array"},
        {"type": "object"},
        {"type": "null"},
    ]
}

IR_NODE_SCHEMA: dict[str, Any] = {
    "$id": "https://paperops.dev/schema/slide-dsl-node.json",
    "type": "object",
    "required": ["type"],
    "properties": {
        "type": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_-]*$"},
        "class": {"type": "string"},
        "id": {"type": "string"},
        "text": {"type": "string"},
        "style": {"$ref": "#/definitions/style"},
        "props": {"type": "object", "additionalProperties": True},
        "children": {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "string"},
                    {"$ref": "#/definitions/node"},
                ]
            },
        },
    },
    "additionalProperties": False,
}

IR_DOCUMENT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://paperops.dev/schema/slide-dsl.json",
    "title": "paperops.slide.dsl",
    "type": "object",
    "required": ["slides"],
    "properties": {
        "$schema": {"type": "string"},
        "meta": {"type": "object", "additionalProperties": True},
        "theme": {"type": "string"},
        "sheet": {"type": "string"},
        "styles": {
            "type": "object",
            "additionalProperties": {"$ref": "#/definitions/style"},
        },
        "defines": {
            "type": "object",
            "additionalProperties": {"$ref": "#/definitions/node"},
        },
        "slides": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/definitions/node"},
        },
    },
    "additionalProperties": False,
    "definitions": {
        "style": {
            "type": "object",
            "patternProperties": {
                "^.*$": STYLE_VALUE_SCHEMA,
            },
            "additionalProperties": False,
            "properties": {key: val for key, val in STYLE_KEY_SCHEMAS.items()},
        },
        "node": {"$ref": "#/"},  # will be normalized by runtime validators
    },
}

# Keep recursive schemas valid JSON-like for introspection.
IR_DOCUMENT_SCHEMA["definitions"]["node"] = {
    "type": "object",
    "required": ["type"],
    "properties": IR_NODE_SCHEMA["properties"],
    "additionalProperties": False,
    "$recursiveRef": "#",
}


def _validate_node_type(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("Node 'type' must be a non-empty string")


def _validate_style_value(value: Any, schema: dict[str, Any]) -> None:
    allowed_types = set(schema.get("type", []) or [])
    if not allowed_types:
        # fallback: no explicit type constraints
        return

    for t in allowed_types:
        if t == "string" and isinstance(value, str):
            return
        if t == "number" and isinstance(value, int | float):
            return
        if t == "integer" and isinstance(value, int):
            return
        if t == "boolean" and isinstance(value, bool):
            return
        if t == "array" and isinstance(value, list):
            return
        if t == "object" and isinstance(value, dict):
            return
        if t == "null" and value is None:
            return
    raise TypeError(f"Style value has unsupported type: {type(value)!r}")


def _validate_style(style: Any, path: str) -> None:
    if not isinstance(style, dict):
        raise TypeError(f"{path}: style must be an object")
    for key, value in style.items():
        if not isinstance(key, str):
            raise TypeError(f"{path}: style key must be a string")
        value_schema = STYLE_KEY_SCHEMAS.get(key, STYLE_VALUE_SCHEMA)
        if key not in STYLE_KEY_SCHEMAS and "/definitions/style" in str(value_schema):
            # no-op placeholder kept for introspection-only schemas
            continue
        _validate_style_value(
            value,
            value_schema if isinstance(value_schema, dict) else STYLE_VALUE_SCHEMA,
        )


def validate_node(raw: Any, *, path: str = "node") -> None:
    """Validate a single IR node dictionary.

    This is intentionally strict to ensure phase-0 API expectations are met:
    - `type` required
    - no unknown top-level keys
    - proper child/inline field types
    """
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: node must be a mapping")
    allowed = set(IR_NODE_SCHEMA["properties"].keys())
    unknown = set(raw.keys()) - allowed
    if unknown:
        raise ValueError(f"{path}: node contains unknown fields: {sorted(unknown)}")

    if "type" not in raw:
        raise ValueError(f"{path}: missing required field 'type'")
    _validate_node_type(raw["type"])

    if "class" in raw and not isinstance(raw["class"], str):
        raise TypeError(f"{path}: class must be string")
    if "id" in raw and not isinstance(raw["id"], str):
        raise TypeError(f"{path}: id must be string")
    if "style" in raw:
        _validate_style(raw["style"], f"{path}.style")
    if "text" in raw and raw["text"] is not None and not isinstance(raw["text"], str):
        raise TypeError(f"{path}: text must be a string when present")
    if "props" in raw and not isinstance(raw["props"], dict):
        raise TypeError(f"{path}: props must be an object")
    if "children" in raw:
        children = raw["children"]
        if not isinstance(children, list):
            raise TypeError(f"{path}: children must be a list")
        for idx, child in enumerate(children):
            if isinstance(child, str):
                continue
            if isinstance(child, dict):
                validate_node(child, path=f"{path}.children[{idx}]")
                continue
            raise TypeError(
                f"{path}.children[{idx}] must be either string or node object"
            )


def validate_document(raw: Any, *, path: str = "document") -> None:
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: document must be a mapping")
    required = set(IR_DOCUMENT_SCHEMA["required"])
    unknown = set(raw.keys()) - {
        "$schema",
        "meta",
        "theme",
        "sheet",
        "styles",
        "defines",
        "slides",
    }
    if unknown:
        raise ValueError(f"{path}: document contains unknown fields: {sorted(unknown)}")
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(
            f"{path}: document missing required field(s): {sorted(missing)}"
        )
    if not isinstance(raw.get("slides"), list):
        raise TypeError(f"{path}.slides must be a list")
    if not raw["slides"]:
        raise ValueError(f"{path}.slides must contain at least one node")
    for idx, slide in enumerate(raw["slides"]):
        if not isinstance(slide, dict):
            raise TypeError(f"{path}.slides[{idx}] must be a node object")
        validate_node(slide, path=f"{path}.slides[{idx}]")
