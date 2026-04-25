"""Semantic components for SlideCraft.

Each component declares its prop schema and expands into one or more atomic IR
nodes before style resolution.
"""

from __future__ import annotations

from typing import Any

from paperops.slides.components.registry import register_component
from paperops.slides.ir.node import Node


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _flatten_node_text(children: list[Any]) -> str:
    texts: list[str] = []
    for child in children:
        if isinstance(child, str):
            texts.append(child)
        elif isinstance(child, Node):
            if child.text is not None:
                texts.append(_as_text(child.text))
            if child.children:
                texts.append(_flatten_node_text(list(child.children)))
        elif isinstance(child, dict) and "text" in child:
            text = child.get("text")
            if text is not None:
                texts.append(_as_text(text))
    return "".join(texts).strip()


def _normalize_children(children: list[Any] | None) -> list[Any]:
    normalized: list[Any] = []
    for item in _coerce_list(children):
        if isinstance(item, Node):
            normalized.append(item.to_dict())
        elif isinstance(item, (dict, str)):
            normalized.append(item)
    return normalized


def _text_node(content: Any, *, classes: str = "text") -> dict[str, Any]:
    return {"type": "text", "class": classes, "text": _as_text(content)}


@register_component(
    "card",
    props_schema={},
    default_classes=["card"],
)
class Card:
    @staticmethod
    def expand(props, children=None, style=None):
        body = _normalize_children(children)
        lines = [_flatten_node_text([item]) for item in body]
        text = "\n".join(line for line in lines if line)
        return {
            "type": "box",
            "class": "card",
            "props": {"text": text},
        }


@register_component(
    "kpi",
    props_schema={
        "properties": {
            "label": {"type": "string", "required": True},
            "value": {"type": "string", "required": True},
            "delta": {"type": "string"},
            "trend": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
                "default": "neutral",
            },
        },
        "required": ["label", "value"],
    },
    default_classes=["kpi"],
)
class KPI:
    @staticmethod
    def expand(props, children=None, style=None):
        label = props["label"]
        value = props["value"]
        delta = props.get("delta")
        trend = props.get("trend", "neutral")
        payload = [
            _text_node(label, classes="label"),
            _text_node(value, classes="value"),
        ]
        if delta is not None:
            payload.append(
                _text_node(delta, classes=f"delta {trend}")
            )
        payload.extend(_normalize_children(children))

        return {
            "type": "card",
            "children": payload,
        }


@register_component(
    "callout",
    props_schema={
        "properties": {
            "kind": {"type": "string"},
            "text": {"type": "string"},
        }
    },
    default_classes=["callout"],
)
class Callout:
    @staticmethod
    def expand(props, children=None, style=None):
        kind = _as_text(props.get("kind", "note"))
        text = props.get("text")
        if text is None and children:
            text = _flatten_node_text(_normalize_children(children))
            children = []
        if text is None:
            text = ""
        payload = [
            _text_node(kind, classes="badge"),
            _text_node(text, classes="prose"),
        ]
        payload.extend(_normalize_children(children))
        return {"type": "card", "children": payload}


@register_component(
    "quote",
    props_schema={
        "properties": {
            "text": {"type": "string", "required": True},
            "author": {"type": "string"},
        },
        "required": ["text"],
    },
    default_classes=["quote"],
)
class Quote:
    @staticmethod
    def expand(props, children=None, style=None):
        body = props["text"]
        return {
            "type": "card",
            "children": [
                _text_node("“", classes="quote-mark"),
                _text_node(body, classes="body"),
                *(
                    [_text_node(props.get("author"), classes="author")]
                    if props.get("author")
                    else []
                ),
            ],
        }


@register_component(
    "pullquote",
    props_schema={
        "properties": {
            "text": {"type": "string", "required": True},
            "author": {"type": "string"},
        },
        "required": ["text"],
    },
    default_classes=["pullquote"],
)
class PullQuote:
    @staticmethod
    def expand(props, children=None, style=None):
        body = props["text"]
        payload = [
            _text_node(body, classes="body"),
        ]
        if props.get("author"):
            payload.append(_text_node(props.get("author"), classes="author"))
        payload.extend(_normalize_children(children))
        return {"type": "card", "children": payload}


@register_component(
    "keypoint",
    props_schema={
        "properties": {
            "number": {"type": "string"},
            "title": {"type": "string"},
            "body": {"type": "string"},
        }
    },
    default_classes=["keypoint"],
)
class KeyPoint:
    @staticmethod
    def expand(props, children=None, style=None):
        payload = []
        if props.get("number"):
            payload.append(_text_node(props["number"], classes="number"))
        if props.get("title"):
            payload.append(_text_node(props["title"], classes="title"))
        if props.get("body"):
            payload.append(_text_node(props["body"], classes="body"))
        extra = _normalize_children(children)
        if extra:
            payload.extend(extra)
        return {"type": "card", "children": payload}


@register_component(
    "stepper",
    props_schema={
        "properties": {
            "steps": {"type": "array"},
            "start": {"type": ["number", "string"], "default": 1},
        }
    },
    default_classes=["stepper"],
)
class Stepper:
    @staticmethod
    def expand(props, children=None, style=None):
        raw_steps = props.get("steps") or []
        start = int(props.get("start", 1))
        steps = _coerce_list(raw_steps)
        payload: list[Any] = []

        for index, step in enumerate(steps):
            if isinstance(step, dict):
                label = _as_text(step.get("label", step.get("title", "")))
                desc = _as_text(step.get("desc", step.get("description", "")))
            elif isinstance(step, str):
                label = step
                desc = ""
            else:
                label = str(step)
                desc = ""

            step_no = start + index
            step_nodes = [
                _text_node(step_no, classes="number"),
                _text_node(label, classes="label"),
            ]
            if desc:
                step_nodes.append(_text_node(desc, classes="desc"))
            payload.append({"type": "stack", "class": "step", "children": step_nodes})

        payload.extend(_normalize_children(children))
        return {"type": "stack", "class": "stepper", "children": payload}


@register_component(
    "timeline",
    props_schema={
        "properties": {
            "items": {"type": "array"},
        }
    },
    default_classes=["timeline"],
)
class Timeline:
    @staticmethod
    def expand(props, children=None, style=None):
        payload: list[Any] = []
        for item in _coerce_list(props.get("items")):
            date = ""
            title = ""
            desc = ""
            if isinstance(item, dict):
                date = _as_text(item.get("date", ""))
                title = _as_text(item.get("title", ""))
                desc = _as_text(item.get("desc", item.get("description", "")))
            elif isinstance(item, str):
                title = item
            else:
                title = str(item)

            node_children = []
            if date:
                node_children.append(_text_node(date, classes="date"))
            if title:
                node_children.append(_text_node(title, classes="title"))
            if desc:
                node_children.append(_text_node(desc, classes="desc"))
            if not node_children:
                node_children = [_text_node("", classes="empty")]
            payload.append({"type": "stack", "class": "timeline-item", "children": node_children})

        if not payload:
            payload = [{"type": "text", "class": "timeline-empty", "text": ""}]
        extra = _normalize_children(children)
        if extra:
            payload.extend(extra)
        return {"type": "stack", "class": "timeline", "children": payload}


@register_component(
    "figure",
    props_schema={
        "properties": {
            "src": {"type": "string"},
            "body": {"type": "string"},
            "chart_type": {"type": "string", "enum": ["line", "bar", "pie", "area"]},
            "caption": {"type": "string"},
            "source": {"type": "string"},
        }
    },
    default_classes=["figure"],
)
class Figure:
    @staticmethod
    def expand(props, children=None, style=None):
        payload: list[Any] = []

        if props.get("src"):
            payload.append(
                {
                    "type": "image",
                    "class": "media",
                    "props": {"src": props.get("src")},
                }
            )
        elif props.get("body"):
            payload.append(
                {
                    "type": "svg",
                    "class": "media",
                    "props": {"body": props.get("body")},
                }
            )
        elif props.get("chart_type"):
            payload.append(
                {
                    "type": "chart",
                    "class": "media",
                    "props": {
                        "chart_type": props.get("chart_type"),
                    },
                }
            )
        elif children:
            payload.extend(_normalize_children(children))

        if not payload:
            payload.append({"type": "text", "class": "caption", "text": "figure"})

        caption = props.get("caption")
        if caption is not None:
            payload.append(_text_node(caption, classes="caption"))
        source = props.get("source")
        if source:
            payload.append(_text_node(source, classes="source"))

        return {"type": "card", "children": payload}


@register_component(
    "caption",
    props_schema={
        "properties": {
            "text": {"type": "string", "required": True},
        },
        "required": ["text"],
    },
    default_classes=["caption"],
)
class Caption:
    @staticmethod
    def expand(props, children=None, style=None):
        if props.get("text"):
            return {
                "type": "text",
                "class": "caption",
                "text": props["text"],
            }

        text = _flatten_node_text(_normalize_children(children))
        return {
            "type": "text",
            "class": "caption",
            "text": text,
        }


@register_component(
    "spacer",
    props_schema={
        "properties": {
            "size": {"type": ["number", "string"]},
            "orientation": {
                "type": "string",
                "enum": ["h", "v", "horizontal", "vertical"],
                "default": "h",
            },
            "width": {"type": ["number", "string"]},
            "height": {"type": ["number", "string"]},
        }
    },
    default_classes=["spacer"],
)
class Spacer:
    @staticmethod
    def expand(props, children=None, style=None):
        style_payload = dict(props or {})
        size = style_payload.pop("size", None)
        orientation = style_payload.pop("orientation", "h")

        if style_payload.get("width") is None and style_payload.get("height") is None and size is not None:
            if orientation in {"v", "vertical"}:
                style_payload["height"] = size
            else:
                style_payload["width"] = size

        return {
            "type": "spacer",
            "class": "spacer",
            "style": style_payload,
            "children": [],
        }


@register_component(
    "note",
    props_schema={},
    default_classes=["note"],
)
class Note:
    @staticmethod
    def expand(props, children=None, style=None):
        return {
            "type": "note",
            "children": _normalize_children(children),
        }


__all__ = [
    "Callout",
    "Card",
    "Caption",
    "Figure",
    "KeyPoint",
    "KPI",
    "Note",
    "PullQuote",
    "Quote",
    "Spacer",
    "Stepper",
    "Timeline",
]
