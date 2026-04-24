"""IR-first Python DSL for minimal deck authoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from paperops.slides.dsl.json_loader import DEFAULT_SCHEMA, Document, load_json_document
from paperops.slides.ir import Node


def _coerce_style(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"style must be a mapping, got {type(value)!r}")
    return dict(value)


def _coerce_props(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"props must be a mapping, got {type(value)!r}")
    return dict(value)


def _normalize_children_argument(children: Any) -> list["Node | str"]:
    if children is None:
        return []
    if isinstance(children, (Node, _BuilderNode, str)):
        return _coerce_children((children,))
    if isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
        return _coerce_children(list(children))
    return _coerce_children((children,))


def _coerce_children(raw_children: Iterable[Any] | None) -> list["Node | str"]:
    normalized: list["Node | str"] = []
    if raw_children is None:
        return normalized

    for item in raw_children:
        if item is None:
            continue
        if isinstance(item, _BuilderNode):
            normalized.append(item.to_node())
        elif isinstance(item, Node):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append(item)
        else:
            raise TypeError(f"Unsupported child type: {type(item)!r}")

    return normalized


def _set_textive_props(
    node_type: str,
    text: str | None,
    props: dict[str, Any] | None,
    children: list[Node | str],
) -> dict[str, Any] | None:
    if node_type not in {"text", "title", "subtitle", "heading"}:
        return props

    merged = dict(props or {})
    if text is not None:
        merged["text"] = text
    elif (
        len(children) == 1
        and isinstance(children[0], str)
        and "text" not in merged
    ):
        merged["text"] = children[0]
        return merged
    elif not children and "text" not in merged:
        merged["text"] = ""
    return merged


class _BuilderNode:
    """Base class for lightweight IR node builders."""

    node_type = "node"

    def __init__(
        self,
        *children: Any,
        text: str | None = None,
        class_: str | None = None,
        id_: str | None = None,
        style: Mapping[str, Any] | None = None,
        props: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        self.node_type = self.node_type
        self.class_ = class_
        self.id_ = id_
        self.text = text
        self.style = _coerce_style(style)

        merged_props = dict(_coerce_props(props) or {})
        merged_props.update(extra)

        self._children = _coerce_children(children)
        merged_props = _set_textive_props(self.node_type, text, merged_props, self._children)

        if (
            self.node_type in {"text", "title", "subtitle", "heading"}
            and text is None
            and len(self._children) == 1
            and isinstance(self._children[0], str)
            and merged_props is not None
            and merged_props.get("text") == self._children[0]
        ):
            self._children = []
            if self.text is None and merged_props is not None and len(merged_props) == 1:
                self.text = str(merged_props["text"])
            if merged_props is not None and len(merged_props) == 1:
                merged_props = {}
        elif (
            self.node_type in {"text", "title", "subtitle", "heading"}
            and isinstance(merged_props, dict)
            and set(merged_props.keys()) == {"text"}
        ):
            if self.text is None:
                self.text = str(merged_props["text"])
            merged_props = {}
        if merged_props:
            self.props = merged_props
        else:
            self.props = None

    def __getitem__(self, value: Any) -> "_BuilderNode":
        self._children.extend(_coerce_children(_normalize_children_argument(value)))
        return self

    def with_children(self, *children: Any) -> "_BuilderNode":
        self._children.extend(_coerce_children(children))
        return self

    def with_prop(self, **props: Any) -> "_BuilderNode":
        merged = dict(self.props or {})
        merged.update(props)
        self.props = merged
        return self

    def with_style(self, **style: Any) -> "_BuilderNode":
        merged = dict(self.style or {})
        merged.update(style)
        self.style = merged
        return self

    def to_node(self) -> Node:
        props = dict(self.props) if self.props is not None else None
        text = self.text
        if text is None and self.node_type in {"text", "title", "subtitle", "heading"} and props is not None:
            text = props.pop("text", None)
        return Node(
            type=self.node_type,
            class_=self.class_,
            id=self.id_,
            style=dict(self.style) if self.style is not None else None,
            text=text,
            props=props,
            children=self._children or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_node().to_dict()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{type(self).__name__}(type={self.node_type!r}, children={len(self._children)})"


class Slide(_BuilderNode):
    node_type = "slide"


class Flex(_BuilderNode):
    node_type = "flex"


class HStack(_BuilderNode):
    node_type = "hstack"


class VStack(_BuilderNode):
    node_type = "vstack"


class Stack(_BuilderNode):
    node_type = "stack"


class Grid(_BuilderNode):
    node_type = "grid"


class Layer(_BuilderNode):
    node_type = "layer"


class Padding(_BuilderNode):
    node_type = "padding"


class Text(_BuilderNode):
    node_type = "text"


class Title(_BuilderNode):
    node_type = "title"


class Subtitle(_BuilderNode):
    node_type = "subtitle"


class Heading(_BuilderNode):
    node_type = "heading"


class Box(_BuilderNode):
    node_type = "box"


class KPI(_BuilderNode):
    node_type = "kpi"

    def __init__(self, label: str, value: str, delta: str | None = None, trend: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            **kwargs,
            props={
                "label": label,
                "value": value,
                "delta": delta,
                "trend": trend,
            },
        )


class Deck:
    """Top-level IR container and output helper."""

    def __init__(
        self,
        *,
        theme: str = "minimal",
        sheet: str | None = None,
        meta: Mapping[str, Any] | None = None,
        styles: Mapping[str, Mapping[str, Any]] | None = None,
        defines: Mapping[str, Any] | None = None,
    ) -> None:
        self.theme = str(theme)
        self.sheet = str(sheet) if sheet is not None else None
        self.meta = dict(meta) if meta is not None else None
        self.styles = {key: dict(value) for key, value in (styles or {}).items()}
        self.defines = (
            {key: dict(value) for key, value in (defines or {}).items()}
            if defines is not None
            else None
        )
        self._slides: list[Slide] = []

    def __iadd__(self, slide: Slide) -> "Deck":
        if not isinstance(slide, Slide):
            raise TypeError("Deck accepts Slide instances only.")
        self._slides.append(slide)
        return self

    def add(self, slide: Slide) -> Slide:
        self.__iadd__(slide)
        return slide

    def slide(self, *children: Any, **kwargs: Any) -> Slide:
        slide = Slide(*children, **kwargs)
        self._slides.append(slide)
        return slide

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "$schema": DEFAULT_SCHEMA,
            "theme": self.theme,
            "slides": [slide.to_node().to_dict() for slide in self._slides],
        }
        if self.sheet is not None:
            payload["sheet"] = self.sheet
        if self.meta is not None:
            payload["meta"] = dict(self.meta)
        if self.styles:
            payload["styles"] = dict(self.styles)
        if self.defines is not None:
            payload["defines"] = dict(self.defines)
        return payload

    def to_json_document(self) -> Document:
        return Document.from_dict(self.to_dict())

    def render(self, out: str | Path, *, strict: bool = False) -> Path:
        from paperops.slides import build as build_module

        return build_module.render_json(self.to_dict(), out=out, strict=strict)


def render_json(
    source: str | Path | Mapping[str, Any] | Document,
    *,
    out: str | Path,
    strict: bool = False,
) -> Path:
    """Compatibility helper for rendering an existing IR document."""
    if isinstance(source, Document):
        payload = source.to_dict()
    elif isinstance(source, Mapping):
        payload = dict(source)
    else:
        payload = load_json_document(source, strict=strict).to_dict()

    from paperops.slides import build as build_module
    return build_module.render_json(payload, out=out, strict=strict)


__all__ = [
    "Deck",
    "Document",
    "Flex",
    "Grid",
    "Heading",
    "HStack",
    "KPI",
    "Layer",
    "Padding",
    "Slide",
    "Stack",
    "Subtitle",
    "Text",
    "Title",
    "VStack",
    "Box",
    "render_json",
]
