"""Canonical JSON IR node model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

from paperops.slides.ir import schema

NodeChild = Union[str, "Node"]


@dataclass(frozen=True)
class Node:
    """A single IR node.

    Fields are normalized to a JSON-friendly shape and can be serialized back
    without information loss.
    """

    type: str
    class_: str | None = None
    id: str | None = None
    style: dict[str, Any] | None = None
    text: str | None = None
    props: dict[str, Any] | None = None
    children: list[NodeChild] | None = None

    def __post_init__(self):
        if not self.type or not isinstance(self.type, str):
            raise TypeError("Node.type is required and must be a string")
        if self.class_ is not None and not isinstance(self.class_, str):
            raise TypeError("Node.class must be a string when set")
        if self.id is not None and not isinstance(self.id, str):
            raise TypeError("Node.id must be a string when set")
        if self.style is not None and not isinstance(self.style, dict):
            raise TypeError("Node.style must be a mapping when set")
        if self.props is not None and not isinstance(self.props, dict):
            raise TypeError("Node.props must be a mapping when set")

        if self.children is not None:
            for child in self.children:
                if not isinstance(child, (str, Node)):
                    raise TypeError(
                        "Node.children may only contain strings or Node instances"
                    )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.text is not None:
            payload["text"] = self.text
        if self.class_ is not None:
            payload["class"] = self.class_
        if self.id is not None:
            payload["id"] = self.id
        if self.style is not None:
            payload["style"] = self.style
        if self.props is not None:
            payload["props"] = self.props
        if self.children is not None:
            payload["children"] = [
                child.to_dict() if isinstance(child, Node) else child
                for child in self.children
            ]
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Node":
        schema.validate_node(raw)
        text = raw.get("text")
        props = raw.get("props")
        return cls(
            type=raw["type"],
            class_=raw.get("class"),
            id=raw.get("id"),
            style=raw.get("style"),
            text=text,
            props=dict(props) if props is not None else None,
            children=(
                [
                    item if isinstance(item, str) else cls.from_dict(item)
                    for item in raw.get("children", [])
                ]
                if "children" in raw
                else None
            ),
        )
