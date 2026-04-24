"""MDX parser that maps components and inline HTML into canonical IR nodes."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperops.slides.components.registry import registry
from paperops.slides.dsl.json_loader import DEFAULT_SCHEMA, Document
from paperops.slides.dsl.markdown_parser import _parse_frontmatter
from paperops.slides.dsl.markdown_parser import parse_markdown_fragment
from paperops.slides.ir.schema import STYLE_KEY_SCHEMAS
from paperops.slides.ir.node import Node


@dataclass(frozen=True)
class MDXParseError(ValueError):
    """Structured parser error for MDX parsing."""

    code: str
    path: str
    message: str
    suggestion: list[str] | None = None

    def __post_init__(self) -> None:
        suffix = f" suggestion={self.suggestion}" if self.suggestion else ""
        object.__setattr__(self, "args", (f"[{self.code}] {self.path}: {self.message}{suffix}",))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }
        if self.suggestion:
            payload["suggestion"] = list(self.suggestion)
        return payload


_OPEN_TAG_RE = re.compile(
    r"^\s*<(?P<name>[A-Za-z][A-Za-z0-9_]*)(?P<attrs>(?:\s+[^\s<>/]*(?:\s*=\s*(?:\{\{.*?\}\}|\"[^\"]*\"|'[^']*'|[^\\\\s/>]+))?)*)\s*(?P<self>/)?\s*>\s*$",
    re.DOTALL,
)
_SINGLE_TAG_RE = re.compile(
    r"^\s*<(?P<name>[A-Za-z][A-Za-z0-9_]*)(?P<attrs>(?:\s+[^\s<>/]*(?:\s*=\s*(?:\{\{.*?\}\}|\"[^\"]*\"|'[^']*'|[^\\\\s/>]+))?)*)\s*>\s*(?P<body>.*?)\s*</(?P=name)\s*>\s*$",
    re.DOTALL,
)
_CLOSE_TAG_RE = re.compile(r"^\s*</(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*>\s*$")
_STYLE_KEYS = set(STYLE_KEY_SCHEMAS)
_IR_LAYOUT_TYPES = {"grid", "flex", "hstack", "stack", "vstack", "layer", "absolute", "padding"}


def _is_component_tag(name: str) -> bool:
    return bool(name) and name[0].isupper()


def _coerce_js_scalar(value: str) -> Any:
    raw = value.strip()
    if not raw:
        return True
    lowered = raw.lower()
    if lowered in {"true", "on", "yes"}:
        return True
    if lowered in {"false", "off", "no"}:
        return False
    if lowered in {"none", "null"}:
        return None
    if (
        (raw.startswith("\"") and raw.endswith("\""))
        or (raw.startswith("'") and raw.endswith("'"))
    ):
        return raw[1:-1]
    if re.fullmatch(r"-?\d+", raw):
        try:
            return int(raw)
        except ValueError:
            pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _is_supported_text_component(name: str) -> bool:
    return name in {"title", "subtitle", "heading", "text", "prose"}


def _node_is_text_only(node: Node | str) -> bool:
    if isinstance(node, str):
        return True
    if node.text is None and not node.children:
        return False
    if node.children is None:
        return True
    return all(_node_is_text_only(child) for child in node.children)


def _flatten_node_text(node: Node | str) -> str:
    if isinstance(node, str):
        return node
    text = node.text or ""
    if node.children:
        text += "".join(_flatten_node_text(child) for child in node.children)
    return text


def _build_single_line_component(
    node_type: str,
    props: dict[str, Any],
    style: dict[str, Any],
    class_name: str | None,
    node_id: str | None,
    body: str,
    path: str,
) -> Node:
    children = parse_markdown_fragment(body, path=path)
    if _is_supported_text_component(node_type) and children and all(
        _node_is_text_only(child) for child in children
    ):
        text = "".join(_flatten_node_text(child) for child in children).strip()
        if text:
            return Node(
                type=node_type,
                class_=class_name,
                id=node_id,
                style=style or None,
                text=text,
                props=props or None,
                children=None,
            )

    return Node(
        type=node_type,
        class_=class_name,
        id=node_id,
        style=style or None,
        props=props or None,
        children=children or None,
    )


def _tokenize_attr_tokens(raw: str) -> list[str]:
    """Split a JSX-like attribute string into tokens."""

    tokens: list[str] = []
    i = 0
    n = len(raw)
    quote: str | None = None
    brace = 0
    bracket = 0
    buffer: list[str] = []

    while i < n:
        ch = raw[i]

        if quote is None and ch in {"'", '"'}:
            quote = ch
            buffer.append(ch)
            i += 1
            continue

        if quote == ch:
            quote = None
            buffer.append(ch)
            i += 1
            continue

        if quote is None and raw.startswith("{{", i):
            brace += 1
            buffer.append("{{")
            i += 2
            continue

        if quote is None and raw.startswith("}}", i) and brace > 0:
            brace -= 1
            buffer.append("}}")
            i += 2
            continue

        if quote is None and ch == "[":
            bracket += 1
            buffer.append(ch)
            i += 1
            continue

        if quote is None and ch == "]":
            bracket = max(bracket - 1, 0)
            buffer.append(ch)
            i += 1
            continue

        if (
            quote is None
            and brace == 0
            and bracket == 0
            and ch.isspace()
        ):
            token = "".join(buffer).strip()
            if token:
                tokens.append(token)
            buffer = []
            i += 1
            continue

        buffer.append(ch)
        i += 1

    tail = "".join(buffer).strip()
    if tail:
        tokens.append(tail)
    return tokens


def _parse_style_object(raw: str, path: str, *, at: str) -> dict[str, Any]:
    text = raw.strip()
    if not (text.startswith("{{") and text.endswith("}}")):
        raise MDXParseError("INVALID_SYNTAX", path, f"Invalid JSX style object in {at}")

    body = text[2:-2].strip()
    if not body:
        return {}

    # Parse `key: value, key: value` with nested braces respected.
    items: list[str] = []
    token: list[str] = []
    quote: str | None = None
    nested = 0
    i = 0

    while i < len(body):
        ch = body[i]
        if quote is None and ch in {"'", '"'}:
            quote = ch
            token.append(ch)
            i += 1
            continue
        if quote == ch:
            quote = None
            token.append(ch)
            i += 1
            continue
        if quote is None and ch in {"{", "["}:
            nested += 1
            token.append(ch)
            i += 1
            continue
        if quote is None and ch in {"}", "]"}:
            nested = max(0, nested - 1)
            token.append(ch)
            i += 1
            continue
        if quote is None and ch == "," and nested == 0:
            current = "".join(token).strip()
            if current:
                items.append(current)
            token = []
            i += 1
            continue
        token.append(ch)
        i += 1

    tail = "".join(token).strip()
    if tail:
        items.append(tail)

    style: dict[str, Any] = {}
    for item in items:
        if ":" not in item:
            raise MDXParseError(
                "INVALID_SYNTAX",
                path,
                f"Invalid style token {item!r} in {at}",
            )
        raw_key, raw_value = item.split(":", 1)
        key = raw_key.strip().strip('"').strip("'")
        if not key:
            continue
        style[key] = _coerce_js_scalar(raw_value.strip())
    return style


def _parse_attr_value(value: str, path: str, *, at: str) -> Any:
    if value.startswith("{{") and value.endswith("}}"):
        return _parse_style_object(value, path, at=at)
    if value == "{}":
        return {}
    return _coerce_js_scalar(value)


def _parse_component_attrs(raw: str, path: str) -> tuple[dict[str, Any], dict[str, Any], str | None, str | None]:
    attrs = _tokenize_attr_tokens(raw)
    props: dict[str, Any] = {}
    style: dict[str, Any] = {}
    class_name: str | None = None
    node_id: str | None = None

    for raw_attr in attrs:
        if not raw_attr:
            continue
        if "=" not in raw_attr:
            props[raw_attr] = True
            continue

        key, raw_value = raw_attr.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        value = _parse_attr_value(raw_value, path, at=key)

        if key in {"class", "className"}:
            if isinstance(value, str):
                class_name = value.strip()
            continue
        if key == "id" and isinstance(value, str):
            node_id = value.strip()
            continue

        if key == "style":
            if not isinstance(value, dict):
                raise MDXParseError(
                    "INVALID_SYNTAX",
                    path,
                    "<Component style=...> expects a JSX object.",
                )
            style.update(value)
            continue

        if key in _STYLE_KEYS:
            style[key] = value
            continue

        props[key] = value

    return props, style, class_name, node_id


def _extract_component_open(line: str, path: str) -> tuple[str, dict[str, Any], dict[str, Any], str | None, str | None, bool] | None:
    match = _OPEN_TAG_RE.match(line)
    if not match:
        return None

    name = match.group("name")
    if not name or not _is_component_tag(name):
        return None

    attrs_raw = match.group("attrs") or ""
    is_self_closing = bool(match.group("self"))

    props, style, class_name, node_id = _parse_component_attrs(attrs_raw, path)
    lowered = name.lower()
    if not registry.has(lowered) and lowered not in _IR_LAYOUT_TYPES:
        suggestion = difflib.get_close_matches(
            lowered,
            registry.snapshot().keys(),
            n=4,
            cutoff=0.5,
        )
        suggestion_text = ", ".join(suggestion) if suggestion else "a different component name?"
        raise MDXParseError(
            "UNKNOWN_TYPE",
            path,
            f"Unknown component type '{name}'. Did you mean: {suggestion_text}?",
            suggestion=[*suggestion] if suggestion else ["Did you mean a different component name?"],
        )

    return lowered, props, style, class_name, node_id, is_self_closing


def _is_matching_close(line: str, expected: str) -> bool:
    match = _CLOSE_TAG_RE.match(line)
    return bool(match and match.group("name").lower() == expected.lower())


def _flush_markdown_lines(lines: list[str], path: str) -> list[Node]:
    if not lines:
        return []
    text = "\n".join(lines)
    return parse_markdown_fragment(text, path=path)


def _parse_mdx_nodes(
    lines: list[str],
    start: int,
    path: str,
    *,
    close_tag: str | None = None,
    allow_slide_sep: bool = False,
) -> tuple[list[Node], int]:
    nodes: list[Node] = []
    i = start
    buffer: list[str] = []

    while i < len(lines):
        line = lines[i]

        if close_tag is not None and _is_matching_close(line, close_tag):
            nodes.extend(_flush_markdown_lines(buffer, path=f"{path}[{i}]"))
            buffer = []
            return nodes, i + 1

        if close_tag is None and allow_slide_sep and line.strip() == "---":
            nodes.extend(_flush_markdown_lines(buffer, path=f"{path}[{i}]"))
            buffer = []
            return nodes, i

        single_line_match = _SINGLE_TAG_RE.match(line)
        if single_line_match is not None:
            single_line_name = single_line_match.group("name")
            if _is_component_tag(single_line_name):
                single_line_attrs = single_line_match.group("attrs") or ""
                single_line_body = (single_line_match.group("body") or "").strip()
                props, style, class_name, node_id = _parse_component_attrs(
                    single_line_attrs,
                    path=f"{path}[{i}]",
                )
                single_line_type = single_line_name.lower()
                if (
                    not registry.has(single_line_type)
                    and single_line_type not in _IR_LAYOUT_TYPES
                ):
                    suggestion = difflib.get_close_matches(
                        single_line_type,
                        registry.snapshot().keys(),
                        n=4,
                        cutoff=0.5,
                    )
                    suggestion_text = ", ".join(suggestion) if suggestion else "a different component name?"
                    raise MDXParseError(
                        "UNKNOWN_TYPE",
                        f"{path}[{i}]",
                        f"Unknown component type '{single_line_name}'. Did you mean: {suggestion_text}?",
                        suggestion=[*suggestion] if suggestion else ["Did you mean a different component name?"],
                    )
                nodes.extend(_flush_markdown_lines(buffer, path=f"{path}[{i}]"))
                buffer = []
                nodes.append(
                    _build_single_line_component(
                        single_line_type,
                        props,
                        style,
                        class_name,
                        node_id,
                        single_line_body,
                        path=f"{path}[{i}]",
                    )
                )
                i += 1
                continue

        parsed = _extract_component_open(line, path=f"{path}[{i}]")
        if parsed is not None:
            node_type, props, style, class_name, node_id, is_self_closing = parsed
            nodes.extend(_flush_markdown_lines(buffer, path=f"{path}[{i}]"))
            buffer = []

            if is_self_closing:
                nodes.append(
                    Node(
                        type=node_type,
                        class_=class_name,
                        id=node_id,
                        style=style or None,
                        props=props,
                        children=None,
                    ),
                )
                i += 1
                continue

            child_nodes, i = _parse_mdx_nodes(
                lines,
                i + 1,
                path=path,
                close_tag=node_type,
                allow_slide_sep=False,
            )
            nodes.append(
                Node(
                    type=node_type,
                    class_=class_name,
                    id=node_id,
                    style=style or None,
                    props=props,
                    children=child_nodes or None,
                ),
            )
            continue

        buffer.append(line)
        i += 1

    if close_tag is not None:
        raise MDXParseError("INVALID_SYNTAX", path, f"Unclosed component <{close_tag}>")

    nodes.extend(_flush_markdown_lines(buffer, path=f"{path}[{i}]"))
    return nodes, i


def parse_mdx_fragment(source: str, *, path: str = "mdx") -> list[Node]:
    nodes, _ = _parse_mdx_nodes(
        source.splitlines(),
        0,
        path=path,
        close_tag=None,
        allow_slide_sep=False,
    )
    return nodes


def parse_mdx_text(
    source: str,
    *,
    path: str = "mdx",
) -> tuple[Document, dict[str, Any]]:
    metadata, body = _parse_frontmatter(source)

    frontmatter = {
        key: metadata[key]
        for key in ["theme", "sheet", "meta", "styles", "defines", "$schema"]
        if key in metadata
    }
    metadata_rest = {
        key: value for key, value in metadata.items() if key not in frontmatter
    }

    lines = body.splitlines()
    slides: list[list[Node]] = [[]]
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        if raw_line.strip() == "---":
            if slides and slides[-1] != []:
                slides.append([])
            else:
                slides.append([])
            i += 1
            continue

        chunk_nodes, i = _parse_mdx_nodes(
            lines,
            i,
            path=path,
            close_tag=None,
            allow_slide_sep=True,
        )
        if chunk_nodes:
            slides[-1].extend(chunk_nodes)
    while slides and slides[-1] == [] and len(slides) > 1:
        slides.pop()

    if not slides:
        slides = [[]]

    payload: dict[str, Any] = {
        "$schema": frontmatter.get("$schema", DEFAULT_SCHEMA),
        "theme": frontmatter.get("theme", "minimal"),
        "slides": [
            {"type": "slide", "children": [node.to_dict() for node in nodes]}
            for nodes in slides
        ],
    }

    if "sheet" in frontmatter:
        payload["sheet"] = frontmatter["sheet"]
    if "styles" in frontmatter:
        payload["styles"] = frontmatter["styles"]
    if "defines" in frontmatter:
        payload["defines"] = frontmatter["defines"]
    if "meta" in frontmatter:
        payload["meta"] = frontmatter["meta"]

    return Document.from_dict(payload), metadata_rest


def to_canonical_ir(source: str | Path) -> Document:
    return parse_mdx_text(
        Path(source).read_text(encoding="utf-8") if isinstance(source, Path) else str(source),
        path=str(source),
    )[0]


def load_mdx_document(source: str | Path) -> Document:
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
        path = str(source)
    else:
        text = str(source)
        path = "<memory>"

    document, _ = parse_mdx_text(text, path=path)
    return document


__all__ = [
    "MDXParseError",
    "load_mdx_document",
    "parse_mdx_fragment",
    "parse_mdx_text",
    "to_canonical_ir",
]
