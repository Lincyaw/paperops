"""Markdown + frontmatter + inline HTML parser for the slide authoring DSL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from paperops.slides.dsl.inline_html import parse_inline_html
from paperops.slides.dsl.json_loader import DEFAULT_SCHEMA, Document
from paperops.slides.ir.node import Node


@dataclass(frozen=True)
class MarkdownParseError(ValueError):
    code: str
    path: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", (f"[{self.code}] {self.path}: {self.message}",))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


_FRONT_MATTER_DELIM = "---"
_SLIDE_SEPARATOR = "---"

_ATTR_BLOCK_RE = re.compile(r"^\{(.*)\}$")
_HEADING_RE = re.compile(r"^(?P<hash>#{1,6})\s+(?P<body>.+)$")
_FENCE_OPEN_RE = re.compile(r"^(?P<marker>:{3,})(?P<rest>.*)$")
_LIST_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>(?:-|\*|\+|\d+\.))\s+(?P<text>.*)$")


def _coerce_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""

    lower = value.lower()
    if lower in {"null", "none"}:
        return None
    if lower in {"true", "on", "yes"}:
        return True
    if lower in {"false", "off", "no"}:
        return False

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
        return value[1:-1]

    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_frontmatter_block(
    lines: list[str],
    base_indent: int = 0,
) -> tuple[dict[str, Any], int]:
    data: dict[str, Any] = {}
    i = 0

    while i < len(lines):
        raw_line = lines[i]
        if not raw_line.strip():
            i += 1
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent < base_indent:
            break

        stripped = raw_line.strip()
        if ":" not in stripped:
            i += 1
            continue

        key, value_part = stripped.split(":", 1)
        key = key.strip()
        i += 1
        if not key:
            continue

        value_part = value_part.strip()
        if value_part:
            data[key] = _coerce_scalar(value_part)
            continue

        nested_lines: list[str] = []
        while i < len(lines):
            candidate = lines[i]
            nested_indent = len(candidate) - len(candidate.lstrip(" "))
            if not candidate.strip() and nested_indent <= indent:
                break
            if nested_indent <= indent:
                break
            nested_lines.append(candidate)
            i += 1

        if not nested_lines:
            data[key] = {}
            continue

        is_list = any(item.lstrip().startswith("-") for item in nested_lines)
        if is_list:
            values: list[Any] = []
            for nested_line in nested_lines:
                text = nested_line.strip()
                if not text.startswith("-"):
                    continue
                raw_value = text[1:].strip()
                if raw_value:
                    values.append(_coerce_scalar(raw_value))
            data[key] = values
            continue

        nested, _ = _parse_frontmatter_block(nested_lines, base_indent=indent + 2)
        data[key] = nested

    return data, i


def _parse_frontmatter(source: str) -> tuple[dict[str, Any], str]:
    lines = source.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER_DELIM:
        return {}, source

    end = None
    for index in range(1, len(lines)):
        if lines[index].strip() == _FRONT_MATTER_DELIM:
            end = index
            break
    if end is None:
        return {}, source

    metadata, _ = _parse_frontmatter_block(lines[1:end], base_indent=0)
    return metadata, "\n".join(lines[end + 1 :])


def _tokenize_attr_tokens(raw: str) -> list[str]:
    tokens: list[str] = []
    buffer: list[str] = []
    i = 0
    n = len(raw)
    quote: str | None = None
    brace = 0

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

        if quote is None and ch == "{":
            brace += 1
            buffer.append(ch)
            i += 1
            continue
        if quote is None and ch == "}" and brace > 0:
            brace -= 1
            buffer.append(ch)
            i += 1
            continue

        if quote is None and brace == 0 and ch.isspace():
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


def _parse_attribute_block(raw: str) -> tuple[str | None, str | None, dict[str, Any]]:
    match = _ATTR_BLOCK_RE.match(raw.strip())
    if not match:
        return None, None, {}

    body = match.group(1).strip()
    if not body:
        return None, None, {}

    class_names: list[str] = []
    node_id: str | None = None
    style: dict[str, Any] = {}

    for token in _tokenize_attr_tokens(body):
        if token.startswith("."):
            if token[1:]:
                class_names.append(token[1:])
            continue
        if token.startswith("#"):
            node_id = token[1:]
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            key = key.strip()
            if key:
                value = value.strip()
                if (
                    (value.startswith('"') and value.endswith('"'))
                    or (value.startswith("'") and value.endswith("'"))
                ):
                    value = value[1:-1]
                style[key] = _coerce_scalar(value)
            continue
        if token:
            class_names.append(token)

    return (" ".join(filter(None, class_names)) or None), node_id, style


def _split_trailing_attributes(text: str) -> tuple[str, str | None]:
    if not text.rstrip().endswith("}"):
        return text, None
    open_index = text.rfind("{")
    if open_index < 0:
        return text, None
    possible = text[open_index:].strip()
    if "}" not in possible:
        return text, None
    return text[:open_index].rstrip(), possible


def _has_markup(runs: list[str | Node]) -> bool:
    return any(isinstance(item, Node) for item in runs)


def _parse_fence_open(line: str) -> tuple[int, str | None, str | None, dict[str, Any]]:
    match = _FENCE_OPEN_RE.match(line)
    if not match:
        return 0, None, None, {}

    marker = match.group("marker")
    raw_rest = match.group("rest").strip()
    if not marker:
        return 0, None, None, {}

    node_type: str | None = "prose"
    class_name: str | None = None
    style: dict[str, Any] = {}

    if not raw_rest:
        return len(marker), node_type, class_name, style

    if raw_rest.startswith("{"):
        if not raw_rest.endswith("}"):
            raise MarkdownParseError(
                "INVALID_SYNTAX",
                "markdown",
                f"Invalid fenced div attribute block: {line!r}",
            )
        class_name, _, style = _parse_attribute_block(raw_rest)
        return len(marker), node_type, class_name, style

    attr_part = ""
    token_part = raw_rest
    if "{" in raw_rest and raw_rest.rstrip().endswith("}"):
        index = raw_rest.find("{")
        token_part = raw_rest[:index].strip()
        attr_part = raw_rest[index:].strip()

    if token_part:
        if (
            token_part.startswith(".")
            or token_part.startswith("#")
            or "=" in token_part
            or " " in token_part
        ):
            parsed_class, _, parsed_style = _parse_attribute_block("{" + token_part + "}")
            class_name = parsed_class
            style.update(parsed_style)
        else:
            head, *tail = token_part.split(".")
            if head:
                node_type = head
            if tail:
                class_name = " ".join(part for part in tail if part)

    if attr_part:
        parsed_class, _, parsed_style = _parse_attribute_block(attr_part)
        if parsed_class:
            class_name = ((class_name or "") + " " + parsed_class).strip() or None
        style.update(parsed_style)

    return len(marker), node_type, class_name, style


def _is_fence_close(line: str, marker_len: int) -> bool:
    stripped = line.strip()
    return len(stripped) >= marker_len and len(stripped) <= marker_len and all(ch == ":" for ch in stripped)


def _parse_fenced_block(
    lines: list[str],
    start: int,
    *,
    path: str,
    stop_fence: int,
) -> tuple[Node, int]:
    marker_len, node_type, node_class, style = _parse_fence_open(lines[start])
    if marker_len == 0 or node_type is None:
        raise MarkdownParseError(
            "INVALID_SYNTAX",
            path,
            f"Invalid fenced block: {lines[start]!r}",
        )

    children, next_index = _parse_block_nodes(
        lines,
        start + 1,
        path=path,
        allow_slide_sep=False,
        stop_fence=marker_len,
    )
    if next_index >= len(lines) or not _is_fence_close(lines[next_index], stop_fence):
        raise MarkdownParseError(
            "INVALID_SYNTAX",
            path,
            f"Unclosed fenced block: {lines[start]!r}",
        )

    return (
        Node(
            type=node_type,
            class_=node_class,
            style=style or None,
            children=children or None,
        ),
        next_index + 1,
    )


def _merge_adjacent_text_nodes(items: list[str | Node]) -> list[str | Node]:
    merged: list[str | Node] = []
    for item in items:
        if isinstance(item, str) and merged and isinstance(merged[-1], str):
            merged[-1] = merged[-1] + item
        else:
            merged.append(item)
    return merged


def _parse_inline_runs(text: str, *, path: str) -> list[str | Node]:
    out: list[str | Node] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            out.append(text[i + 1])
            i += 2
            continue

        if text.startswith("**", i):
            close = text.find("**", i + 2)
            if close >= 0:
                inner = _parse_inline_runs(text[i + 2 : close], path=path)
                out.append(Node(type="strong", children=inner if inner else []))
                i = close + 2
                continue
            out.append("**")
            i += 2
            continue

        if text.startswith("*", i):
            close = text.find("*", i + 1)
            if close >= 0:
                inner = _parse_inline_runs(text[i + 1 : close], path=path)
                out.append(Node(type="em", children=inner if inner else []))
                i = close + 1
                continue
            out.append("*")
            i += 1
            continue

        if text.startswith("`", i):
            close = text.find("`", i + 1)
            if close >= 0:
                code_text = text[i + 1 : close]
                out.append(Node(type="code", children=[code_text]))
                i = close + 1
                continue
            out.append("`")
            i += 1
            continue

        if ch == "<":
            parsed, consumed = parse_inline_html(text, path=path, start=i)
            if parsed is not None:
                if parsed.tag == "br" and parsed.attrs == {}:
                    out.append(Node(type="br", props={}, children=[]))
                    i = consumed
                    continue

                child_runs = (
                    _parse_inline_runs(parsed.content, path=path) if parsed.content else []
                )
                out.append(Node(type=parsed.tag, props=parsed.attrs, children=child_runs))
                i = consumed
                continue

            out.append(ch)
            i += 1
            continue

        out.append(ch)
        i += 1

    return _merge_adjacent_text_nodes(out)


def _has_only_text(nodes: list[str | Node] | None) -> bool:
    if not nodes:
        return False
    return all(isinstance(item, str) for item in nodes)


def _extract_text(nodes: list[str | Node] | None) -> str:
    if not nodes:
        return ""

    parts: list[str] = []
    for item in nodes:
        if isinstance(item, str):
            parts.append(item)
        else:
            if item.text is not None:
                parts.append(item.text)
            parts.extend([_extract_text(list(item.children or []))])
    return "".join(parts)


def _apply_node_attributes(
    node: Node,
    class_name: str | None,
    node_id: str | None,
    style: dict[str, Any],
) -> Node:
    merged_class = node.class_
    if class_name is not None:
        merged = (merged_class or "").split()
        for token in class_name.split():
            if token and token not in merged:
                merged.append(token)
        merged_class = " ".join(merged).strip() or None

    merged_style = dict(node.style or {})
    merged_style.update(style)
    return Node(
        type=node.type,
        class_=merged_class,
        id=node_id if node_id is not None else node.id,
        style=merged_style or None,
        text=node.text,
        props=node.props,
        children=node.children,
    )


def _parse_heading(line: str, path: str) -> tuple[Node, int]:
    match = _HEADING_RE.match(line)
    if not match:
        raise MarkdownParseError("INVALID_SYNTAX", path, f"Invalid heading: {line!r}")

    level = len(match.group("hash"))
    body = match.group("body")
    body, attr = _split_trailing_attributes(body)
    class_name = None
    node_id = None
    style: dict[str, Any] = {}
    if attr is not None:
        class_name, node_id, style = _parse_attribute_block(attr)

    node_type = "title" if level == 1 else "heading"
    runs = _parse_inline_runs(body, path=path)
    text = _extract_text(runs)
    children: list[str | Node] | None = None
    if _has_only_text(runs):
        runs = None
        children = None
    else:
        children = runs
    return (
        _apply_node_attributes(
            Node(
                type=node_type,
                text=text,
                class_=class_name,
                id=node_id,
                style=style or None,
                children=children,
            ),
            class_name,
            node_id,
            style,
        ),
        1,
    )


def _parse_list_block(lines: list[str], start: int, *, path: str) -> tuple[Node, int]:
    i = start
    first = _LIST_RE.match(lines[i])
    if not first:
        raise MarkdownParseError("INVALID_SYNTAX", path, f"Invalid list block: {lines[i]!r}")

    base_indent = len(first.group("indent"))
    items: list[Node] = []

    while i < len(lines):
        current = _LIST_RE.match(lines[i])
        if not current:
            break
        if len(current.group("indent")) != base_indent:
            break

        marker = current.group("marker")
        if marker.endswith("."):
            _ = marker

        raw = current.group("text")
        item_runs = _parse_inline_runs(raw, path=path)
        item_text = _extract_text(item_runs)
        item_children = item_runs if _has_markup(item_runs) else None
        items.append(Node(type="prose", text=item_text, children=item_children))
        i += 1

    return Node(type="list", children=items), i


def _parse_paragraph(lines: list[str], start: int, *, path: str) -> tuple[Node, int]:
    raw_lines: list[str] = []
    i = start

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            break
        if stripped == _SLIDE_SEPARATOR:
            break
        if _FENCE_OPEN_RE.match(line):
            break
        if _HEADING_RE.match(stripped):
            break
        if _LIST_RE.match(line):
            break
        if stripped.startswith("{") and _ATTR_BLOCK_RE.match(stripped):
            break

        if stripped.startswith(">"):
            raw_lines.append(stripped.lstrip("> ").lstrip(">"))
        else:
            raw_lines.append(line.rstrip())
        i += 1

    paragraph_text = " ".join(item.strip() for item in raw_lines).strip()
    if not paragraph_text:
        return Node(type="prose", text="", children=None), i

    runs = _parse_inline_runs(paragraph_text, path=path)
    text = _extract_text(runs)
    children: list[str | Node] | None = runs if _has_markup(runs) else None
    return Node(type="prose", text=text, children=children), i


def _is_paragraph_attribute(line: str) -> bool:
    return bool(_ATTR_BLOCK_RE.match(line))


def _parse_block_nodes(
    lines: list[str],
    start: int,
    *,
    path: str,
    allow_slide_sep: bool,
    stop_fence: int | None = None,
) -> tuple[list[Node], int]:
    nodes: list[Node] = []
    i = start

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stop_fence is not None and _is_fence_close(line, stop_fence):
            break
        if not stripped:
            i += 1
            continue
        if allow_slide_sep and stripped == _SLIDE_SEPARATOR:
            break

        if _FENCE_OPEN_RE.match(line):
            marker_len, _node_type, _class_name, _style = _parse_fence_open(line)
            if marker_len:
                node, i = _parse_fenced_block(
                    lines,
                    i,
                    path=path,
                    stop_fence=marker_len,
                )
                nodes.append(node)
                continue

        if _HEADING_RE.match(stripped):
            node, consumed = _parse_heading(stripped, path=f"{path}[{i}]")
            nodes.append(node)
            i += consumed
            continue

        if _LIST_RE.match(line):
            node, i = _parse_list_block(lines, i, path=f"{path}[{i}]")
            nodes.append(node)
            continue

        if _is_paragraph_attribute(stripped) and nodes:
            class_name, node_id, style = _parse_attribute_block(stripped)
            nodes[-1] = _apply_node_attributes(nodes[-1], class_name, node_id, style)
            i += 1
            continue

        node, i = _parse_paragraph(lines, i, path=f"{path}[{i}]")
        if node.text is not None or node.children is not None:
            nodes.append(node)

    return nodes, i


def parse_markdown_fragment(source: str, *, path: str = "markdown") -> list[Node]:
    nodes, _ = _parse_block_nodes(
        source.splitlines(),
        0,
        path=path,
        allow_slide_sep=False,
    )
    return nodes


def parse_markdown_text(
    source: str,
    *,
    path: str = "markdown",
    parse_front_matter: bool = True,
) -> tuple[Document, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    body = source
    if parse_front_matter:
        metadata, body = _parse_frontmatter(source)

    frontmatter = {
        key: metadata[key]
        for key in ["theme", "sheet", "meta", "styles", "defines", "$schema"]
        if key in metadata
    }
    metadata_rest = {key: value for key, value in metadata.items() if key not in frontmatter}

    lines = body.splitlines()
    slides: list[list[Node]] = [[]]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == _SLIDE_SEPARATOR:
            if slides and slides[-1] != []:
                slides.append([])
            else:
                slides.append([])
            i += 1
            continue

        block_nodes, i = _parse_block_nodes(
            lines,
            i,
            path=path,
            allow_slide_sep=True,
        )
        if block_nodes:
            slides[-1].extend(block_nodes)

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
    return parse_markdown_text(
        Path(source).read_text(encoding="utf-8") if isinstance(source, Path) else str(source),
        path=str(source),
    )[0]


def load_markdown_document(source: str | Path) -> Document:
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
        path = str(source)
    else:
        text = str(source)
        path = "<memory>"

    document, _ = parse_markdown_text(text, path=path, parse_front_matter=True)
    return document


__all__ = [
    "MarkdownParseError",
    "load_markdown_document",
    "parse_markdown_fragment",
    "parse_markdown_text",
    "to_canonical_ir",
]
