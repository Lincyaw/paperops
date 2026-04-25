"""Inline HTML filter and run-style extraction used by Markdown/MDX parsers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

INLINE_HTML_TAGS = {
    "b",
    "strong",
    "i",
    "em",
    "u",
    "s",
    "del",
    "sub",
    "sup",
    "code",
    "br",
    "a",
    "span",
    "mark",
}

# Style keys that may appear on inline <span>.
SPAN_STYLE_KEYS = {
    "color",
    "background-color",
    "font-weight",
    "font-style",
    "text-decoration",
}

_SPAN_COLOR = re.compile(r"^#[0-9A-Fa-f]{3}([0-9A-Fa-f]{3})?$")
_RGB_FN = re.compile(
    r"^rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$",
    re.I,
)
_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_./-]*$")


@dataclass
class InlineHtmlError(ValueError):
    """Structured parse-time inline HTML error."""

    code: str
    message: str
    path: str = ""
    suggestion: list[str] | None = None

    def __post_init__(self) -> None:
        parts = [f"[{self.code}]"]
        if self.path:
            parts.append(f"{self.path}:")
        parts.append(self.message)
        self.args = (" ".join(parts),)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.path:
            payload["path"] = self.path
        if self.suggestion:
            payload["suggestion"] = list(self.suggestion)
        return payload


@dataclass(frozen=True)
class ParsedInlineHtml:
    tag: str
    attrs: dict[str, Any]
    content: str
    self_closing: bool
    consumed: int


def _coerce_scalar(value: str) -> bool | int | float | str:
    if value.lower() in {"true", "yes", "on"}:
        return True
    if value.lower() in {"false", "no", "off"}:
        return False
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_attributes(raw: str, *, path: str = "") -> dict[str, Any]:
    """Parse HTML attribute payload from a tag.

    Supports a small HTML-like subset:
    - key="value"
    - key='value'
    - key=value
    - key (boolean)
    """

    attrs: dict[str, Any] = {}
    i = 0
    n = len(raw)

    while i < n:
        while i < n and raw[i].isspace():
            i += 1
        if i >= n:
            break

        name_start = i
        while i < n and re.match(r"[A-Za-z0-9_:-]", raw[i]):
            i += 1
        name = raw[name_start:i].strip()
        if not name:
            i += 1
            continue

        while i < n and raw[i].isspace():
            i += 1

        if i >= n or raw[i] != "=":
            attrs[name] = True
            continue
        i += 1

        while i < n and raw[i].isspace():
            i += 1
        if i >= n:
            attrs[name] = ""
            break

        quote = raw[i]
        if quote in {'"', "'"}:
            i += 1
            value_start = i
            while i < n:
                ch = raw[i]
                if ch == "\\" and i + 1 < n:
                    i += 2
                    continue
                if ch == quote:
                    break
                i += 1
            attrs[name] = _coerce_scalar(raw[value_start:i])
            if i < n:
                i += 1
            continue

        value_start = i
        while i < n and not raw[i].isspace():
            i += 1
        attrs[name] = _coerce_scalar(raw[value_start:i])

    return attrs


def parse_style_declarations(raw: str, *, path: str = "") -> dict[str, str]:
    """Parse a ``style="..."`` declaration block."""

    style: dict[str, str] = {}
    for chunk in (chunk.strip() for chunk in raw.split(";")):
        if not chunk:
            continue
        if ":" not in chunk:
            raise InlineHtmlError(
                "UNSUPPORTED_INLINE_HTML",
                f"Invalid style declaration {chunk!r}",
                path=path,
            )
        key, value = chunk.split(":", 1)
        key = key.strip()
        if key not in SPAN_STYLE_KEYS:
            raise InlineHtmlError(
                "UNSUPPORTED_INLINE_HTML",
                f"Unsupported <span> style property {key!r}",
                path=path,
                suggestion=[
                    "Allowed: color, background-color, font-weight, font-style, text-decoration"
                ],
            )
        style[key] = value.strip()
    return style


def _normalize_style_token(value: str, *, path: str = "") -> str:
    value = value.strip()
    if not value:
        raise InlineHtmlError(
            "UNSUPPORTED_INLINE_HTML",
            "Empty inline HTML style value",
            path=path,
        )
    if value.lower() in {"normal", "none", "auto", "inherit"}:
        return value.lower()
    if _SPAN_COLOR.fullmatch(value):
        return value
    if _RGB_FN.fullmatch(value):
        return value
    if _TOKEN.fullmatch(value):
        return value
    raise InlineHtmlError(
        "UNSUPPORTED_INLINE_HTML",
        f"Unsupported style value {value!r}; use hex/rgb token or theme token",
        path=path,
    )


def normalize_span_style(raw: dict[str, str], *, path: str = "") -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        if key in {"color", "background-color"}:
            normalized[key] = _normalize_style_token(str(value), path=path)
        else:
            # Keep CSS keyword-like values unchanged for non-color props.
            if not _TOKEN.fullmatch(str(value)) and " " not in str(value):
                # Permit token and simple phrases like "line-through".
                _normalize_style_token(str(value), path=path)
            normalized[key] = str(value)
    return normalized


def _find_matching_close(remaining: str, tag: str) -> int | None:
    close_tag = re.compile(rf"</\s*{re.escape(tag)}\s*>", re.I)
    pos = close_tag.search(remaining)
    return pos.start() if pos else None


def _consume_matching_close(remaining: str, tag: str, start_offset: int) -> int:
    close = re.search(rf"</\s*{re.escape(tag)}\s*>", remaining, flags=re.I)
    if close is None:
        raise InlineHtmlError(
            "UNSUPPORTED_INLINE_HTML",
            f"Missing closing tag for <{tag}>",
            path="",
        )
    return start_offset + close.end()


def parse_inline_html(
    text: str, *, path: str = "", start: int = 0
) -> tuple[ParsedInlineHtml | None, int]:
    """Parse one inline HTML token at ``text[start:]``.

    Returns (element, consumed_chars). If no valid inline HTML starts here,
    returns (None, 0).
    """

    if start >= len(text) or text[start] != "<":
        return None, 0

    # Requirement: < followed by whitespace or digit should stay literal.
    if start + 1 < len(text) and (
        text[start + 1].isspace() or text[start + 1].isdigit()
    ):
        return None, 0

    # Match opening tag and keep raw attribute body.
    open_match = re.match(r"<\s*([A-Za-z][A-Za-z0-9_-]*)([^>]*)>", text[start:])
    if not open_match:
        return None, 0

    tag = open_match.group(1).lower()
    raw_attrs = (open_match.group(2) or "").rstrip()

    if tag not in INLINE_HTML_TAGS:
        raise InlineHtmlError(
            "UNSUPPORTED_INLINE_HTML",
            f"Unsupported inline HTML tag <{tag}>",
            path=path,
            suggestion=["Use fenced div (:::)", "Use a registered MDX component"],
        )

    is_self_closing = raw_attrs.rstrip().endswith("/")
    if is_self_closing:
        raw_attrs = raw_attrs[:-1]
    attrs = parse_attributes(raw_attrs.strip(), path=path)

    open_end = start + open_match.end()

    if tag == "br":
        if is_self_closing:
            return (
                ParsedInlineHtml(
                    tag="br",
                    attrs=attrs,
                    content="",
                    self_closing=True,
                    consumed=open_match.end(),
                ),
                open_match.end(),
            )
        # Accept paired <br>...</br> as an alias.
        consumed = (
            _consume_matching_close(text[start:], tag=tag, start_offset=start) - start
        )
        return (
            ParsedInlineHtml(
                tag="br",
                attrs=attrs,
                content="",
                self_closing=False,
                consumed=consumed,
            ),
            consumed,
        )

    if is_self_closing:
        raise InlineHtmlError(
            "UNSUPPORTED_INLINE_HTML",
            f"<{tag}> cannot be self-closing",
            path=path,
        )

    close = re.search(rf"</\s*{re.escape(tag)}\s*>", text[open_end:], flags=re.I)
    if close is None:
        raise InlineHtmlError(
            "UNSUPPORTED_INLINE_HTML",
            f"Missing closing tag for <{tag}>",
            path=path,
        )

    content = text[open_end : open_end + close.start()]
    consumed = open_match.end() + close.end()

    if tag == "a" and "href" not in attrs:
        raise InlineHtmlError(
            "UNSUPPORTED_INLINE_HTML",
            "<a> requires href attribute",
            path=path,
        )

    if tag == "span" and "style" in attrs:
        style_value = attrs["style"]
        if not isinstance(style_value, str):
            raise InlineHtmlError(
                "UNSUPPORTED_INLINE_HTML",
                "<span> style must be a string",
                path=path,
            )
        attrs["style"] = normalize_span_style(
            parse_style_declarations(style_value, path=path), path=path
        )

    return (
        ParsedInlineHtml(
            tag=tag,
            attrs=attrs,
            content=content,
            self_closing=False,
            consumed=consumed,
        ),
        start + consumed,
    )


def run_style_overrides(tag: str, attrs: dict[str, Any]) -> dict[str, Any]:
    """Map inline tag + attributes into text run style overrides."""

    overrides: dict[str, Any] = {}

    if tag in {"b", "strong"}:
        overrides["bold"] = True
    elif tag in {"i", "em"}:
        overrides["italic"] = True
    elif tag == "u":
        overrides["underline"] = True
    elif tag in {"s", "del"}:
        overrides["strike"] = True
    elif tag == "sub":
        overrides["vertAlign"] = "sub"
    elif tag == "sup":
        overrides["vertAlign"] = "sup"
    elif tag == "code":
        overrides["font_family"] = "font_mono"
    elif tag == "mark":
        overrides["highlight"] = True
    elif tag == "a":
        href = attrs.get("href")
        if href is not None:
            overrides["href"] = str(href)
    elif tag == "span":
        style = attrs.get("style")
        if isinstance(style, dict):
            if "color" in style:
                overrides["color"] = style["color"]
            if "background-color" in style:
                overrides["background_color"] = style["background-color"]
            if "font-weight" in style:
                overrides["font_weight"] = style["font-weight"]
            if "font-style" in style:
                overrides["font_style"] = style["font-style"]
            if "text-decoration" in style:
                overrides["text_decoration"] = style["text-decoration"]

    return overrides
