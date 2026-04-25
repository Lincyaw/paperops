from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

from paperops.slides.core.theme import Theme
from paperops.slides.core.tokens import UnknownTokenCategoryError, UnknownTokenError
from paperops.slides.ir.node import Node
from paperops.slides.ir.schema import STYLE_KEY_SCHEMAS
from paperops.slides.style.computed import ComputedStyle
from paperops.slides.style.selector import matches_selector
from paperops.slides.style.stylesheet import StyleSheet

_KNOWN_STYLE_KEYS = set(STYLE_KEY_SCHEMAS.keys()) | {"lang"}
_INHERITABLE_STYLE_KEYS = {
    "color",
    "font",
    "font-family",
    "font-weight",
    "font-style",
    "line-height",
    "text-align",
    "lang",
}

_BLACKLIST_STYLE_KEYS: dict[str, dict[str, list[str] | str]] = {
    "display": {
        "hint": "Layout is selected by component type, not CSS display.",
        "suggestions": ["<Flex>", "<Grid>", "<Stack>", "<Absolute>"],
    },
    "position": {
        "hint": "Absolute positioning is a component-level concern in paperops.",
        "suggestions": ["<Absolute>"],
    },
    "float": {
        "hint": "Float-based layout is unsupported in this DSL.",
        "suggestions": ["<Grid>"],
    },
    "clear": {
        "hint": "Float-clearing is unsupported in this DSL.",
        "suggestions": ["<Grid>"],
    },
    "z-index": {
        "hint": "Stacking order is component-based and explicit in <Layer>.",
        "suggestions": ["<Layer>"],
    },
    "transform": {
        "hint": "Use animation or components instead of CSS transform.",
        "suggestions": ["<Rotate>", "<Scale>", "animate"],
    },
    "transition": {
        "hint": "Transitions are handled with animation keys.",
        "suggestions": ["animate", "duration"],
    },
}

_COLOR_KEYS = {
    "color",
    "bg",
    "border",
    "border-left",
    "border-top",
    "border-right",
    "border-bottom",
}
_FONT_KEYS = {"font"}
_SPACING_KEYS = {
    "padding",
    "padding-x",
    "padding-y",
    "padding-top",
    "padding-right",
    "padding-bottom",
    "padding-left",
    "margin",
    "margin-x",
    "margin-y",
    "margin-top",
    "margin-right",
    "margin-bottom",
    "margin-left",
    "gap",
    "row-gap",
    "column-gap",
    "width",
    "height",
    "min-width",
    "max-width",
    "min-height",
    "max-height",
    "aspect-ratio",
    "cols",
    "rows",
}
_SHADOW_KEYS = {"shadow"}
_DURATION_KEYS = {"duration", "delay", "stagger"}


@dataclass(frozen=True)
class StyleResolutionError(ValueError):
    """Structured style error used by the style engine."""

    code: str
    message: str
    path: str
    key: str
    hint: str | None = None
    suggestion: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "key": self.key,
        }
        if self.hint is not None:
            payload["hint"] = self.hint
        if self.suggestion is not None:
            payload["suggestion"] = self.suggestion
        return payload


@dataclass(frozen=True)
class CascadeResult:
    """Computed style output for an IR tree."""

    computed: dict[int, ComputedStyle]
    errors: list[StyleResolutionError]
    root: Node | None = None
    trace: list[dict[str, Any]] | None = None

    def style_for(self, node_or_id: Node | int) -> ComputedStyle:
        key = node_or_id if isinstance(node_or_id, int) else id(node_or_id)
        return self.computed[key]

    def __getitem__(self, node_or_id: Node | int) -> ComputedStyle:
        return self.style_for(node_or_id)


def resolve_computed_styles(
    root: Node | Mapping[str, Any],
    *,
    theme: Theme,
    theme_defaults: Mapping[str, Mapping[str, Any]] | None = None,
    style_sheet: StyleSheet | Mapping[str, Mapping[str, Any]] | None = None,
    deck_style: StyleSheet | Mapping[str, Mapping[str, Any]] | None = None,
    strict: bool = True,
) -> CascadeResult:
    """Compute ComputedStyle for every node in an IR tree."""
    sheet = _as_stylesheet(style_sheet)
    deck = _as_stylesheet(deck_style)
    defaults = dict(theme_defaults or {})
    computed: dict[int, ComputedStyle] = {}
    errors: list[StyleResolutionError] = []
    trace: list[dict[str, Any]] = []

    root_node = _coerce_node(root, path="root")
    _walk_node(
        root_node,
        ancestors=(),
        sibling_meta=(None,),
        theme=theme,
        defaults=defaults,
        style_sheet=sheet,
        deck_style=deck,
        path="root",
        parent_computed=None,
        computed=computed,
        errors=errors,
        trace=trace,
    )

    if strict and errors:
        raise errors[0]
    return CascadeResult(computed=computed, errors=errors, root=root_node, trace=trace)


def _walk_node(
    node: Node,
    *,
    ancestors: tuple[Node, ...],
    sibling_meta: tuple[tuple[int, int] | None, ...],
    theme: Theme,
    defaults: Mapping[str, Mapping[str, Any]],
    style_sheet: StyleSheet,
    deck_style: StyleSheet,
    path: str,
    parent_computed: ComputedStyle | None,
    computed: dict[int, ComputedStyle],
    errors: list[StyleResolutionError],
    trace: list[dict[str, Any]],
) -> None:
    current_computed = _compute_node_style(
        node=node,
        theme=theme,
        defaults=defaults,
        style_sheet=style_sheet,
        deck_style=deck_style,
        ancestors=ancestors,
        sibling_meta=sibling_meta,
        parent_style=parent_computed,
        path=path,
        errors=errors,
        trace=trace,
    )
    object.__setattr__(node, "computed_style", current_computed)
    computed[id(node)] = current_computed

    children = _node_children(node)
    if not children:
        return

    totals = _same_type_counts(children)
    running = {}

    for index, child in enumerate(children):
        if not isinstance(child, Node):
            continue
        next_meta = _same_type_meta(child, totals, running)
        _walk_node(
            child,
            ancestors=(*ancestors, node),
            sibling_meta=(*sibling_meta, next_meta),
            theme=theme,
            defaults=defaults,
            style_sheet=style_sheet,
            deck_style=deck_style,
            path=f"{path}.children[{index}]",
            parent_computed=current_computed,
            computed=computed,
            errors=errors,
            trace=trace,
        )


def _compute_node_style(
    *,
    node: Node,
    theme: Theme,
    defaults: Mapping[str, Mapping[str, Any]],
    style_sheet: StyleSheet,
    deck_style: StyleSheet,
    ancestors: tuple[Node, ...],
    sibling_meta: tuple[tuple[int, int] | None, ...],
    parent_style: ComputedStyle | None,
    path: str,
    errors: list[StyleResolutionError],
    trace: list[dict[str, Any]],
) -> ComputedStyle:
    style_candidates: dict[str, tuple[tuple[int, int, int, int], Any]] = {}
    matched_rules: list[str] = []

    theme_defaults = defaults.get(node.type, {}) if defaults else {}
    _apply_style_source(
        style_candidates,
        declarations=theme_defaults,
        origin_rank=0,
        source_order=0,
        specificity=0,
        node=node,
        theme=theme,
        path=path,
        errors=errors,
    )

    for rule in style_sheet:
        if matches_selector(rule.selector, node, ancestors, sibling_meta):
            matched_rules.append(rule.selector_text)
            _apply_style_source(
                style_candidates,
                declarations=rule.declarations,
                origin_rank=1,
                source_order=rule.source_index,
                specificity=rule.specificity,
                node=node,
                theme=theme,
                path=path,
                errors=errors,
            )

    for rule in deck_style:
        if matches_selector(rule.selector, node, ancestors, sibling_meta):
            matched_rules.append(rule.selector_text)
            _apply_style_source(
                style_candidates,
                declarations=rule.declarations,
                origin_rank=2,
                source_order=rule.source_index,
                specificity=rule.specificity,
                node=node,
                theme=theme,
                path=path,
                errors=errors,
            )

    inline_style = _node_style(node)
    _apply_style_source(
        style_candidates,
        declarations=inline_style,
        origin_rank=3,
        source_order=0,
        specificity=0,
        node=node,
        theme=theme,
        path=path,
        errors=errors,
    )

    resolved = {
        key: value for key, (_, value) in style_candidates.items() if value is not None
    }

    for key in _INHERITABLE_STYLE_KEYS:
        if key in resolved and resolved[key] == "inherit":
            if parent_style is None:
                resolved.pop(key, None)
                continue
            inherited = parent_style.get(key)
            if inherited is not None:
                resolved[key] = inherited
            else:
                resolved.pop(key, None)
        elif key not in resolved and parent_style is not None:
            inherited = parent_style.get(key)
            if inherited is not None:
                resolved[key] = inherited

    trace.append(
        {
            "stage": "style",
            "node": path,
            "matched_rules": matched_rules,
            "computed_style_keys": sorted(resolved.keys()),
        }
    )
    return ComputedStyle(resolved, parent=parent_style)


def _apply_style_source(
    style_candidates: dict[str, tuple[tuple[int, int, int, int], Any]],
    declarations: Mapping[str, Any] | None,
    *,
    origin_rank: int,
    source_order: int,
    specificity: int,
    node: Node,
    theme: Theme,
    path: str,
    errors: list[StyleResolutionError],
) -> None:
    if not declarations:
        return

    for declaration_index, (key, raw_value) in enumerate(declarations.items()):
        if key not in _KNOWN_STYLE_KEYS and key not in _BLACKLIST_STYLE_KEYS:
            errors.append(
                StyleResolutionError(
                    code="UNKNOWN_STYLE_KEY",
                    message=f"Unknown style key: {key!r}",
                    path=path,
                    key=key,
                    hint="Unknown style key is not in the allowlist.",
                )
            )
            continue

        if key in _BLACKLIST_STYLE_KEYS:
            info = _BLACKLIST_STYLE_KEYS[key]
            errors.append(
                StyleResolutionError(
                    code="UNKNOWN_STYLE_KEY",
                    message=f"Unknown style key: {key!r}",
                    path=path,
                    key=key,
                    hint=str(info.get("hint", "")),
                    suggestion=list(info["suggestions"]),
                )
            )
            continue

        try:
            resolved_value = _resolve_style_value(theme, key, raw_value)
        except (
            UnknownTokenCategoryError,
            UnknownTokenError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(
                StyleResolutionError(
                    code="INVALID_STYLE_VALUE",
                    message=f"Unable to resolve style key {key!r}: {raw_value!r}",
                    path=path,
                    key=key,
                    hint=str(exc),
                )
            )
            continue

        score = (origin_rank, specificity, source_order, declaration_index)
        prev = style_candidates.get(key)
        if prev is None or score > prev[0]:
            style_candidates[key] = (score, resolved_value)


def _resolve_style_value(theme: Theme, key: str, value: Any) -> Any:
    if value in {"inherit", "auto", "none", None}:
        return value

    if key in _COLOR_KEYS:
        return theme.resolve_token("colors", value)
    if key in _FONT_KEYS:
        return theme.resolve_token("fonts", value)
    if key in _SPACING_KEYS:
        return theme.resolve_token("spacing", value)
    if key in {"radius"}:
        return theme.resolve_token("radius", value)
    if key in _SHADOW_KEYS:
        return theme.resolve_token("shadow", value)
    if key in _DURATION_KEYS:
        return theme.resolve_token("duration", value)
    if key == "line-height" and isinstance(value, str):
        return float(value) if _is_float(value) else value
    return value


def _is_float(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _as_stylesheet(
    sheet: StyleSheet | Mapping[str, Mapping[str, Any]] | None,
) -> StyleSheet:
    if sheet is None:
        return StyleSheet({})
    if isinstance(sheet, StyleSheet):
        return sheet
    return StyleSheet(sheet)


def _coerce_node(raw: Node | Mapping[str, Any], path: str) -> Node:
    if isinstance(raw, Node):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path}: node must be a Node or dict")
    return Node.from_dict(dict(raw))


def _node_children(node: Node) -> Sequence[Node | str] | None:
    return node.children or None


def _node_style(node: Node) -> Mapping[str, Any] | None:
    return node.style or None


def _same_type_counts(children: Sequence[Node | str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for child in children:
        if not isinstance(child, Node):
            continue
        counts[child.type] = counts.get(child.type, 0) + 1
    return counts


def _same_type_meta(
    child: Node,
    totals: dict[str, int],
    running: dict[str, int],
) -> tuple[int, int]:
    current = running.get(child.type, 0) + 1
    running[child.type] = current
    return current, totals[child.type]
