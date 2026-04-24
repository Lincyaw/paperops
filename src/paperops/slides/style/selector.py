from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from paperops.slides.ir.node import Node

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class PseudoClass:
    name: str
    argument: int | None = None


@dataclass(frozen=True)
class SimpleSelector:
    tag: str | None
    id: str | None
    classes: tuple[str, ...]
    pseudos: tuple[PseudoClass, ...]
    universal: bool = False


@dataclass(frozen=True)
class ParsedSelector:
    """A parsed selector in left-to-right order."""

    steps: tuple[SimpleSelector, ...]
    combinators: tuple[str, ...]


def parse_selector(selector: str) -> ParsedSelector:
    """Parse a CSS-subset selector string.

    Supported:
      - tag selectors: `card`
      - class selectors: `.kpi`
      - id selectors: `#hero`
      - combinations: `.card.kpi`
      - descendant: `.card .value`
      - child: `.card > .value`
      - wildcard: `*`
      - pseudo: :first, :last, :only, :nth(N)
    """

    if not isinstance(selector, str) or not selector.strip():
        raise ValueError("Selector must be a non-empty string")
    text = _normalize_selector(selector)
    tokens = [token for token in text.split() if token]
    if not tokens:
        raise ValueError(f"Invalid selector: {selector!r}")

    steps: list[SimpleSelector] = []
    combinators: list[str] = []
    next_combinator = "descendant"

    for token in tokens:
        if token == ">":
            if not steps:
                raise ValueError(f"Selector cannot start with child combinator: {selector!r}")
            if combinators and combinators[-1] == "child":
                raise ValueError(f"Consecutive '>' combinators: {selector!r}")
            next_combinator = "child"
            continue

        step = _parse_simple(token, source=selector)
        steps.append(step)
        if len(steps) > 1:
            combinators.append(next_combinator)
            next_combinator = "descendant"
        elif combinators:
            # Defensive; should never happen because first step has no left relation.
            raise ValueError(f"Unexpected selector state for: {selector!r}")

    if steps and (combinators and len(combinators) != len(steps) - 1):
        raise ValueError(f"Invalid selector tokenization: {selector!r}")
    return ParsedSelector(steps=tuple(steps), combinators=tuple(combinators))


def specificity(selector: ParsedSelector | str) -> int:
    """Compute selector specificity with weights used in issue #2.

    id=100, class/pseudo=10, tag=1.
    """

    parsed = selector if isinstance(selector, ParsedSelector) else parse_selector(selector)
    ids = 0
    tags = 0
    classes_pseudos = 0
    for step in parsed.steps:
        if not step.universal and step.tag:
            tags += 1
        if step.id:
            ids += 1
        classes_pseudos += len(step.classes)
        classes_pseudos += sum(1 for pseudo in step.pseudos)
    return ids * 100 + classes_pseudos * 10 + tags * 1


def matches_selector(
    selector: ParsedSelector,
    node: Node | dict[str, Any],
    ancestors: tuple[Node | dict[str, Any], ...],
    sibling_meta: tuple[tuple[int, int] | None, ...],
) -> bool:
    """Return whether `node` is matched by `selector`.

    `ancestors` is root->parent. `sibling_meta` stores, for each node in
    ancestors plus the current node, the 1-based same-type index and total count.
    """

    if not selector.steps:
        return False
    if len(sibling_meta) != len(ancestors) + 1:
        raise ValueError("sibling_meta length must be len(ancestors) + 1")

    current_depth = len(ancestors)
    for right_step in range(len(selector.steps) - 1, -1, -1):
        current_meta = sibling_meta[current_depth]
        current_node = _get_node_by_depth(node, ancestors, current_depth)
        if current_node is None:
            return False
        if not _matches_simple(selector.steps[right_step], current_node, current_meta):
            return False

        if right_step == 0:
            return True

        relation = selector.combinators[right_step - 1]
        if relation == "child":
            if current_depth == 0:
                return False
            current_depth -= 1
        elif relation == "descendant":
            found = False
            for candidate_depth in range(current_depth - 1, -1, -1):
                candidate = _get_node_by_depth(node, ancestors, candidate_depth)
                candidate_meta = sibling_meta[candidate_depth]
                if candidate is None or candidate_meta is None:
                    continue
                if _matches_simple(
                    selector.steps[right_step - 1], candidate, candidate_meta
                ):
                    current_depth = candidate_depth
                    found = True
                    break
            if not found:
                return False
        else:
            raise ValueError(f"Unknown combinator: {relation}")
    return False


def _normalize_selector(selector: str) -> str:
    with_spaces = []
    for char in selector:
        if char == ">":
            with_spaces.append(" > ")
        else:
            with_spaces.append(char)
    return "".join(with_spaces).strip()


def _parse_simple(selector: str, source: str) -> SimpleSelector:
    idx = 0
    n = len(selector)
    tag: str | None = None
    selector_id: str | None = None
    classes: list[str] = []
    pseudos: list[PseudoClass] = []
    universal = False

    def read_name(start: int) -> tuple[str, int]:
        end = start
        while end < n and _is_name_char(selector[end]):
            end += 1
        value = selector[start:end]
        if not value or not _NAME_RE.match(value):
            raise ValueError(f"Invalid selector token {value!r} in {source!r}")
        return value, end

    if idx < n and selector[idx] == "*":
        universal = True
        idx += 1
        if idx < n and _is_name_char(selector[idx]):
            raise ValueError(f"Unexpected token after '*': {source!r}")
    elif idx < n and _is_name_char(selector[idx]):
        tag, idx = read_name(idx)

    while idx < n:
        char = selector[idx]
        if char == ".":
            name, idx = read_name(idx + 1)
            classes.append(name)
        elif char == "#":
            if selector_id is not None:
                raise ValueError(f"Only one id selector is supported: {source!r}")
            selector_id, idx = read_name(idx + 1)
        elif char == ":":
            pseudo, idx = _read_pseudo(selector, idx, source)
            pseudos.append(pseudo)
        else:
            raise ValueError(f"Unexpected char {char!r} in selector {source!r}")

    if not universal and tag is None and selector_id is None and not classes and not pseudos:
        raise ValueError(f"Empty selector component in {source!r}")
    return SimpleSelector(
        tag=tag,
        id=selector_id,
        classes=tuple(classes),
        pseudos=tuple(pseudos),
        universal=universal,
    )


def _read_pseudo(selector: str, start: int, source: str) -> tuple[PseudoClass, int]:
    if selector.startswith(":first", start):
        return PseudoClass(name="first"), start + len(":first")
    if selector.startswith(":last", start):
        return PseudoClass(name="last"), start + len(":last")
    if selector.startswith(":only", start):
        return PseudoClass(name="only"), start + len(":only")
    if not selector.startswith(":nth(", start):
        raise ValueError(f"Unsupported pseudo in selector {source!r}: ...")
    close = selector.find(")", start)
    if close == -1:
        raise ValueError(f"Unclosed :nth(...) in selector {source!r}")
    arg = selector[start + 5 : close].strip()
    if not arg.isdigit():
        raise ValueError(f"Invalid :nth() value in selector {source!r}")
    return PseudoClass(name="nth", argument=int(arg)), close + 1


def _is_name_char(ch: str) -> bool:
    return ch.isalnum() or ch in {"_", "-"}


def _get_node_by_depth(
    node: Node | dict[str, Any],
    ancestors: tuple[Node | dict[str, Any], ...],
    depth: int,
) -> Node | dict[str, Any] | None:
    if depth == len(ancestors):
        return node
    if 0 <= depth < len(ancestors):
        return ancestors[depth]
    return None


def _get_node_type(node: Node | dict[str, Any]) -> str:
    if isinstance(node, Node):
        return node.type
    return str(node["type"])


def _get_node_id(node: Node | dict[str, Any]) -> str | None:
    if isinstance(node, Node):
        return node.id
    return node.get("id")


def _get_node_classes(node: Node | dict[str, Any]) -> tuple[str, ...]:
    if isinstance(node, Node):
        classes = node.class_
    else:
        classes = node.get("class", "")
    if not classes:
        return ()
    return tuple(cls for cls in str(classes).split() if cls)


def _matches_simple(
    simple: SimpleSelector,
    node: Node | dict[str, Any],
    sibling_meta: tuple[int, int] | None,
) -> bool:
    if not simple.universal and simple.tag is not None:
        if _get_node_type(node) != simple.tag:
            return False
    if simple.id is not None:
        if _get_node_id(node) != simple.id:
            return False

    node_classes = set(_get_node_classes(node))
    if any(cls not in node_classes for cls in simple.classes):
        return False
    return _matches_pseudos(simple.pseudos, sibling_meta)


def _matches_pseudos(
    pseudos: tuple[PseudoClass, ...], sibling_meta: tuple[int, int] | None
) -> bool:
    if not pseudos:
        return True
    if sibling_meta is None:
        return False
    position, total = sibling_meta
    for pseudo in pseudos:
        if pseudo.name == "first":
            if position != 1:
                return False
        elif pseudo.name == "last":
            if position != total:
                return False
        elif pseudo.name == "only":
            if total != 1:
                return False
        elif pseudo.name == "nth":
            if pseudo.argument != position:
                return False
        else:
            return False
    return True
