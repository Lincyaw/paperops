"""Macro expansion for IR documents.

Supported features:

- defines + `$use` in-place expansion
- nested macro expansion
- variable substitution (including special `$CHILDREN`)
- circular dependency detection with clear error codes
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


class MacroExpansionError(ValueError):
    """Base error for macro-expansion failures."""

    code: str
    path: str

    def __init__(self, code: str, message: str, path: str):
        super().__init__(message)
        self.code = code
        self.path = path


class CircularMacroError(MacroExpansionError):
    """Raised for `$use` cycles."""


class UndefinedMacroError(MacroExpansionError):
    """Raised when an invocation references unknown define."""


class UnresolvedMacroVarError(MacroExpansionError):
    """Raised when a `$var` placeholder is unresolved in strict mode."""


def _is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def _is_list(value: Any) -> bool:
    return isinstance(value, list)


def _normalize_defines(raw_defines: Any) -> dict[str, dict[str, Any]]:
    if not _is_dict(raw_defines):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for name, template in raw_defines.items():
        if isinstance(name, str) and _is_dict(template):
            normalized[name] = deepcopy(template)
    return normalized


def _strip_comment_keys(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if not str(key).startswith("_")}


def _is_var_placeholder(value: str) -> str | None:
    if not value.startswith("$") or len(value) <= 1:
        return None
    return value[1:]


class _MacroExpander:
    def __init__(self, defines: dict[str, dict[str, Any]], *, strict: bool):
        self._defines = defines
        self._strict = strict
        self._stack: list[str] = []
        self._expanded_cache: dict[str, dict[str, Any]] = {}

    def expand_document(self, raw_document: Mapping[str, Any]) -> dict[str, Any]:
        if not _is_dict(raw_document):
            raise TypeError("document must be a mapping")

        payload = _strip_comment_keys(dict(raw_document))
        defines = _normalize_defines(payload.get("defines"))
        self._defines = defines
        self._expanded_cache = {}
        expanded_defines = self._expand_all_defines()

        slides = payload.get("slides")
        if not _is_list(slides):
            raise TypeError("document.slides must be a list")

        return {
            **payload,
            "defines": expanded_defines if expanded_defines else payload.get("defines"),
            "slides": [
                self._expand_node(item, path=f"slides[{index}]", defines=expanded_defines)
                for index, item in enumerate(slides)
            ],
        }

    def _expand_all_defines(self) -> dict[str, dict[str, Any]]:
        expanded: dict[str, dict[str, Any]] = {}
        for name in self._defines:
            expanded[name] = self._expand_defined_node(name)
        return expanded

    def _expand_defined_node(self, name: str) -> dict[str, Any]:
        if name in self._expanded_cache:
            return self._expanded_cache[name]
        if name in self._stack:
            chain = " -> ".join(self._stack + [name])
            raise CircularMacroError("CIRCULAR_MACRO", f"Circular macro reference: {chain}", name)

        raw_node = self._defines.get(name)
        if raw_node is None:
            raise UndefinedMacroError("UNDEFINED_MACRO", f"Undefined macro {name!r}", name)

        self._stack.append(name)
        try:
            expanded_node = self._expand_node(
                deepcopy(raw_node),
                path=f"defines.{name}",
                defines={**self._expanded_cache, **self._defines},
            )
            if not _is_dict(expanded_node):
                raise MacroExpansionError(
                    "INVALID_MACRO",
                    f"Macro definition {name!r} must expand to a node mapping",
                    f"defines.{name}",
                )
            cache_value = expanded_node
            self._expanded_cache[name] = cache_value
        finally:
            self._stack.pop()

        return cache_value

    def _expand_node(
        self,
        raw_node: Any,
        *,
        path: str,
        defines: dict[str, dict[str, Any]],
    ) -> Any:
        if _is_list(raw_node):
            return [
                self._expand_node(item, path=f"{path}[{index}]", defines=defines)
                for index, item in enumerate(raw_node)
            ]

        if not _is_dict(raw_node):
            return deepcopy(raw_node)

        node = _strip_comment_keys(dict(raw_node))
        if "$use" in node:
            return self._expand_macro(node, path=path, defines=defines)

        return {
            key: self._expand_node(value, path=f"{path}.{key}", defines=defines)
            for key, value in node.items()
        }

    def _expand_macro(
        self,
        node: dict[str, Any],
        *,
        path: str,
        defines: dict[str, dict[str, Any]],
    ) -> Any:
        raw_name = node.get("$use")
        if not isinstance(raw_name, str):
            raise MacroExpansionError("INVALID_USE", f"`$use` must be a string at {path}", path)

        if raw_name not in defines:
            raise UndefinedMacroError("UNDEFINED_MACRO", f"Undefined macro {raw_name!r} at {path}", path)

        if raw_name in self._stack:
            chain = " -> ".join(self._stack + [raw_name])
            raise CircularMacroError("CIRCULAR_MACRO", f"Circular macro reference: {chain}", path)

        children = node.get("children", [])
        if children is None:
            children = []
        if not _is_list(children):
            raise TypeError(f"`children` must be a list at {path}")

        variables = {
            key: value
            for key, value in node.items()
            if key != "$use" and not str(key).startswith("_") and key != "children"
        }
        self._stack.append(raw_name)
        try:
            expanded_template = self._expand_node(
                deepcopy(defines[raw_name]),
                path=f"{path}.$use({raw_name})",
                defines=defines,
            )
        finally:
            self._stack.pop()

        return self._substitute(
            expanded_template,
            variables=variables,
            children=children,
            path=path,
        )

    def _substitute(
        self,
        value: Any,
        *,
        variables: dict[str, Any],
        children: list[Any],
        path: str,
    ) -> Any:
        if isinstance(value, str):
            var_name = _is_var_placeholder(value)
            if var_name is None:
                return value
            if var_name == "CHILDREN":
                return self._expand_node(
                    deepcopy(children),
                    path=f"{path}.$children",
                    defines=self._defines,
                )
            if var_name in variables:
                return deepcopy(variables[var_name])
            if self._strict:
                raise UnresolvedMacroVarError(
                    "UNRESOLVED_MACRO_VAR",
                    f"Unresolved macro variable ${var_name} at {path}",
                    path,
                )
            return value

        if _is_list(value):
            items: list[Any] = []
            for index, item in enumerate(value):
                substituted = self._substitute(
                    item,
                    variables=variables,
                    children=children,
                    path=f"{path}[{index}]",
                )
                if _is_list(substituted) and _is_var_placeholder(item) == "CHILDREN":
                    items.extend(substituted)
                else:
                    items.append(substituted)
            return items

        if _is_dict(value):
            return {
                key: self._substitute(
                    item,
                    variables=variables,
                    children=children,
                    path=f"{path}.{key}",
                )
                for key, item in value.items()
            }

        return deepcopy(value)



def expand_document(raw_document: Mapping[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """Expand `$use` in a raw IR document payload."""
    defines = _normalize_defines(raw_document.get("defines") if _is_dict(raw_document) else {})
    return _MacroExpander(defines, strict=strict).expand_document(raw_document)


def expand_document_node(
    node: Any,
    defines: Mapping[str, Any] | None,
    *,
    strict: bool = False,
    path: str = "root",
) -> Any:
    """Expand a standalone node with explicit defines."""
    defines_map = _normalize_defines(defines)
    return _MacroExpander(defines_map, strict=strict)._expand_node(
        node,
        path=path,
        defines=defines_map,
    )
