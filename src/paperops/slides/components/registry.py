"""Component registry and validation helpers.

The registry is the contract between parsed IR nodes and the semantic expansion
pipeline.  Every registered component declares:

- ``props_schema``: what props are accepted and which are required
- ``default_classes``: classes injected ahead of user-specified classes
- optional ``expand`` callback to expand semantic components into IR subtrees
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from paperops.slides.ir.node import Node


@dataclass(frozen=True)
class ComponentDefinition:
    """Metadata for one component in the registry."""

    type: str
    props_schema: dict[str, Any] = None
    default_classes: list[str] | tuple[str, ...] = None
    expand: Callable[..., Any] | None = None


@dataclass(frozen=True)
class ComponentError(ValueError):
    """Structured validation error produced by component checks."""

    code: str
    path: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "args",
            (f"[{self.code}] {self.path}: {self.message}",),
        )


@dataclass
class ComponentRegistry:
    """Global registry for slide components."""

    _definitions: dict[str, ComponentDefinition]

    def __init__(self) -> None:
        self._definitions = {}

    def clear(self) -> None:
        self._definitions.clear()

    def snapshot(self) -> dict[str, ComponentDefinition]:
        return dict(self._definitions)

    def register(self, name: str, definition: ComponentDefinition) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("component type must be a non-empty string")
        if not isinstance(definition, ComponentDefinition):
            raise TypeError("definition must be ComponentDefinition")
        if definition.type != name:
            raise ValueError("definition type must match registry key")
        if name in self._definitions:
            raise ValueError(f"Component {name!r} already registered")
        self._definitions[name] = definition

    def get(self, name: str) -> ComponentDefinition:
        return self._definitions[name]

    def has(self, name: str) -> bool:
        return name in self._definitions

    def as_dict(self) -> dict[str, ComponentDefinition]:
        return dict(self._definitions)


def _normalise_schema(raw_schema: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Return ``(properties, required_keys)`` from either schema format.

    Supported input examples:
    - ``{"properties": {...}, "required": ["label"]}``
    - ``{"label": {"type": "string", "required": True}}``
    """

    if not raw_schema:
        return {}, set()

    if "properties" in raw_schema and isinstance(raw_schema["properties"], dict):
        properties = {
            key: dict(value) if isinstance(value, dict) else {"type": value}
            for key, value in raw_schema["properties"].items()
            if isinstance(key, str)
        }
        required = set(raw_schema.get("required", []) or [])
        return properties, set(filter(None, (str(key) for key in required)))

    properties: dict[str, dict[str, Any]] = {}
    required: set[str] = set()
    for key, rule in raw_schema.items():
        if not isinstance(key, str):
            continue
        if key in {"properties", "required"}:
            continue
        if isinstance(rule, Mapping):
            properties[key] = dict(rule)
            if rule.get("required") is True:
                required.add(key)
        else:
            properties[key] = {"type": rule}

    for required_name in raw_schema.get("required", []) if isinstance(raw_schema.get("required"), list) else []:
        if isinstance(required_name, str):
            required.add(required_name)
            properties.setdefault(required_name, {"type": "string"})

    return properties, required


def _validate_scalar(value: Any, expected: str, path: str) -> None:
    if expected in {"str", "string"}:
        if not isinstance(value, str):
            raise ComponentError("INVALID_PROP_VALUE", path, f"expected string, got {type(value).__name__}")
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ComponentError("INVALID_PROP_VALUE", path, f"expected number, got {type(value).__name__}")
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ComponentError("INVALID_PROP_VALUE", path, f"expected integer, got {type(value).__name__}")
    elif expected in {"boolean", "bool"}:
        if not isinstance(value, bool):
            raise ComponentError("INVALID_PROP_VALUE", path, f"expected boolean, got {type(value).__name__}")
    elif expected == "array":
        if not isinstance(value, list):
            raise ComponentError("INVALID_PROP_VALUE", path, f"expected array, got {type(value).__name__}")
    elif expected == "object":
        if not isinstance(value, dict):
            raise ComponentError("INVALID_PROP_VALUE", path, f"expected object, got {type(value).__name__}")


def _validate_one(value: Any, rule: dict[str, Any], path: str) -> Any:
    expected = rule.get("type")
    if expected is None:
        return value

    if isinstance(expected, (list, tuple)):
        for index, candidate in enumerate(expected):
            try:
                _validate_scalar(value, str(candidate), f"{path}[{index}]")
                break
            except ComponentError:
                continue
        else:
            allowed = ", ".join(str(item) for item in expected)
            raise ComponentError("INVALID_PROP_VALUE", path, f"expected one of [{allowed}], got {type(value).__name__}")

    else:
        if isinstance(expected, list):
            expected = ",".join(str(item) for item in expected)
        _validate_scalar(value, str(expected), path)

    if "enum" in rule and value not in rule.get("enum"):
        raise ComponentError(
            "INVALID_PROP_VALUE",
            path,
            f"invalid value {value!r}; expected one of {list(rule.get('enum'))}",
        )

    return value


def _validate_node_props(
    definition: ComponentDefinition,
    node: Node,
    *,
    path: str,
    strict: bool,
) -> dict[str, Any]:
    """Validate ``props`` for a registered component and return merged defaults."""
    raw_props = dict(node.props or {})
    schema = dict(definition.props_schema or {})
    properties, required = _normalise_schema(schema)

    merged = dict(raw_props)
    for key in required:
        if key not in merged and "default" not in properties.get(key, {}):
            raise ComponentError(
                "MISSING_REQUIRED_PROP",
                f"{path}.props",
                f"Missing required prop {key!r}",
            )

    for key in properties:
        if "default" in properties[key] and key not in merged:
            merged[key] = properties[key]["default"]

    for key, value in list(merged.items()):
        if key not in properties:
            raise ComponentError(
                "UNKNOWN_PROP",
                f"{path}.props",
                f"Unknown prop {key!r}",
            )
            continue
        rule = properties[key]
        merged[key] = _validate_one(value, rule, f"{path}.props.{key}")

    return merged


def _merge_classes(default_classes: list[str] | tuple[str, ...] | None, user_class: str | None) -> str:
    defaults = list(default_classes or [])
    result: list[str] = []
    for token in [*(defaults), *(user_class.split() if isinstance(user_class, str) else [])]:
        token = token.strip()
        if not token or token in result:
            continue
        result.append(token)
    return " ".join(result)


def _build_expanded_node(
    value: Any,
    path: str,
) -> Node:
    if isinstance(value, Node):
        return value
    if isinstance(value, dict):
        return Node.from_dict(value)
    raise TypeError(f"{path}: expected node mapping, got {type(value).__name__}")


def expand_node(
    node: Node,
    *,
    strict: bool = False,
    path: str = "root",
) -> Node:
    """Validate one node, apply defaults, and expand semantic component nodes."""

    if not isinstance(node, Node):
        return node

    children = node.children or []
    expanded_children = [
        expand_node(child, strict=strict, path=f"{path}.children[{index}]")
        if isinstance(child, Node)
        else child
        for index, child in enumerate(children)
    ]

    if not registry.has(node.type):
        if not expanded_children:
            return Node(
                type=node.type,
                class_=node.class_,
                id=node.id,
                style=node.style,
                text=node.text,
                props=node.props,
                children=None,
            )
        return Node(
            type=node.type,
            class_=node.class_,
            id=node.id,
            style=node.style,
            text=node.text,
            props=node.props,
            children=expanded_children,
        )

    definition = registry.get(node.type)
    validated_props = _validate_node_props(
        definition,
        Node(
            type=node.type,
            class_=node.class_,
            id=node.id,
            style=node.style,
            text=node.text,
            props=node.props,
            children=children,
        ),
        path=path,
        strict=strict,
    )
    merged_class = _merge_classes(definition.default_classes, node.class_)

    source_node = Node(
        type=node.type,
        class_=merged_class,
        id=node.id,
        style=node.style,
        text=node.text,
        props=validated_props,
        children=expanded_children,
    )

    if definition.expand is None:
        return source_node

    expanded_node = _call_expand(definition, validated_props, source_node)
    if expanded_node is None:
        return source_node

    base_node = _build_expanded_node(expanded_node, path=f"{path}.expanded")
    merged_default_classes = _merge_classes(definition.default_classes, source_node.class_ or "")
    merged_expanded_classes = _merge_classes(
        merged_default_classes.split() if merged_default_classes else None,
        base_node.class_,
    )

    if base_node.type == source_node.type:
        return Node(
            type=base_node.type,
            class_=merged_expanded_classes,
            id=base_node.id or node.id,
            style=base_node.style,
            text=base_node.text,
            props=base_node.props,
            children=base_node.children,
        )

    merged = Node(
        type=base_node.type,
        class_=merged_expanded_classes,
        id=base_node.id or node.id,
        style=base_node.style,
        text=base_node.text,
        props=base_node.props,
        children=base_node.children,
    )
    return expand_node(
        Node(
            type=merged.type,
            class_=merged.class_,
            id=merged.id,
            style=merged.style,
            text=merged.text,
            props=merged.props,
            children=merged.children,
        ),
        strict=strict,
        path=f"{path}.expanded",
    )


def _call_expand(definition: ComponentDefinition, props: dict[str, Any], node: Node) -> Any:
    if definition.expand is None:
        return None
    expand_fn = definition.expand
    try:
        return expand_fn(props, children=node.children, style=node.style)
    except TypeError:
        return expand_fn(props)


def expand_nodes(nodes: list[Node], *, strict: bool = False) -> list[Node]:
    return [expand_node(node, strict=strict, path=f"slides[{index}]") for index, node in enumerate(nodes)]


def register_component(
    type_name: str,
    *,
    props_schema: dict[str, Any] | None = None,
    default_classes: list[str] | tuple[str, ...] | None = None,
    expand: Callable[..., Any] | None = None,
):
    """Decorator to register one component type."""

    if not type_name or not isinstance(type_name, str):
        raise TypeError("type_name must be a non-empty string")

    def decorator(target: Any):
        target_schema = props_schema
        if target_schema is None:
            target_schema = getattr(target, "props_schema", None)
        target_classes = default_classes
        if target_classes is None:
            target_classes = getattr(target, "default_classes", None)
        target_expand = expand if expand is not None else getattr(target, "expand", None)

        registry.register(
            type_name,
            ComponentDefinition(
                type=type_name,
                props_schema=target_schema or {},
                default_classes=list(target_classes) if target_classes is not None else [],
                expand=target_expand,
            ),
        )
        return target

    return decorator


registry = ComponentRegistry()

__all__ = [
    "ComponentDefinition",
    "ComponentError",
    "ComponentRegistry",
    "expand_node",
    "expand_nodes",
    "register_component",
    "registry",
    "_merge_classes",
]
