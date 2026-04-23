"""Component registry and declaration helper decorators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ComponentDefinition:
    """Metadata describing one registered component."""

    type: str
    props_schema: dict[str, Any] = field(default_factory=dict)
    default_classes: list[str] = field(default_factory=list)
    expand: Callable[[dict[str, Any]], Any] | None = None


class ComponentRegistry:
    """Simple, strict component registry."""

    def __init__(self) -> None:
        self._definitions: dict[str, ComponentDefinition] = {}

    def register(
        self, component_type: str, definition: ComponentDefinition
    ) -> ComponentDefinition:
        if component_type in self._definitions:
            raise ValueError(f"Component '{component_type}' already registered")
        self._definitions[component_type] = definition
        return definition

    def get(self, component_type: str) -> ComponentDefinition:
        return self._definitions[component_type]

    def has(self, component_type: str) -> bool:
        return component_type in self._definitions

    def as_dict(self) -> dict[str, ComponentDefinition]:
        return dict(self._definitions)


def register_component(
    type_name: str,
    *,
    props_schema: dict[str, Any] | None = None,
    default_classes: list[str] | None = None,
    expand: Callable[[dict[str, Any]], Any] | None = None,
):
    """Decorator to register components.

    Usage:
        @register_component("kpi", props_schema={"value": {"type": "number"}})
        class Kpi: ...
    """

    if not type_name or not isinstance(type_name, str):
        raise TypeError("type_name must be a non-empty string")

    def decorator(target: Any):
        target_schema = props_schema
        if target_schema is None:
            target_schema = getattr(target, "props_schema", {})
        target_classes = (
            default_classes
            if default_classes is not None
            else list(getattr(target, "default_classes", []))
        )
        target_expand = (
            expand if expand is not None else getattr(target, "expand", None)
        )

        definition = ComponentDefinition(
            type=type_name,
            props_schema=target_schema or {},
            default_classes=target_classes or [],
            expand=target_expand,
        )
        registry.register(type_name, definition)
        return target

    return decorator


registry = ComponentRegistry()

__all__ = ["ComponentDefinition", "ComponentRegistry", "register_component", "registry"]
