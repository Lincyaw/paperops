from __future__ import annotations

import pytest

from paperops.slides.components.registry import (
    ComponentDefinition,
    ComponentRegistry,
    register_component,
    registry,
)


def test_register_component_and_get_definition(monkeypatch):
    # keep global registry deterministic and isolated for this test
    original_state = registry._definitions.copy()
    registry._definitions.clear()

    try:

        @register_component(
            "kpi",
            props_schema={"value": {"type": "number"}},
            default_classes=["metric", "card"],
        )
        class KpiDefinition:
            pass

        assert registry.has("kpi")
        definition = registry.get("kpi")
        assert isinstance(definition, ComponentDefinition)
        assert definition.type == "kpi"
        assert definition.props_schema == {"value": {"type": "number"}}
        assert definition.default_classes == ["metric", "card"]
    finally:
        registry._definitions.clear()
        registry._definitions.update(original_state)


def test_duplicate_component_registration_raises_error():
    local_registry = ComponentRegistry()
    local_registry.register("kpi", ComponentDefinition(type="kpi"))

    with pytest.raises(ValueError, match="already registered"):
        local_registry.register("kpi", ComponentDefinition(type="kpi"))
