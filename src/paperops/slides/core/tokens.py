"""Token parsing and resolution helpers for the slide DSL foundation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


class UnknownTokenError(ValueError):
    """Raised when a token name is not defined for a token category."""

    def __init__(self, category: str, token: Any, available: list[str]):
        self.category = category
        self.token = token
        self.available = available
        super().__init__(f"Unknown {category} token {token!r}. Available: {available}")


class UnknownTokenCategoryError(ValueError):
    """Raised when an unsupported token category is requested."""


@dataclass(frozen=True)
class ShadowSpec:
    """Resolved shadow token value."""

    dx: float
    dy: float
    blur: float
    color: str
    opacity: float = 1.0

    def as_dict(self) -> dict[str, float | str]:
        return {
            "dx": self.dx,
            "dy": self.dy,
            "blur": self.blur,
            "color": self.color,
            "opacity": self.opacity,
        }


class _UnitConverter:
    _REGEX = re.compile(
        r"^\s*(?P<number>[-+]?(?:\d+(?:\.\d+)?|\.\d+))(?:\s*(?P<unit>[a-zA-Z]+))?\s*$"
    )

    @staticmethod
    def parse(value: str) -> tuple[float, str | None]:
        match = _UnitConverter._REGEX.match(value)
        if not match:
            raise ValueError(f"Invalid numeric token value: {value!r}")
        number = float(match.group("number"))
        unit = match.group("unit")
        return number, unit.lower() if unit else None


def _as_float(value: Any, category: str, available: list[str] | None = None) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Invalid {category} value: {value!r}")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    if available is not None:
        raise UnknownTokenError(category, value, available)
    raise TypeError(f"Invalid {category} value: {value!r}")


def _ensure_hex(value: Any, category: str, available: list[str] | None = None) -> str:
    if isinstance(value, tuple) and len(value) == 3:
        r, g, b = value
        return f"#{int(r):02X}{int(g):02X}{int(b):02X}"
    if isinstance(value, str):
        if value.startswith("#"):
            if len(value) == 7 and all(
                ch in "0123456789abcdefABCDEF" for ch in value[1:]
            ):
                return value
            if value.lower() == "inherit" or value.lower() == "auto":
                return value.lower()
        if value in available or available is not None:
            if available is not None and value in available:
                return value
            raise UnknownTokenError(category, value, available)
    raise TypeError(f"Invalid {category} value: {value!r}")


def _resolve_color(theme, value: Any):
    available = sorted(theme.colors.keys())
    if value == "inherit" or value == "auto":
        return value
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return _ensure_hex(tuple(value), "colors", None)
    if isinstance(value, str):
        if value.startswith("#"):
            if len(value) == 7 and all(
                ch in "0123456789abcdefABCDEF" for ch in value[1:]
            ):
                return value
            raise ValueError(f"Invalid color value: {value!r}")
        if value in theme.colors:
            return theme.colors[value]
    if isinstance(value, int | float):
        raise TypeError(
            "Numeric colors are unsupported. Use a hex color (e.g. '#3B6B9D') "
            "or a color token name."
        )
    raise UnknownTokenError("colors", value, available)


def _resolve_fonts(theme, value: Any):
    available = sorted(theme.fonts.keys())
    if value == "inherit" or value == "auto":
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value in theme.fonts:
            return float(theme.fonts[value])
        value = value.strip()
        if (
            value.endswith("pt")
            or value.endswith("in")
            or value.endswith("px")
            or value.endswith("em")
            or value.endswith("rem")
        ):
            try:
                number, unit = _UnitConverter.parse(value)
            except ValueError as exc:
                raise UnknownTokenError("fonts", value, available) from exc
            if unit == "pt":
                return float(number)
            if unit == "px":
                return float(number) / 96 * 72
            if unit in {"in", "inch", "inches"}:
                return float(number) * 72
            if unit == "em":
                base_font = float(theme.fonts.get("body", 16))
                return float(number) * base_font
            if unit == "rem":
                base_font = float(theme.fonts.get("body", 16))
                return float(number) * base_font
            raise ValueError(f"Unsupported font size unit: {value!r}")
        try:
            return float(value)
        except ValueError:
            raise UnknownTokenError("fonts", value, available)
    raise UnknownTokenError("fonts", value, available)


def _resolve_spacing(theme, value: Any):
    available = sorted(theme.spacing.keys())
    if value == "inherit" or value == "auto":
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value in theme.spacing:
            return float(theme.spacing[value])
        value = value.strip()
        try:
            number, unit = _UnitConverter.parse(value)
        except ValueError as exc:
            raise UnknownTokenError("spacing", value, available) from exc
        if not unit:
            return float(number)
        if unit == "in":
            return float(number)
        if unit == "pt":
            return float(number) / 72.0
        if unit == "px":
            return float(number) / 96.0
        if unit == "cm":
            return float(number) / 2.54
        if unit == "mm":
            return float(number) / 25.4
        raise ValueError(f"Unsupported spacing unit: {value!r}")
    raise UnknownTokenError("spacing", value, available)


def _resolve_radius(theme, value: Any):
    available = sorted(theme.radius.keys())
    if value == "inherit" or value == "auto":
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value in theme.radius:
            return float(theme.radius[value])
        value = value.strip()
        try:
            number, unit = _UnitConverter.parse(value)
        except ValueError as exc:
            raise UnknownTokenError("radius", value, available) from exc
        if not unit:
            return float(number)
        if unit == "in":
            return float(number)
        if unit == "px":
            return float(number) / 96.0
        if unit == "pt":
            return float(number) / 72.0
        raise ValueError(f"Unsupported radius unit: {value!r}")
    raise UnknownTokenError("radius", value, available)


def _resolve_duration(theme, value: Any):
    available = sorted(theme.duration.keys())
    if value == "inherit" or value == "auto":
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value in theme.duration:
            return float(theme.duration[value])
        value = value.strip()
        try:
            number, unit = _UnitConverter.parse(value)
        except ValueError as exc:
            raise UnknownTokenError("duration", value, available) from exc
        if not unit:
            return float(number)
        if unit == "s":
            return float(number)
        if unit == "ms":
            return float(number) / 1000
        raise ValueError(f"Unsupported duration unit: {value!r}")
    raise UnknownTokenError("duration", value, available)


def _resolve_shadow(theme, value: Any):
    available = sorted(theme.shadow.keys())
    if value == "inherit" or value == "auto":
        return value
    if isinstance(value, str) and value in theme.shadow:
        spec = theme.shadow[value]
    elif isinstance(value, dict):
        if "dx" not in value or "dy" not in value:
            raise ValueError(
                "Shadow token dictionary must contain at least 'dx' and 'dy'"
            )
        spec = {
            "dx": value["dx"],
            "dy": value["dy"],
            "blur": value["blur"],
            "color": value.get("color", "#000000"),
            "opacity": value.get("opacity", 1.0),
        }
    elif isinstance(value, (tuple, list)):
        if len(value) != 5:
            raise ValueError(
                "Shadow token sequence must be [dx, dy, blur, color, opacity]"
            )
        spec = {
            "dx": value[0],
            "dy": value[1],
            "blur": value[2],
            "color": value[3],
            "opacity": value[4],
        }
    elif value in theme.shadow:
        spec = theme.shadow[value]
    else:
        raise UnknownTokenError("shadow", value, available)

    if isinstance(spec, ShadowSpec):
        return spec
    return ShadowSpec(
        dx=float(spec["dx"]),
        dy=float(spec["dy"]),
        blur=float(spec["blur"]),
        color=(
            _resolve_color(theme, spec["color"])
            if isinstance(spec, dict)
            else _ensure_hex(spec["color"], "shadow", None)
        ),
        opacity=float(spec.get("opacity", 1.0)),
    )


def _resolve_density(theme, value: Any):
    allowed = {"compact", "normal", "airy"}
    if value == "inherit" or value == "auto":
        return value
    if not isinstance(value, str):
        raise TypeError(f"Invalid density value: {value!r}")
    if value not in allowed:
        raise UnknownTokenError("density", value, sorted(allowed))
    return value


def _resolve_baseline(theme, value: Any):
    available = ["baseline"]
    if value == "inherit" or value == "auto":
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            number, unit = _UnitConverter.parse(value)
        except ValueError:
            raise UnknownTokenError("baseline", value, available)
        if unit is None:
            return float(number)
        if unit == "in":
            return float(number)
        if unit == "pt":
            return float(number) / 72.0
        raise ValueError(f"Unsupported baseline unit: {value!r}")
    raise UnknownTokenError("baseline", value, available)


def resolve_token(theme, category: str, value: Any):
    """Resolve a style token with a single entrypoint.

    Supported `category` values:
      - "colors"
      - "fonts"
      - "spacing"
      - "radius"
      - "shadow"
      - "duration"
      - "density"
      - "baseline"

    Supported value types:
      - token name (resolved from the theme map)
      - number (returned as float)
      - unit strings like "0.5in", "12pt"
      - hex colors for color category
      - "inherit" / "auto" passthrough
    """

    match category:
        case "colors":
            return _resolve_color(theme, value)
        case "fonts":
            return _resolve_fonts(theme, value)
        case "spacing":
            return _resolve_spacing(theme, value)
        case "radius":
            return _resolve_radius(theme, value)
        case "shadow":
            return _resolve_shadow(theme, value)
        case "duration":
            return _resolve_duration(theme, value)
        case "density":
            return _resolve_density(theme, value)
        case "baseline":
            return _resolve_baseline(theme, value)
        case _:
            raise UnknownTokenCategoryError(f"Unknown token category: {category!r}")


def resolve(theme, category: str, value: Any):
    """Backward-compatible alias for resolve_token."""
    return resolve_token(theme, category, value)
