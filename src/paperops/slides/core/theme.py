"""Theme system — single source of truth for all visual properties."""

from __future__ import annotations
from dataclasses import dataclass, field
import functools
from copy import deepcopy


_REQUIRED_COLORS = {
    "primary", "secondary", "accent",
    "positive", "negative", "highlight", "warning",
    "text", "text_mid", "text_light",
    "bg", "bg_alt", "bg_accent", "border",
}

_REQUIRED_FONTS = {"title", "subtitle", "heading", "body", "caption", "small"}


@dataclass
class Theme:
    name: str
    colors: dict[str, str] = field(default_factory=dict)
    fonts: dict[str, float] = field(default_factory=dict)  # name → point size
    font_family: str = "Calibri"
    font_mono: str = "Consolas"

    def __post_init__(self):
        missing_colors = _REQUIRED_COLORS - set(self.colors.keys())
        if missing_colors:
            raise ValueError(f"Theme '{self.name}' missing required colors: {missing_colors}")
        missing_fonts = _REQUIRED_FONTS - set(self.fonts.keys())
        if missing_fonts:
            raise ValueError(f"Theme '{self.name}' missing required font sizes: {missing_fonts}")

    def override(self, **kwargs) -> Theme:
        """Create a new theme with selective overrides.

        Accepts: name, colors (merged), fonts (merged), font_family, font_mono.
        """
        new = deepcopy(self)
        for key, value in kwargs.items():
            if key == "colors":
                new.colors.update(value)
            elif key == "fonts":
                new.fonts.update(value)
            elif hasattr(new, key):
                setattr(new, key, value)
            else:
                raise ValueError(f"Unknown theme attribute: {key}")
        new.__post_init__()
        return new

    def resolve_color(self, color) -> str:
        """Resolve a color value to hex string.

        Accepts:
          - str semantic name ("primary") → lookup in theme
          - str hex ("#3B6B9D") → pass through
          - tuple (r, g, b) → convert to hex
        """
        if isinstance(color, tuple):
            r, g, b = color
            return f"#{r:02X}{g:02X}{b:02X}"
        if isinstance(color, str):
            if color.startswith("#"):
                return color
            if color in self.colors:
                return self.colors[color]
            raise ValueError(f"Unknown color name: '{color}'. Available: {sorted(self.colors.keys())}")
        raise TypeError(f"Invalid color type: {type(color)}")

    def resolve_font_size(self, size) -> float:
        """Resolve a font size to points (float).

        Accepts:
          - str semantic name ("body") → lookup in theme
          - int/float → use directly as pt
        """
        if isinstance(size, str):
            if size in self.fonts:
                return self.fonts[size]
            raise ValueError(f"Unknown font size name: '{size}'. Available: {sorted(self.fonts.keys())}")
        if isinstance(size, (int, float)):
            return float(size)
        raise TypeError(f"Invalid font size type: {type(size)}")


class _Themes:
    """Namespace for built-in themes. Instances are cached."""

    @functools.cached_property
    def executive(self) -> Theme:
        return Theme(
            name="executive",
            colors={
                "primary":    "#16324F",
                "secondary":  "#356B6F",
                "accent":     "#C8792A",
                "positive":   "#2E6B57",
                "negative":   "#B44D3E",
                "highlight":  "#4768A8",
                "warning":    "#B6923E",
                "text":       "#16212B",
                "text_mid":   "#51606F",
                "text_light": "#7C8A98",
                "bg":         "#F7F6F2",
                "bg_alt":     "#ECEAE3",
                "bg_accent":  "#E2E8EE",
                "border":     "#C5CBD3",
            },
            fonts={
                "title": 30,
                "subtitle": 20,
                "heading": 22,
                "body": 16,
                "caption": 12,
                "small": 10,
            },
            font_family="Liberation Sans",
            font_mono="Liberation Mono",
        )

    @functools.cached_property
    def academic_seminar(self) -> Theme:
        return Theme(
            name="academic_seminar",
            colors={
                "primary":    "#1F3A5F",
                "secondary":  "#4B627D",
                "accent":     "#C47A2C",
                "positive":   "#2F6B57",
                "negative":   "#B34E48",
                "highlight":  "#3F5F90",
                "warning":    "#A8842C",
                "text":       "#18232F",
                "text_mid":   "#546170",
                "text_light": "#7F8A96",
                "bg":         "#FFFFFF",
                "bg_alt":     "#F4F6F8",
                "bg_accent":  "#EAF0F5",
                "border":     "#D2D8DF",
            },
            fonts={
                "title": 26,
                "subtitle": 18,
                "heading": 18,
                "body": 16,
                "caption": 12,
                "small": 10,
            },
            font_family="Liberation Sans",
            font_mono="Liberation Mono",
        )

    @functools.cached_property
    def professional(self) -> Theme:
        return Theme(
            name="professional",
            colors={
                "primary":    "#3B6B9D",
                "secondary":  "#4A8B7F",
                "accent":     "#C27C3E",
                "positive":   "#4A7C5F",
                "negative":   "#B85450",
                "highlight":  "#6B5B8D",
                "warning":    "#C4A34D",
                "text":       "#1E293B",
                "text_mid":   "#64748B",
                "text_light": "#94A3B8",
                "bg":         "#FFFFFF",
                "bg_alt":     "#F7F8FA",
                "bg_accent":  "#EEF0F5",
                "border":     "#C8CED8",
            },
            fonts={
                "title": 32, "subtitle": 24, "heading": 20,
                "body": 18, "caption": 14, "small": 11,
            },
            font_family="Calibri",
            font_mono="Consolas",
        )

    @functools.cached_property
    def minimal(self) -> Theme:
        return Theme(
            name="minimal",
            colors={
                "primary":    "#2D3748",
                "secondary":  "#4A5568",
                "accent":     "#3182CE",
                "positive":   "#38A169",
                "negative":   "#E53E3E",
                "highlight":  "#805AD5",
                "warning":    "#D69E2E",
                "text":       "#1A202C",
                "text_mid":   "#718096",
                "text_light": "#A0AEC0",
                "bg":         "#FFFFFF",
                "bg_alt":     "#F7FAFC",
                "bg_accent":  "#EDF2F7",
                "border":     "#E2E8F0",
            },
            fonts={
                "title": 32, "subtitle": 24, "heading": 20,
                "body": 18, "caption": 14, "small": 11,
            },
            font_family="Calibri",
            font_mono="Consolas",
        )

    @functools.cached_property
    def academic(self) -> Theme:
        return Theme(
            name="academic",
            colors={
                "primary":    "#8B4513",
                "secondary":  "#2E6B4F",
                "accent":     "#4A6FA5",
                "positive":   "#2E6B4F",
                "negative":   "#A0522D",
                "highlight":  "#6B4C8D",
                "warning":    "#B8860B",
                "text":       "#2C2C2C",
                "text_mid":   "#5A5A5A",
                "text_light": "#8A8A8A",
                "bg":         "#FFFEF8",
                "bg_alt":     "#F5F3ED",
                "bg_accent":  "#EDE8DC",
                "border":     "#C8C0B0",
            },
            fonts={
                "title": 32, "subtitle": 24, "heading": 20,
                "body": 18, "caption": 14, "small": 11,
            },
            font_family="Georgia",
            font_mono="Consolas",
        )


themes = _Themes()
