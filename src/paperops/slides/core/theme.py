"""Token-based theme definitions for paperops.slides."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import functools
import re
from typing import Any

from paperops.slides.core.tokens import (
    ShadowSpec,
    UnknownTokenCategoryError,
    UnknownTokenError,
    resolve_token as _resolve_token,
)


def _validate_hex(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("#") or len(value) != 7:
        raise ValueError(f"Invalid hex color: {value!r}")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ValueError(f"Invalid hex color: {value!r}")
    return value


def _validate_required_tokens(
    theme_name: str, label: str, required: set[str], values: dict[str, Any]
) -> None:
    missing = required - set(values.keys())
    if missing:
        raise ValueError(
            f"Theme '{theme_name}' missing required {label}: {sorted(missing)}"
        )


_REQUIRED_COLORS = {
    "primary",
    "secondary",
    "accent",
    "positive",
    "negative",
    "highlight",
    "warning",
    "text",
    "text_mid",
    "text_light",
    "bg",
    "bg_alt",
    "bg_accent",
    "border",
}

_REQUIRED_FONTS = {"title", "subtitle", "heading", "body", "caption", "small"}
_REQUIRED_SPACING = {"xs", "sm", "md", "lg", "xl", "2xl"}
_REQUIRED_RADIUS = {"none", "sm", "md", "lg", "full"}
_REQUIRED_SHADOW = {"sm", "md", "lg"}
_REQUIRED_DURATION = {"instant", "fast", "base", "slow"}
_ALLOWED_DENSITY = {"compact", "normal", "airy"}


@dataclass
class Theme:
    """Theme token container."""

    name: str
    colors: dict[str, str] = field(default_factory=dict)
    fonts: dict[str, float] = field(default_factory=dict)
    font_family: str = "Calibri"
    font_mono: str = "Consolas"
    spacing: dict[str, float] = field(default_factory=dict)
    radius: dict[str, float] = field(default_factory=dict)
    shadow: dict[str, ShadowSpec] = field(default_factory=dict)
    duration: dict[str, float] = field(default_factory=dict)
    density: str = "normal"
    baseline: float = 0.1

    def __post_init__(self):
        _validate_required_tokens(self.name, "colors", _REQUIRED_COLORS, self.colors)
        _validate_required_tokens(self.name, "font scales", _REQUIRED_FONTS, self.fonts)
        _validate_required_tokens(self.name, "spacing", _REQUIRED_SPACING, self.spacing)
        _validate_required_tokens(self.name, "radius", _REQUIRED_RADIUS, self.radius)
        _validate_required_tokens(self.name, "shadow", _REQUIRED_SHADOW, self.shadow)
        _validate_required_tokens(
            self.name, "duration", _REQUIRED_DURATION, self.duration
        )

        if self.density not in _ALLOWED_DENSITY:
            raise ValueError(
                f"Theme '{self.name}' has invalid density '{self.density}'. "
                f"Must be one of {_ALLOWED_DENSITY}"
            )
        if self.baseline <= 0:
            raise ValueError(f"Theme '{self.name}' baseline must be positive")

        for key in _REQUIRED_COLORS:
            self.colors[key] = _validate_hex(self.colors[key])

        for key in _REQUIRED_FONTS:
            self.fonts[key] = float(self.fonts[key])

        for key in _REQUIRED_SPACING:
            self.spacing[key] = float(self.spacing[key])

        for key in _REQUIRED_RADIUS:
            self.radius[key] = float(self.radius[key])

        for key in _REQUIRED_DURATION:
            self.duration[key] = float(self.duration[key])

        for key in _REQUIRED_SHADOW:
            spec = self.shadow[key]
            if isinstance(spec, dict):
                self.shadow[key] = ShadowSpec(
                    float(spec["dx"]),
                    float(spec["dy"]),
                    float(spec["blur"]),
                    _validate_hex(spec["color"]),
                    float(spec.get("opacity", 1.0)),
                )
            elif not isinstance(spec, ShadowSpec):
                raise TypeError(
                    f"Theme '{self.name}' shadow '{key}' must be ShadowSpec or dict, got {type(spec)!r}"
                )

        try:
            self.resolve_token("spacing", "md")
            self.resolve_token("colors", "primary")
        except (UnknownTokenError, UnknownTokenCategoryError):
            # Keep error message stable and fail fast at construction time.
            raise

    def override(self, **kwargs) -> "Theme":
        """Create a new theme from `self`, applying shallow token overrides."""
        new = deepcopy(self)
        for key, value in kwargs.items():
            if key in {"colors", "fonts", "spacing", "radius", "shadow", "duration"}:
                current = getattr(new, key)
                current.update(value)
                setattr(new, key, current)
            elif key == "font_family" or key == "font_mono" or key == "density":
                setattr(new, key, value)
            elif key == "baseline":
                new.baseline = float(value)
            elif key == "name":
                new.name = value
            else:
                raise ValueError(f"Unknown theme attribute: {key}")
        new.__post_init__()
        return new

    def resolve_token(self, category: str, value):
        """Resolve a token for this theme through the unified token parser."""
        return _resolve_token(self, category, value)

    def resolve_color(self, color) -> str:
        """Backward-compatible color resolver used by existing code."""
        return self.resolve_token("colors", color)

    def resolve_font_size(self, size) -> float:
        """Backward-compatible font-size resolver used by existing code."""
        resolved = self.resolve_token("fonts", size)
        if isinstance(resolved, str):
            return float(resolved)
        return float(resolved)

    def resolve_shadow(self, key: str) -> ShadowSpec:
        return self.resolve_token("shadow", key)


class _Themes:
    """Namespace for built-in themes. Instances are cached."""

    @functools.cached_property
    def executive(self) -> Theme:
        return Theme(
            name="executive",
            colors={
                "primary": "#16324F",
                "secondary": "#356B6F",
                "accent": "#C8792A",
                "positive": "#2E6B57",
                "negative": "#B44D3E",
                "highlight": "#4768A8",
                "warning": "#B6923E",
                "text": "#16212B",
                "text_mid": "#51606F",
                "text_light": "#7C8A98",
                "bg": "#F7F6F2",
                "bg_alt": "#ECEAE3",
                "bg_accent": "#E2E8EE",
                "border": "#C5CBD3",
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
            spacing={
                "xs": 0.08,
                "sm": 0.16,
                "md": 0.32,
                "lg": 0.64,
                "xl": 1.0,
                "2xl": 1.6,
            },
            radius={
                "none": 0.0,
                "sm": 0.05,
                "md": 0.12,
                "lg": 0.24,
                "full": 99.0,
            },
            shadow={
                "sm": {
                    "dx": 0.02,
                    "dy": 0.02,
                    "blur": 0.08,
                    "color": "#C5CBD3",
                    "opacity": 0.35,
                },
                "md": {
                    "dx": 0.04,
                    "dy": 0.04,
                    "blur": 0.15,
                    "color": "#C5CBD3",
                    "opacity": 0.40,
                },
                "lg": {
                    "dx": 0.06,
                    "dy": 0.06,
                    "blur": 0.35,
                    "color": "#C5CBD3",
                    "opacity": 0.45,
                },
            },
            duration={
                "instant": 0.0,
                "fast": 0.2,
                "base": 0.4,
                "slow": 0.8,
            },
            density="normal",
            baseline=0.1,
        )

    @functools.cached_property
    def academic_seminar(self) -> Theme:
        return Theme(
            name="academic_seminar",
            colors={
                "primary": "#1F3A5F",
                "secondary": "#4B627D",
                "accent": "#C47A2C",
                "positive": "#2F6B57",
                "negative": "#B34E48",
                "highlight": "#3F5F90",
                "warning": "#A8842C",
                "text": "#18232F",
                "text_mid": "#546170",
                "text_light": "#7F8A96",
                "bg": "#FFFFFF",
                "bg_alt": "#F4F6F8",
                "bg_accent": "#EAF0F5",
                "border": "#D2D8DF",
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
            spacing={
                "xs": 0.09,
                "sm": 0.18,
                "md": 0.36,
                "lg": 0.72,
                "xl": 1.2,
                "2xl": 1.9,
            },
            radius={
                "none": 0.0,
                "sm": 0.06,
                "md": 0.16,
                "lg": 0.28,
                "full": 99.0,
            },
            shadow={
                "sm": {
                    "dx": 0.02,
                    "dy": 0.02,
                    "blur": 0.10,
                    "color": "#D2D8DF",
                    "opacity": 0.30,
                },
                "md": {
                    "dx": 0.05,
                    "dy": 0.05,
                    "blur": 0.16,
                    "color": "#D2D8DF",
                    "opacity": 0.40,
                },
                "lg": {
                    "dx": 0.08,
                    "dy": 0.08,
                    "blur": 0.34,
                    "color": "#D2D8DF",
                    "opacity": 0.50,
                },
            },
            duration={
                "instant": 0.0,
                "fast": 0.2,
                "base": 0.5,
                "slow": 0.9,
            },
            density="compact",
            baseline=0.1,
        )

    @functools.cached_property
    def professional(self) -> Theme:
        return Theme(
            name="professional",
            colors={
                "primary": "#3B6B9D",
                "secondary": "#4A8B7F",
                "accent": "#C27C3E",
                "positive": "#4A7C5F",
                "negative": "#B85450",
                "highlight": "#6B5B8D",
                "warning": "#C4A34D",
                "text": "#1E293B",
                "text_mid": "#64748B",
                "text_light": "#94A3B8",
                "bg": "#FFFFFF",
                "bg_alt": "#F7F8FA",
                "bg_accent": "#EEF0F5",
                "border": "#C8CED8",
            },
            fonts={
                "title": 32,
                "subtitle": 24,
                "heading": 20,
                "body": 18,
                "caption": 14,
                "small": 11,
            },
            font_family="Calibri",
            font_mono="Consolas",
            spacing={
                "xs": 0.08,
                "sm": 0.15,
                "md": 0.30,
                "lg": 0.60,
                "xl": 0.96,
                "2xl": 1.55,
            },
            radius={
                "none": 0.0,
                "sm": 0.04,
                "md": 0.10,
                "lg": 0.18,
                "full": 99.0,
            },
            shadow={
                "sm": {
                    "dx": 0.02,
                    "dy": 0.02,
                    "blur": 0.07,
                    "color": "#C8CED8",
                    "opacity": 0.25,
                },
                "md": {
                    "dx": 0.04,
                    "dy": 0.04,
                    "blur": 0.14,
                    "color": "#C8CED8",
                    "opacity": 0.35,
                },
                "lg": {
                    "dx": 0.07,
                    "dy": 0.07,
                    "blur": 0.30,
                    "color": "#C8CED8",
                    "opacity": 0.42,
                },
            },
            duration={
                "instant": 0.0,
                "fast": 0.15,
                "base": 0.35,
                "slow": 0.75,
            },
            density="normal",
            baseline=0.09,
        )

    @functools.cached_property
    def minimal(self) -> Theme:
        return Theme(
            name="minimal",
            colors={
                "primary": "#2D3748",
                "secondary": "#4A5568",
                "accent": "#3182CE",
                "positive": "#38A169",
                "negative": "#E53E3E",
                "highlight": "#805AD5",
                "warning": "#D69E2E",
                "text": "#1A202C",
                "text_mid": "#718096",
                "text_light": "#A0AEC0",
                "bg": "#FFFFFF",
                "bg_alt": "#F7FAFC",
                "bg_accent": "#EDF2F7",
                "border": "#E2E8F0",
            },
            fonts={
                "title": 32,
                "subtitle": 24,
                "heading": 20,
                "body": 18,
                "caption": 14,
                "small": 11,
            },
            font_family="Calibri",
            font_mono="Consolas",
            spacing={
                "xs": 0.07,
                "sm": 0.14,
                "md": 0.28,
                "lg": 0.56,
                "xl": 0.86,
                "2xl": 1.45,
            },
            radius={
                "none": 0.0,
                "sm": 0.03,
                "md": 0.08,
                "lg": 0.16,
                "full": 99.0,
            },
            shadow={
                "sm": {
                    "dx": 0.02,
                    "dy": 0.02,
                    "blur": 0.06,
                    "color": "#E2E8F0",
                    "opacity": 0.20,
                },
                "md": {
                    "dx": 0.04,
                    "dy": 0.04,
                    "blur": 0.12,
                    "color": "#E2E8F0",
                    "opacity": 0.25,
                },
                "lg": {
                    "dx": 0.06,
                    "dy": 0.06,
                    "blur": 0.24,
                    "color": "#E2E8F0",
                    "opacity": 0.3,
                },
            },
            duration={
                "instant": 0.0,
                "fast": 0.2,
                "base": 0.4,
                "slow": 0.8,
            },
            density="airy",
            baseline=0.1,
        )

    @functools.cached_property
    def academic(self) -> Theme:
        return Theme(
            name="academic",
            colors={
                "primary": "#8B4513",
                "secondary": "#2E6B4F",
                "accent": "#4A6FA5",
                "positive": "#2E6B4F",
                "negative": "#A0522D",
                "highlight": "#6B4C8D",
                "warning": "#B8860B",
                "text": "#2C2C2C",
                "text_mid": "#5A5A5A",
                "text_light": "#8A8A8A",
                "bg": "#FFFEF8",
                "bg_alt": "#F5F3ED",
                "bg_accent": "#EDE8DC",
                "border": "#C8C0B0",
            },
            fonts={
                "title": 32,
                "subtitle": 24,
                "heading": 20,
                "body": 18,
                "caption": 14,
                "small": 11,
            },
            font_family="Georgia",
            font_mono="Consolas",
            spacing={
                "xs": 0.1,
                "sm": 0.2,
                "md": 0.4,
                "lg": 0.75,
                "xl": 1.2,
                "2xl": 2.0,
            },
            radius={
                "none": 0.0,
                "sm": 0.04,
                "md": 0.1,
                "lg": 0.22,
                "full": 99.0,
            },
            shadow={
                "sm": {
                    "dx": 0.02,
                    "dy": 0.03,
                    "blur": 0.12,
                    "color": "#C8C0B0",
                    "opacity": 0.25,
                },
                "md": {
                    "dx": 0.04,
                    "dy": 0.05,
                    "blur": 0.18,
                    "color": "#C8C0B0",
                    "opacity": 0.35,
                },
                "lg": {
                    "dx": 0.07,
                    "dy": 0.09,
                    "blur": 0.28,
                    "color": "#C8C0B0",
                    "opacity": 0.45,
                },
            },
            duration={
                "instant": 0.0,
                "fast": 0.18,
                "base": 0.42,
                "slow": 0.85,
            },
            density="normal",
            baseline=0.11,
        )


themes = _Themes()
