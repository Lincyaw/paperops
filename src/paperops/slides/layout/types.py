"""Typed layout primitives shared across the SlideCraft layout engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from paperops.slides.core.constants import Region


@dataclass(frozen=True)
class Constraints:
    """Measurement constraints in slide units (inches)."""

    min_width: float = 0.0
    max_width: float | None = None
    min_height: float = 0.0
    max_height: float | None = None

    def clamp_width(self, value: float) -> float:
        result = max(value, self.min_width)
        if self.max_width is not None:
            result = min(result, self.max_width)
        return result

    def clamp_height(self, value: float) -> float:
        result = max(value, self.min_height)
        if self.max_height is not None:
            result = min(result, self.max_height)
        return result


@dataclass(frozen=True)
class IntrinsicSize:
    """Intrinsic size contract returned by layout participants."""

    min_width: float
    preferred_width: float
    min_height: float
    preferred_height: float
    baseline: float | None = None
    breakpoints: tuple[float, ...] = ()

    def clamp(self, constraints: Constraints) -> "IntrinsicSize":
        return IntrinsicSize(
            min_width=constraints.clamp_width(self.min_width),
            preferred_width=constraints.clamp_width(self.preferred_width),
            min_height=constraints.clamp_height(self.min_height),
            preferred_height=constraints.clamp_height(self.preferred_height),
            baseline=self.baseline,
            breakpoints=self.breakpoints,
        )


@dataclass(frozen=True)
class TrackSpec:
    """Grid track definition."""

    kind: str
    value: float = 0.0


def auto() -> TrackSpec:
    return TrackSpec("auto", 0.0)


def fixed(value: float) -> TrackSpec:
    return TrackSpec("fixed", float(value))


def fr(value: float = 1.0) -> TrackSpec:
    return TrackSpec("fr", float(value))


@dataclass
class LayoutIssue:
    """Stable machine-friendly issue emitted by the layout/review pipeline."""

    code: str
    message: str
    severity: str = "warning"
    source: str = "layout"
    node_path: str | None = None
    slide: int | None = None
    region: Region | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "type": self.code,
            "message": self.message,
            "detail": self.message,
            "severity": self.severity,
            "source": self.source,
        }
        if self.node_path is not None:
            payload["node_path"] = self.node_path
        if self.slide is not None:
            payload["slide"] = self.slide
        if self.region is not None:
            payload["region"] = self.region
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload
