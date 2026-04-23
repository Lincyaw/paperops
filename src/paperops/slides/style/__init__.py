"""Style resolution primitives for `paperops.slides`.

The package exposes a compact CSS-like engine used by the IR pipeline:
selectors, stylesheet ordering, cascade/inheritance, and resolved style values.
"""

from paperops.slides.style.computed import ComputedStyle
from paperops.slides.style.cascade import (
    CascadeResult,
    StyleResolutionError,
    resolve_computed_styles,
)
from paperops.slides.style.selector import ParsedSelector, parse_selector, specificity
from paperops.slides.style.stylesheet import StyleSheet

__all__ = [
    "CascadeResult",
    "ComputedStyle",
    "ParsedSelector",
    "StyleResolutionError",
    "StyleSheet",
    "parse_selector",
    "resolve_computed_styles",
    "specificity",
]
