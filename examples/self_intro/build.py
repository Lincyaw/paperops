"""Import path that delegates to the hyphenated ``examples/self-intro`` deck."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_IMPLEMENTATION_PATH = Path(__file__).resolve().parents[1] / "self-intro" / "build.py"

if not _IMPLEMENTATION_PATH.exists():
    raise FileNotFoundError(
        f"Expected self-intro example implementation at {_IMPLEMENTATION_PATH}"
    )

_spec = importlib.util.spec_from_file_location(
    "paperops_examples_self_intro_impl", _IMPLEMENTATION_PATH
)
if _spec is None or _spec.loader is None:
    raise ImportError(
        f"Cannot load self-intro example implementation from {_IMPLEMENTATION_PATH}"
    )

_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)  # type: ignore[union-attr]

IMPLEMENTED_SLIDE_COUNT = _impl.IMPLEMENTED_SLIDE_COUNT
OUTPUT_FILE = _impl.OUTPUT_FILE
SLIDE_TITLES = _impl.SLIDE_TITLES


def build_presentation(
    *, output_path: Path | None = None, render_preview: bool = False
):
    destination = output_path or Path(__file__).with_name(OUTPUT_FILE.name)
    return _impl.build_presentation(
        output_path=destination, render_preview=render_preview
    )


make_theme = _impl.make_theme

__all__ = [
    "IMPLEMENTED_SLIDE_COUNT",
    "OUTPUT_FILE",
    "SLIDE_TITLES",
    "build_presentation",
    "make_theme",
]


def main() -> None:
    build_presentation()


if __name__ == "__main__":
    main()
