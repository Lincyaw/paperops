from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SOURCE = Path(__file__).resolve().parent.parent / "self-intro" / "build.py"
_SPEC = spec_from_file_location("examples.self_intro._hyphen_build", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load builder from {_SOURCE}")

_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

OUTPUT_FILE = _MODULE.OUTPUT_FILE
IMPLEMENTED_SLIDE_COUNT = _MODULE.IMPLEMENTED_SLIDE_COUNT
build_presentation = _MODULE.build_presentation
SLIDE_TITLES = _MODULE.SLIDE_TITLES
make_theme = _MODULE.make_theme
stat_card = _MODULE.stat_card
Presentation = _MODULE.Presentation


if __name__ == "__main__":
    build_presentation()
