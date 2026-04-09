"""Import wrapper for the self-intro deck located in examples/self-intro/build.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[1] / "self-intro" / "build.py"
_SPEC = importlib.util.spec_from_file_location("examples.self_intro._legacy_build", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load self-intro deck from {_SOURCE}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name, _value in vars(_MODULE).items():
    if _name.startswith("__") and _name not in {"__doc__", "__all__"}:
        continue
    globals()[_name] = _value
