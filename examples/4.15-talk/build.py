from __future__ import annotations

from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("talk_415_presentation", HERE / "presentation.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


if __name__ == "__main__":
    module.build_presentation(output_path=HERE / "talk_4_15.pptx", render_preview=False)
