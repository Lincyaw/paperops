from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from presentation import build_presentation

OUTPUT_DIR = Path(__file__).with_name("variants")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_presentation(output_path=OUTPUT_DIR / "talk_4_15_seminar.pptx", sheet="seminar")
    build_presentation(output_path=OUTPUT_DIR / "talk_4_15_keynote.pptx", sheet="keynote")


if __name__ == "__main__":
    main()
