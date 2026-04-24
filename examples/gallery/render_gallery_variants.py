"""Build the gallery deck under six preset sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paperops.slides.build import render_json


ROOT = Path(__file__).resolve().parent
CONTENT = ROOT / "gallery_content.json"
SHEETS = ("minimal", "academic", "seminar", "keynote", "whitepaper", "pitch")


def _load_content() -> dict:
    return json.loads(CONTENT.read_text(encoding="utf-8"))


def build_gallery(output_dir: Path) -> list[Path]:
    payload = _load_content()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []

    for sheet in SHEETS:
        payload["sheet"] = sheet
        out = output_dir / f"gallery_{sheet}.pptx"
        results.append(render_json(payload, out=out))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Render gallery variants")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory for generated .pptx files",
    )
    args = parser.parse_args()
    build_gallery(args.output_dir)


if __name__ == "__main__":
    main()
