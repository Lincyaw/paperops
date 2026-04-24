import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import examples.self_intro.build as self_intro_build
from examples.self_intro.build import (
    IMPLEMENTED_SLIDE_COUNT,
    OUTPUT_FILE,
    SLIDE_TITLES,
    build_presentation,
    make_theme,
)
from paperops.slides.layout.autofit import _load_pil_font
from paperops.slides.preview import _wrap_preview_text, check_presentation


def _slide_text(slide) -> str:
    parts: list[str] = []
    for shape in slide.shapes:
        if hasattr(shape, "text_frame"):
            text = shape.text_frame.text.strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _slide_notes(slide) -> str:
    return slide.notes_slide.notes_text_frame.text.strip()


def test_self_intro_slide_titles():
    assert SLIDE_TITLES[:5] == [
        "Diagnostic intelligence should be trained around world models",
        "World-model RCA should verify causal loops, not predict labels",
        "Fault injection makes RCA learnable",
        "Simulation turns RCA data into controlled experiments",
        "Verified propagation makes reasoning scoreable",
    ]
    assert SLIDE_TITLES[-2:] == [
        "World models create leverage across the diagnostic workflow",
        "The opportunity is to train around causal structure",
    ]


def test_self_intro_module_exports_only_live_helpers():
    assert not hasattr(self_intro_build, "SLIDE_SPECS")
    assert not hasattr(self_intro_build, "title_block")


def test_self_intro_theme_uses_installed_font():
    assert make_theme().font_family == "Liberation Sans"


def test_build_self_intro_deck(tmp_path: Path):
    out_path = tmp_path / OUTPUT_FILE.name
    prs = build_presentation(output_path=out_path, render_preview=False)
    assert out_path.exists()
    assert out_path.suffix == ".pptx"
    assert len(prs.slides) == len(SLIDE_TITLES)
    assert IMPLEMENTED_SLIDE_COUNT == len(SLIDE_TITLES)


def test_all_self_intro_slides_are_grounded_and_not_drafts(tmp_path: Path):
    out_path = tmp_path / OUTPUT_FILE.name
    prs = build_presentation(output_path=out_path, render_preview=False)
    review = prs.review()

    slide_texts = [_slide_text(prs.slides[index]) for index in range(len(SLIDE_TITLES))]
    slide_notes = [_slide_notes(prs.slides[index]) for index in range(len(SLIDE_TITLES))]

    assert review["total_issues"] == 0
    assert len(slide_texts) == IMPLEMENTED_SLIDE_COUNT == len(SLIDE_TITLES)
    assert all("Draft scaffold for later grounding." not in text for text in slide_texts)
    assert all(note for note in slide_notes)

    assert "Trusted diagnosis" in slide_texts[0]
    assert "Black-box RCA" in slide_texts[1]
    assert "Finite faults" in slide_texts[2]
    assert "Replayable" in slide_texts[3]
    assert "Verified propagation" in slide_texts[4]
    assert "Root cause" in slide_texts[5]
    assert "Simple baseline" in slide_texts[6]
    assert "Shallow ground truth" in slide_texts[7]
    assert "9,152" in slide_texts[8]
    assert "0.21" in slide_texts[8]
    assert "Effectiveness" in slide_texts[9]
    assert "0.76" in slide_texts[10]
    assert "Backward diagnosis" in slide_texts[11]
    assert "500 cases" in slide_texts[12]
    assert "Analyzer Agent" in slide_texts[13]
    assert "Simulation" in slide_texts[14]
    assert "Many traces" in slide_texts[15]
    assert "Repair suggestion" in slide_texts[16]
    assert "Score the reasoning path" in slide_texts[17]


def test_build_self_intro_preview_when_requested(tmp_path: Path):
    if shutil.which("soffice") is None or shutil.which("pdftoppm") is None:
        return

    out_path = tmp_path / OUTPUT_FILE.name
    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    stale_file = preview_dir / "slide_999.png"
    stale_file.write_text("stale")

    build_presentation(output_path=out_path, render_preview=True)

    assert preview_dir.exists()
    preview_files = sorted(preview_dir.glob("slide_*.png"))
    assert stale_file not in preview_files
    assert len(preview_files) == len(SLIDE_TITLES)


def test_first_five_self_intro_slides_do_not_overflow_or_overlap(tmp_path: Path):
    out_path = tmp_path / OUTPUT_FILE.name
    build_presentation(output_path=out_path, render_preview=False)

    issues = [
        issue for issue in check_presentation(str(out_path))
        if issue["slide"] <= 5 and issue["type"] in {"text_overflow", "overlap"}
    ]

    assert issues == []


def test_preview_wraps_long_text_in_narrow_shapes():
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(image)
    font = _load_pil_font("DejaVu Sans", 18)

    lines = _wrap_preview_text(
        draw,
        "Verified propagation tells us whether the reasoning path is faithful.",
        font,
        180,
    )

    assert len(lines) >= 2
