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
from paperops.slides.layout.auto_size import _load_pil_font
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
        "Building World Models for Diagnostic Intelligence",
        "The Vision",
        "Premise 1: Atomic Faults",
        "Premise 2: Simulatable",
        "Premise 3: Verifiable",
    ]
    assert SLIDE_TITLES[-2:] == [
        "My Research Philosophy",
        "Three Takeaways",
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
    assert "F: O -> G" in slide_texts[7]
    assert "9,152" in slide_texts[10]
    assert "1,430" in slide_texts[10]
    assert "25 fault types" in slide_texts[10]
    assert "0.21" in slide_texts[10]
    assert "0.37" in slide_texts[10]

    assert "0.76" in slide_texts[11]
    assert "0.63" in slide_texts[11]
    assert "Pass@1" in slide_texts[11]
    assert "Path Reachability" in slide_texts[11]

    assert "Backward diagnosis" in slide_texts[12]
    assert "Forward verification" in slide_texts[12]
    assert "Known intervention" in slide_texts[12]

    assert "500 instances" in slide_texts[13]
    assert "step-wise causal annotations" in slide_texts[13]
    assert "process supervision" in slide_texts[13]

    assert "Analyzer Agent" in slide_texts[14]
    assert "Decider Agent" in slide_texts[14]
    assert "Team Manager Agent" in slide_texts[14]
    assert "97%" in slide_texts[14]
    assert "91%" in slide_texts[14]

    assert "Simulation" in slide_texts[15]
    assert "Hypothesis" in slide_texts[15]
    assert "Verify" in slide_texts[15]
    assert "Reward" in slide_texts[15]

    assert "Many traces" in slide_texts[16]
    assert "world model" in slide_texts[16]

    assert "Root-cause localization" in slide_texts[17]
    assert "Fault prediction" in slide_texts[17]
    assert "Repair suggestion" in slide_texts[17]

    assert "Simulate" in slide_texts[18]
    assert "Compress" in slide_texts[18]
    assert "Generalize" in slide_texts[18]

    assert "Three Takeaways" in slide_texts[19]
    assert "Score the path, not just the answer." in slide_texts[19]
    assert "Train on verification loops, not only labels." in slide_texts[19]
    assert "Build the model once, then reuse it broadly." in slide_texts[19]
    assert "Build the model once" in slide_texts[19]


def test_build_self_intro_preview_when_requested(tmp_path: Path):
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
