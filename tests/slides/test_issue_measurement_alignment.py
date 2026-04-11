from pathlib import Path

import pytest

from paperops.slides import Absolute, AbsoluteItem, Presentation, Text
from paperops.slides.preview import check_presentation


def _find_text_shape(slide, needle: str):
    for shape in slide.shapes:
        if not getattr(shape, "has_text_frame", False):
            continue
        text = shape.text_frame.text or ""
        if needle in text:
            return shape
    raise AssertionError(f"Could not find shape containing: {needle!r}")


def _line_spacing_as_multiplier(paragraph, font_size_pt: float) -> float:
    line_spacing = paragraph.line_spacing
    assert line_spacing is not None
    if hasattr(line_spacing, "pt"):
        return float(line_spacing.pt) / font_size_pt
    return float(line_spacing)


def test_text_frame_margin_and_line_spacing_are_written():
    prs = Presentation()
    sb = prs.slide()
    sb.layout(
        Absolute(
            children=[
                AbsoluteItem(
                    left=0.8,
                    top=1.0,
                    width=4.8,
                    height=1.4,
                    child=Text(
                        text="Margin and spacing sync probe",
                        font_size=16,
                        line_spacing=1.1,
                        margin_x=0.40,
                        margin_y=0.20,
                    ),
                )
            ]
        )
    )

    prs.review()
    shape = _find_text_shape(prs._pptx.slides[0], "Margin and spacing sync probe")
    tf = shape.text_frame
    paragraph = tf.paragraphs[0]

    # Margin values should be propagated to text frame in a symmetric way.
    assert tf.margin_left.inches == pytest.approx(tf.margin_right.inches, abs=1e-3)
    assert tf.margin_top.inches == pytest.approx(tf.margin_bottom.inches, abs=1e-3)
    assert tf.margin_left.inches > 0.05
    assert tf.margin_top.inches > 0.05

    # Paragraph line spacing should be explicitly set from the node.
    assert _line_spacing_as_multiplier(paragraph, font_size_pt=16.0) == pytest.approx(
        1.1, abs=0.05
    )


def test_text_overflow_still_triggers_for_obvious_overflow(tmp_path: Path):
    prs = Presentation()
    sb = prs.slide()
    sb.layout(
        Absolute(
            children=[
                AbsoluteItem(
                    left=0.8,
                    top=1.0,
                    width=1.35,
                    height=0.44,
                    child=Text(
                        text=(
                            "This paragraph is intentionally too long for the box and should "
                            "always trigger overflow detection in saved-file review."
                        ),
                        font_size=18,
                        line_spacing=1.2,
                        margin_x=0.0,
                        margin_y=0.0,
                    ),
                )
            ]
        )
    )

    out_path = tmp_path / "obvious_overflow.pptx"
    prs.save(str(out_path))
    issues = check_presentation(str(out_path))

    assert any(issue["type"] == "text_overflow" for issue in issues)


def test_tight_but_fitting_text_is_not_flagged_as_overflow(tmp_path: Path):
    prs = Presentation()
    sb = prs.slide()
    sb.layout(
        Absolute(
            children=[
                AbsoluteItem(
                    left=0.8,
                    top=1.0,
                    width=4.4,
                    height=0.52,
                    child=Text(
                        text="Top@1 accuracy\nMRR (Mean Reciprocal Rank)",
                        font_size=16,
                        line_spacing=1.0,
                        margin_x=0.0,
                        margin_y=0.0,
                    ),
                )
            ]
        )
    )

    out_path = tmp_path / "tight_fit.pptx"
    prs.save(str(out_path))
    issues = check_presentation(str(out_path))
    overflow_issues = [issue for issue in issues if issue["type"] == "text_overflow"]

    assert overflow_issues == []
