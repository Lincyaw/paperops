from pathlib import Path

from paperops.slides import HStack, Presentation, TextBlock
from paperops.slides.components.composite import Callout
from paperops.slides.core.theme import themes
from paperops.slides.layout.types import Constraints


def _build_simple_deck() -> Presentation:
    prs = Presentation()
    sb = prs.slide(title="Schema Smoke")
    sb.layout(
        HStack(
            gap=0.2,
            children=[
                TextBlock(text="Left"),
                TextBlock(text="Right"),
            ],
        )
    )
    return prs


def test_review_returns_stable_layout_artifact():
    prs = _build_simple_deck()
    report = prs.review()

    assert report["schema_version"] == "2026-04-09"
    assert report["artifact"] == "layout_review"
    assert report["total_slides"] == 1
    assert "issue_counts" in report
    assert "slides" in report
    assert "issues" in report
    assert report["total_issues"] == report["issue_counts"]["total"]
    assert len(report["slides"]) == 1
    assert report["slides"][0]["slide_number"] == 1
    assert report["slides"][0]["title"] == "Schema Smoke"
    assert report["slides"][0]["issue_count"] == len(report["slides"][0]["issues"])

    for issue in report["issues"]:
        assert issue["slide"] == 1
        assert issue["source"] == "layout"
        assert "code" in issue
        assert "message" in issue


def test_review_deck_returns_stable_merged_artifact(tmp_path: Path):
    prs = _build_simple_deck()
    out_path = tmp_path / "deck.pptx"

    report = prs.review_deck(
        output_path=str(out_path),
        render_preview=False,
    )

    assert out_path.exists()
    assert report["schema_version"] == "2026-04-09"
    assert report["artifact"] == "deck_review"
    assert report["total_slides"] == 1
    assert "issues" in report
    assert "issue_counts" in report
    assert "slides" in report
    assert "layout_summary" in report
    assert report["layout_summary"]["artifact"] == "layout_review"
    assert report["layout_summary"]["total_slides"] == 1
    assert report["issue_counts"]["total"] == len(report["issues"])
    assert report["layout_issue_count"] == report["issue_counts"]["layout"]

    slide = report["slides"][0]
    assert slide["slide_number"] == 1
    assert slide["title"] == "Schema Smoke"
    assert slide["issue_count"] == len(slide["issues"])
    assert slide["issue_counts"]["total"] == slide["issue_count"]
    assert "layout_issues" in slide
    assert "saved_file_issues" in slide
    assert "summary" in slide
    assert "score" in slide


def test_review_deck_does_not_rank_issue_free_slides(tmp_path: Path):
    prs = _build_simple_deck()
    out_path = tmp_path / "deck.pptx"

    report = prs.review_deck(
        output_path=str(out_path),
        render_preview=False,
    )

    assert report["issue_counts"]["total"] == 0
    assert report["top_problem_slides"] == []


def test_callout_measure_expands_height_when_width_is_constrained():
    callout = Callout(
        title="Shallow ground truth",
        body="Labels only reach service level. No code-level or propagation-level annotation.",
        color="negative",
    )

    intrinsic = callout.measure(Constraints(max_width=3.57), themes.professional)

    assert intrinsic.preferred_width <= 3.57
    assert intrinsic.preferred_height > 1.3
