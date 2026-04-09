from pathlib import Path

from paperops.slides import (
    Absolute,
    AbsoluteItem,
    Grid,
    GridItem,
    HStack,
    Layer,
    Presentation,
    Rect,
    Spacer,
    Text,
    themes,
)


def _build_simple_deck() -> Presentation:
    prs = Presentation()
    sb = prs.slide(title="Schema Smoke")
    sb.layout(
        HStack(
            gap=0.2,
            children=[
                Text(text="Left"),
                Spacer(grow=1, size_mode_x="fill"),
                Text(text="Right"),
            ],
        )
    )
    return prs


def test_presentation_exposes_core_api_only():
    assert not hasattr(Presentation, "cover")
    assert not hasattr(Presentation, "agenda")
    assert not hasattr(Presentation, "evidence_comparison")


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
    assert report["issue_counts"]["total"] == len(report["issues"])
    assert report["layout_summary"]["artifact"] == "layout_review"
    assert report["top_problem_slides"] == []


def test_grid_item_renders_explicit_children():
    prs = Presentation()
    sb = prs.slide(title="Grid Item")
    sb.layout(Grid(children=[GridItem(child=Text(text="grid child"), row=0, col=0)]))

    prs.review()

    texts = [getattr(shape, "text", "") for shape in prs._pptx.slides[0].shapes]
    assert "grid child" in texts


def test_slide_layout_can_rerender_after_mutation():
    prs = Presentation()
    sb = prs.slide(title="Mutate")
    sb.layout(Text(text="first"))
    prs.review()

    sb.layout(Text(text="second"))
    prs.review()

    texts = [getattr(shape, "text", "") for shape in prs._pptx.slides[0].shapes]
    assert "second" in texts
    assert "first" not in texts


def test_layer_and_absolute_layout_render_without_review_issues(tmp_path: Path):
    prs = Presentation(theme=themes.executive)
    sb = prs.slide(title="Layered")
    sb.layout(
        Layer(
            children=[
                Rect(color="bg_alt"),
                Absolute(
                    children=[
                        AbsoluteItem(
                            left=0.35,
                            top=0.30,
                            width=1.8,
                            child=Text(text="Overlay", font_size="heading", bold=True),
                        ),
                        AbsoluteItem(
                            left=0.35,
                            top=1.05,
                            width=2.4,
                            child=Text(text="Pinned note", font_size="caption", color="text_mid"),
                        ),
                    ],
                ),
            ],
        )
    )

    report = prs.review_deck(
        output_path=str(tmp_path / "layer_absolute.pptx"),
        render_preview=False,
    )

    assert report["issue_counts"]["total"] == 0


def test_executive_theme_is_available():
    theme = themes.executive

    assert theme.name == "executive"
    assert theme.font_family == "Liberation Sans"
    assert theme.resolve_color("primary") == "#16324F"
    assert theme.resolve_font_size("body") == 16


def test_academic_seminar_theme_is_available():
    theme = themes.academic_seminar

    assert theme.name == "academic_seminar"
    assert theme.font_family == "Liberation Sans"
    assert theme.resolve_color("primary") == "#1F3A5F"
    assert theme.resolve_font_size("title") == 26
