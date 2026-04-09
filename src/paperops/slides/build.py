"""Presentation class — main entry point for creating presentations."""

from __future__ import annotations

import os
import sys
import tempfile

from pptx import Presentation as PptxPresentation
from pptx.util import Inches

from paperops.slides.core.constants import SLIDE_HEIGHT, SLIDE_WIDTH
from paperops.slides.core.theme import Theme, themes
from paperops.slides.preview import review_deck_artifacts
from paperops.slides.slides.base import SlideBuilder
from paperops.slides.slides.templates import register_templates


class Presentation:
    """Main entry point for creating presentations."""

    def __init__(self, theme: Theme | None = None):
        self._theme = theme if theme is not None else themes.professional
        self._pptx = PptxPresentation()
        self._pptx.slide_width = Inches(SLIDE_WIDTH)
        self._pptx.slide_height = Inches(SLIDE_HEIGHT)
        self._builders: list[SlideBuilder] = []

    def slide(self, title=None, reference=None, background=None) -> SlideBuilder:
        layout = self._pptx.slide_layouts[6]
        pptx_slide = self._pptx.slides.add_slide(layout)
        sb = SlideBuilder(
            pptx_slide,
            self._theme,
            title=title,
            reference=reference,
            slide_number=len(self._builders) + 1,
        )
        if background is not None:
            sb.background(color=background)
        self._builders.append(sb)
        return sb

    def save(self, path: str):
        all_issues = []
        for slide_index, sb in enumerate(self._builders, 1):
            issues = sb._render(slide_number=slide_index)
            if issues:
                all_issues.extend(issues)

        if all_issues:
            print(f"[SlideCraft] {len(all_issues)} layout issue(s):", file=sys.stderr)
            for issue in all_issues:
                print(f"  [{issue.get('code', issue.get('type'))}] {issue.get('detail', issue.get('message'))}", file=sys.stderr)

        self._pptx.save(path)

    def review(self) -> dict:
        issues: list[dict] = []
        slides: list[dict] = []
        for slide_index, sb in enumerate(self._builders, 1):
            slide_issues = sb._render(slide_number=slide_index) or []
            normalized = [_normalize_issue(issue, slide_index) for issue in slide_issues]
            issues.extend(normalized)
            slides.append({
                "slide_number": slide_index,
                "title": getattr(sb, "_title", None),
                "issue_count": len(normalized),
                "issues": normalized,
            })

        issue_counts = {
            "total": len(issues),
            "error": sum(1 for issue in issues if issue.get("severity") == "error"),
            "warning": sum(1 for issue in issues if issue.get("severity") == "warning"),
            "info": sum(1 for issue in issues if issue.get("severity") == "info"),
        }
        return {
            "schema_version": "2026-04-09",
            "artifact": "layout_review",
            "total_slides": len(self._builders),
            "total_issues": len(issues),
            "issue_counts": issue_counts,
            "issues": issues,
            "slides": slides,
        }

    def review_deck(self, output_path: str, render_preview: bool = True, output_dir: str | None = None) -> dict:
        layout_review = self.review()
        self.save(output_path)
        preview_paths: list[str] = []
        if render_preview:
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                for filename in os.listdir(output_dir):
                    if filename.startswith("slide_") and filename.endswith(".png"):
                        os.unlink(os.path.join(output_dir, filename))
            preview_paths = self.preview(output_dir=output_dir)
        slide_titles = [getattr(sb, "_title", None) for sb in self._builders]
        return review_deck_artifacts(
            output_path,
            layout_issues=layout_review["issues"],
            preview_paths=preview_paths,
            slide_titles=slide_titles,
            layout_summary=layout_review,
        )

    def preview(self, slides=None, output_dir=None) -> list[str]:
        from paperops.slides.preview import render_slide_preview_powerpoint

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="slidecraft_preview_")
        os.makedirs(output_dir, exist_ok=True)

        for slide_index, sb in enumerate(self._builders, 1):
            sb._render(slide_number=slide_index)

        pptx_path = os.path.join(output_dir, "_temp_preview.pptx")
        self.save(pptx_path)
        paths = render_slide_preview_powerpoint(pptx_path, output_dir)
        os.unlink(pptx_path)

        if slides is not None:
            paths = [p for p in paths if any(f"slide_{s + 1:03d}.png" in p for s in slides)]
        return sorted(paths)


def _normalize_issue(issue: dict, slide_number: int) -> dict:
    normalized = dict(issue)
    normalized.setdefault("slide", slide_number)
    normalized.setdefault("source", "layout")
    normalized.setdefault("code", normalized.get("type", "layout_issue"))
    normalized.setdefault("message", normalized.get("detail", normalized.get("code", "layout issue")))
    normalized.setdefault("detail", normalized["message"])
    normalized.setdefault("severity", "warning")
    return normalized


register_templates(Presentation)
