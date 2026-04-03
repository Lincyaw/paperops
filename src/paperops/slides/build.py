"""Presentation class — main entry point for creating presentations."""

from __future__ import annotations

import sys

from pptx import Presentation as PptxPresentation
from pptx.util import Inches

from paperops.slides.core.constants import SLIDE_WIDTH, SLIDE_HEIGHT
from paperops.slides.core.theme import Theme, themes
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
        """Create a custom slide with optional title. Returns SlideBuilder.

        Args:
            background: Optional background color name/hex. Calls sb.background()
                        automatically when provided.
        """
        layout = self._pptx.slide_layouts[6]  # blank layout
        pptx_slide = self._pptx.slides.add_slide(layout)
        sb = SlideBuilder(pptx_slide, self._theme, title=title, reference=reference)
        if background is not None:
            sb.background(color=background)
        self._builders.append(sb)
        return sb

    def save(self, path: str):
        """Build and save the presentation."""
        all_issues = []
        for sb in self._builders:
            issues = sb._render()
            if issues:
                all_issues.extend(issues)

        if all_issues:
            print(f"[SlideCraft] {len(all_issues)} layout issue(s):", file=sys.stderr)
            for iss in all_issues:
                print(f"  [{iss['type']}] {iss['detail']}", file=sys.stderr)

        self._pptx.save(path)

    def review(self) -> dict:
        """Run validation and return structured report."""
        all_issues = []
        for sb in self._builders:
            issues = sb._render()
            if issues:
                all_issues.extend(issues)

        return {
            "total_slides": len(self._builders),
            "total_issues": len(all_issues),
            "issues": all_issues,
        }

    def review_deck(self, output_path: str, render_preview: bool = True, output_dir: str | None = None) -> dict:
        """Build deck artifacts and return an integrated review report."""
        from paperops.slides.preview import review_deck_artifacts
        import os

        layout_report = self.review()
        self.save(output_path)
        preview_paths = []
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
            layout_issues=layout_report["issues"],
            preview_paths=preview_paths,
            slide_titles=slide_titles,
        )

    def preview(self, slides=None, output_dir=None) -> list[str]:
        """Render slides to PNG for visual inspection.

        Args:
            slides: list of 0-based slide indices (default: all)
            output_dir: directory for PNG files (default: auto temp dir)

        Returns: list of PNG file paths
        """
        from paperops.slides.preview import render_slide_preview
        import os
        import tempfile

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="slidecraft_preview_")
        os.makedirs(output_dir, exist_ok=True)

        if slides is None:
            indices = range(len(self._builders))
        else:
            indices = slides

        paths = []
        for idx in indices:
            if idx < 0 or idx >= len(self._builders):
                continue
            sb = self._builders[idx]
            sb._render()  # ensure rendered
            out_path = os.path.join(output_dir, f"slide_{idx + 1:03d}.png")
            render_slide_preview(sb, out_path)
            paths.append(out_path)

        return paths


# Register template methods onto Presentation
register_templates(Presentation)
