from __future__ import annotations

from paperops.slides.style.stylesheet import StyleSheet

SHEET = StyleSheet(
    {
        ".cover": {"padding": "xl", "bg": "bg", "align": "center", "gap": "lg"},
        ".title": {"font": "title", "font-weight": "bold", "color": "text"},
        ".subtitle": {"font": "subtitle", "color": "accent"},
        ".heading": {"font": "heading", "color": "primary"},
        ".text": {"font": "body", "color": "text", "line-height": 1.1},
        ".prose": {"font": "body", "color": "text", "line-height": 1.15},
        ".card": {"bg": "bg_accent", "padding": "lg", "radius": "sm", "gap": "sm"},
        ".kpi": {"bg": "accent", "color": "bg", "padding": "md", "gap": "xs"},
        ".kpi .label": {"color": "bg"},
        ".kpi .value": {"font": "title", "font-weight": "bold"},
        ".delta": {"font": "caption", "color": "bg"},
        ".positive": {"color": "bg"},
        ".negative": {"color": "bg"},
        ".neutral": {"color": "bg"},
        ".callout": {"bg": "bg", "border": "accent", "padding": "md", "radius": "full"},
        ".quote": {"bg": "bg", "padding": "lg", "radius": "full"},
        ".pullquote": {"bg": "accent", "color": "bg", "padding": "xl"},
        ".pullquote .body": {"font": "heading", "font-weight": "bold"},
        ".keypoint": {"bg": "bg_alt", "padding": "md", "radius": "sm", "gap": "xs"},
        ".stepper": {"gap": "lg"},
        ".step": {"bg": "accent", "color": "bg", "padding": "sm"},
        ".timeline": {"gap": "sm"},
        ".timeline-item": {"bg": "bg_alt", "padding": "sm"},
        ".chart": {"bg": "bg", "padding": "lg"},
        ".figure": {"bg": "bg", "padding": "sm"},
        ".caption": {"font": "caption", "color": "text"},
        ".source": {"font": "caption", "color": "text_light"},
        ".note": {"font": "caption", "color": "text_light"},
    }
)
