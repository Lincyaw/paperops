from __future__ import annotations

from paperops.slides.style.stylesheet import StyleSheet

SHEET = StyleSheet(
    {
        ".cover": {"padding": "2xl", "bg": "bg", "align": "center", "gap": "xl"},
        ".title": {"font": "title", "font-weight": "bold", "color": "text"},
        ".subtitle": {"font": "subtitle", "color": "text_mid"},
        ".heading": {"font": "heading", "color": "primary"},
        ".text": {"font": "body", "line-height": 1.15},
        ".prose": {"font": "body", "line-height": 1.2},
        ".card": {"bg": "bg_accent", "padding": "lg", "radius": "md", "gap": "sm"},
        ".kpi": {"bg": "primary", "color": "bg", "padding": "md", "gap": "xs"},
        ".kpi .label": {"color": "bg"},
        ".kpi .value": {"font": "title", "font-weight": "bold"},
        ".kpi .delta": {"color": "bg"},
        ".callout": {"bg": "accent", "color": "bg", "padding": "lg", "gap": "sm"},
        ".quote": {"bg": "accent", "color": "bg", "padding": "xl", "gap": "sm"},
        ".pullquote": {"bg": "primary", "color": "bg", "padding": "xl"},
        ".pullquote .body": {"font": "heading"},
        ".keypoint": {"bg": "bg_alt", "padding": "md", "gap": "xs"},
        ".stepper": {"gap": "sm", "animate": "fade-in"},
        ".timeline": {"gap": "sm"},
        ".timeline-item": {"padding": "sm", "bg": "bg", "border": "border"},
        ".figure": {"bg": "bg_alt", "padding": "md"},
        ".caption": {"font": "caption", "color": "text"},
        ".source": {"font": "caption", "color": "text_mid"},
        ".note": {"color": "text_mid", "font": "small"},
    }
)
