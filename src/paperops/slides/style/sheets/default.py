from __future__ import annotations

from paperops.slides.style.stylesheet import StyleSheet

SHEET = StyleSheet(
    {
        "slide": {"padding": "md", "bg": "bg"},
        ".title": {"font": "title", "color": "text", "margin-bottom": 0.0},
        ".subtitle": {"font": "subtitle", "color": "text_mid", "margin-bottom": 0.0},
        ".heading": {"font": "heading", "color": "text"},
        ".text": {"font": "body", "color": "text"},
        ".prose": {"font": "body", "color": "text"},
        ".card": {"bg": "bg_alt", "padding": "md", "radius": "md", "gap": "sm"},
        ".kpi": {"bg": "bg_accent", "padding": "md", "gap": "xs"},
        ".callout": {"bg": "bg_alt", "padding": "md"},
        ".quote": {"bg": "bg_alt", "padding": "md"},
        ".pullquote": {"bg": "bg_accent", "padding": "lg"},
        ".keypoint": {"bg": "bg_alt", "padding": "sm"},
        ".stepper": {"gap": "sm"},
        ".timeline": {"gap": "sm"},
        ".timeline-item": {"padding": "sm", "border": "border"},
        ".figure": {"bg": "bg_alt", "padding": "sm", "gap": "sm"},
        ".caption": {"font": "caption", "color": "text_mid"},
        ".spacer": {"bg": "none"},
        ".note": {"font": "caption", "color": "text_light"},
    }
)
