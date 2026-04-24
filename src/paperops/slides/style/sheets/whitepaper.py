from __future__ import annotations

from paperops.slides.style.stylesheet import StyleSheet

SHEET = StyleSheet(
    {
        ".cover": {"padding": "xl", "bg": "bg"},
        "slide": {"cols": "1fr 1fr", "gap": "xl", "padding": "lg", "bg": "bg"},
        ".title": {"font": "heading", "font-weight": "bold", "color": "text"},
        ".subtitle": {"font": "subtitle", "color": "text_mid"},
        ".heading": {"font": "heading", "color": "text"},
        ".text": {"font": "body", "line-height": 1.25, "color": "text"},
        ".prose": {"font": "body", "line-height": 1.25},
        ".card": {"bg": "bg_alt", "padding": "md", "border": "border", "radius": "sm"},
        ".kpi": {"bg": "bg_accent", "padding": "sm"},
        ".table": {"bg": "bg", "padding": "xs", "border": "border"},
        ".table .heading": {"font-weight": "bold"},
        ".chart": {"bg": "bg_alt", "padding": "md"},
        ".figure": {"gap": "sm", "bg": "bg_alt", "padding": "md"},
        ".caption": {"font": "caption", "color": "text_mid"},
        ".source": {"font": "caption", "color": "text_light"},
        ".stepper": {"gap": "sm"},
        ".timeline-item": {"padding": "sm", "border": "border"},
        ".note": {"color": "text_mid", "font": "small"},
    }
)
