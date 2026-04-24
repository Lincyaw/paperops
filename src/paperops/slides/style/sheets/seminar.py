from __future__ import annotations

from paperops.slides.style.stylesheet import StyleSheet

SHEET = StyleSheet(
    {
        ".cover": {"padding": "xl", "bg": "bg"},
        ".title": {"font": "heading", "font-weight": "bold", "color": "text"},
        ".subtitle": {"font": "subtitle", "color": "text_mid"},
        ".heading": {"font": "heading", "color": "primary"},
        ".text": {"font": "body", "line-height": 1.22},
        ".prose": {"font": "body", "line-height": 1.35},
        ".card": {"bg": "bg_alt", "padding": "md", "radius": "md", "gap": "xs"},
        ".kpi": {"bg": "bg_accent", "padding": "md", "gap": "sm", "radius": "sm"},
        ".label": {"font": "caption", "color": "text_mid"},
        ".value": {"font": "title", "color": "primary"},
        ".delta": {"font": "caption", "font-weight": "bold"},
        ".positive": {"color": "positive"},
        ".negative": {"color": "negative"},
        ".neutral": {"color": "text_mid"},
        ".chart": {"bg": "bg_alt", "padding": "md", "border": "border"},
        ".grid": {"gap": "md", "cols": "1fr 1fr"},
        ".stepper": {"gap": "md"},
        ".step": {"padding": "sm", "bg": "bg_alt"},
        ".timeline": {"gap": "md"},
        ".timeline-item": {"padding": "sm", "border-left": ["primary", "sm"]},
        ".table": {"bg": "bg", "border": "border"},
        ".figure": {"gap": "xs"},
        ".caption": {"font": "caption", "color": "text"},
    }
)
