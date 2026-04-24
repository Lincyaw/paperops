from __future__ import annotations

from paperops.slides.style.stylesheet import StyleSheet

SHEET = StyleSheet(
    {
        ".cover": {"padding": "xl", "bg": "bg"},
        ".title": {"font": "title", "font-weight": "bold", "color": "text"},
        ".subtitle": {"font": "subtitle", "color": "text_mid"},
        ".heading": {"font": "heading", "color": "text"},
        ".text": {"font": "body", "color": "text", "line-height": 1.35},
        ".prose": {"font": "body", "color": "text", "line-height": 1.4},
        ".card": {"bg": "bg_alt", "padding": "md", "border": "border", "radius": "md"},
        ".kpi": {"bg": "bg_accent", "padding": "sm", "gap": "xs", "border": "border"},
        ".label": {"font": "caption", "color": "text_mid", "font-weight": "bold"},
        ".value": {"font": "heading", "color": "primary"},
        ".delta": {"font": "caption"},
        ".positive": {"color": "positive"},
        ".negative": {"color": "negative"},
        ".neutral": {"color": "text_mid"},
        ".callout": {"bg": "bg", "border": "accent", "padding": "md"},
        ".quote": {"bg": "bg_alt", "padding": "md", "border-left": ["accent", "md"], "gap": "sm"},
        ".pullquote": {"bg": "bg", "padding": "xl", "gap": "xs", "border-left": ["primary", "lg"]},
        ".timeline": {"gap": "md"},
        ".timeline-item": {"padding": "sm", "bg": "bg_alt"},
        ".table": {"border": "border", "padding": "xs", "bg": "bg"},
        ".chart": {"bg": "bg_alt", "padding": "md"},
        ".image": {"border": "border", "padding": "xs"},
        ".figure": {"gap": "xs", "padding": "md"},
        ".caption": {"font": "caption", "color": "text_mid"},
        ".source": {"font": "caption", "color": "text_light"},
        ".note": {"font": "small", "color": "text_light", "margin-top": "xs"},
    }
)
