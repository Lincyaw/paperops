from __future__ import annotations

from paperops.slides.style.stylesheet import StyleSheet

SHEET = StyleSheet(
    {
        ".cover": {"padding": "2xl", "align": "center", "bg": "bg"},
        ".cover title": {
            "font": "title",
            "font-weight": "bold",
            "color": "text",
            "align": "center",
        },
        ".cover subtitle": {"font": "subtitle", "color": "text_mid", "align": "center"},
        ".title": {"font": "title", "color": "text", "margin-bottom": "xs"},
        ".subtitle": {"font": "subtitle", "color": "text_mid"},
        ".heading": {"font": "heading", "color": "text"},
        ".text": {"font": "body", "color": "text"},
        ".prose": {"font": "body", "color": "text", "line-height": 1.2},
        ".card": {"bg": "bg_alt", "padding": "lg", "radius": "sm", "gap": "md"},
        ".kpi": {"bg": "bg_accent", "padding": "sm", "gap": "xs"},
        ".label": {"font": "caption", "color": "text_mid"},
        ".value": {"font": "heading", "color": "text", "font-weight": "bold"},
        ".delta": {"font": "caption", "color": "accent"},
        ".positive": {"color": "positive"},
        ".negative": {"color": "negative"},
        ".neutral": {"color": "text_mid"},
        ".callout": {"bg": "bg_alt", "padding": "md", "border": "border"},
        ".quote": {"bg": "bg_alt", "padding": "lg", "gap": "sm"},
        ".quote-mark": {"font": "title", "color": "text_mid"},
        ".badge": {"bg": "accent", "color": "bg", "padding-x": "sm", "padding-y": "xs"},
        ".step": {"gap": "xs"},
        ".spacer": {"min-height": "md"},
        ".image": {"border": "border", "padding": "xs", "bg": "bg_alt"},
        ".chart": {"bg": "bg_alt", "padding": "md", "border": "border"},
        ".note": {"color": "text_light"},
    }
)
