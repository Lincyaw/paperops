"""Type conversion utilities — bridge between public API and pptx internals."""

try:
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency fallback
    if exc.name != "pptx":
        raise

    class _Length(float):
        @property
        def inches(self) -> float:
            return float(self)

        @property
        def pt(self) -> float:
            return float(self)

    def Inches(value: float) -> _Length:  # type: ignore[override]
        return _Length(value)

    def Pt(value: float) -> _Length:  # type: ignore[override]
        return _Length(value)

    Emu = _Length  # type: ignore[assignment]

    class RGBColor(tuple):  # type: ignore[override]
        def __new__(cls, r: int, g: int, b: int):
            return tuple.__new__(cls, (r, g, b))


def resolve_color(theme, color) -> RGBColor:
    """Convert a color spec to pptx RGBColor using the given theme."""
    hex_str = theme.resolve_color(color)
    hex_str = hex_str.lstrip("#")
    return RGBColor(int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def resolve_font_size(theme, size) -> Pt:
    """Convert a font size spec to pptx Pt using the given theme."""
    pt_val = theme.resolve_font_size(size)
    return Pt(pt_val)


def resolve_size(inches: float) -> Emu:
    """Convert inches (float) to pptx Emu."""
    return Inches(inches)


def hex_to_rgb_tuple(hex_str: str) -> tuple[int, int, int]:
    """Convert hex color string to (r, g, b) tuple."""
    hex_str = hex_str.lstrip("#")
    return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def contrast_ratio(color1_hex: str, color2_hex: str) -> float:
    """Calculate WCAG contrast ratio between two hex colors."""
    def luminance(hex_str):
        r, g, b = hex_to_rgb_tuple(hex_str)
        rs, gs, bs = r / 255, g / 255, b / 255
        r_lin = rs / 12.92 if rs <= 0.03928 else ((rs + 0.055) / 1.055) ** 2.4
        g_lin = gs / 12.92 if gs <= 0.03928 else ((gs + 0.055) / 1.055) ** 2.4
        b_lin = bs / 12.92 if bs <= 0.03928 else ((bs + 0.055) / 1.055) ** 2.4
        return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    l1 = luminance(color1_hex)
    l2 = luminance(color2_hex)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)
