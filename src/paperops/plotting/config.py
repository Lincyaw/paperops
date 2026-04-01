# -*- coding: utf-8 -*-
"""Plotting configuration and utilities for publication-quality academic figures.

Provides preset themes, figure sizing helpers, and save utilities
optimised for top-venue papers (ACM, IEEE, NeurIPS, etc.).
"""

import logging
import tempfile

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logger = logging.getLogger(__name__)

# ======================================================================
# Color Palettes
# ======================================================================

BASE_COLORS = [
    "#dd9f94",
    "#f5e8bd",
    "#b0bda0",
    "#b87264",
    "#464666",
    "#7da4a3",
    "#b3b6be",
    "#a0acc8",
    "#b0bfa1",
    "#db716e",
    "#be958a",
    "#e4ce90",
    "#c98849",
    "#785177",
    "#56777e",
    "#c5ccdb",
]

COLOR_PALETTE_QUALITATIVE = BASE_COLORS

GRAYSCALE_PALETTE = sns.color_palette("gray_r", n_colors=5)

RECOMMENDED_CMAPS = {
    "sequential": "viridis",
    "diverging": "coolwarm",
}

# ======================================================================
# Style Constants
# ======================================================================

HATCH_PATTERNS = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
MARKER_STYLES = ["o", "s", "v", "^", "D", "<", ">", "p", "*"]
LINE_STYLES = ["-", "--", "-.", ":"]

# ======================================================================
# Figure Size Presets (inches)
# ======================================================================

FIGURE_SIZES = {
    "single": (3.5, 2.5),     # Single-column figure
    "double": (7.0, 3.5),     # Double-column / full-width figure
    "square": (3.5, 3.5),     # Square (confusion matrix, heatmap)
    "wide": (7.0, 2.5),       # Wide panoramic (timeline, multi-panel)
}

# ======================================================================
# Theme Definitions
# ======================================================================

_COMMON_CONFIG = {
    # Figure output
    "figure.dpi": 300,
    "figure.autolayout": False,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    # Font embedding (Type 42 for ACM/IEEE compliance)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Axes
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.major.size": 3,
    "ytick.minor.size": 1.5,
    # Legend
    "legend.frameon": False,
    "legend.loc": "best",
    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    # Bar
    "hatch.linewidth": 0.5,
    # Box plot
    "boxplot.boxprops.linewidth": 1.5,
    "boxplot.medianprops.linewidth": 1.5,
    "boxplot.whiskerprops.linewidth": 1.5,
    "boxplot.capprops.linewidth": 1.5,
    "boxplot.flierprops.markeredgecolor": "gray",
    "boxplot.flierprops.marker": ".",
    # Error bars
    "errorbar.capsize": 2,
    # Scatter
    "scatter.marker": "o",
}

THEMES = {
    "classic": {
        **_COMMON_CONFIG,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "gray",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
    },
    "modern": {
        **_COMMON_CONFIG,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
    },
    "grayscale": {
        **_COMMON_CONFIG,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "gray",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
    },
}

# Palette associated with each theme
_THEME_PALETTES = {
    "classic": COLOR_PALETTE_QUALITATIVE,
    "modern": COLOR_PALETTE_QUALITATIVE,
    "grayscale": GRAYSCALE_PALETTE,
}

# ======================================================================
# Core API
# ======================================================================


def apply_plot_config(theme="classic"):
    """Apply a publication-ready plotting theme.

    Args:
        theme: Theme name ("classic", "modern", "grayscale") or a custom
               dict of matplotlib rcParams.
    """
    if isinstance(theme, dict):
        plt.rcParams.update(theme)
        return

    if theme not in THEMES:
        raise ValueError(
            f"Unknown theme '{theme}'. Available: {sorted(THEMES.keys())}"
        )

    plt.rcParams.update(THEMES[theme])

    palette = _THEME_PALETTES[theme]
    sns.set_palette(palette)

    if theme == "classic" or theme == "grayscale":
        sns.set_context("paper")
        sns.set_style("ticks")
    elif theme == "modern":
        sns.set_context("paper")
        sns.set_style("white")

    logger.info("Plotting theme '%s' applied.", theme)


def figure(size="single", nrows=1, ncols=1, **kwargs):
    """Create a figure with preset academic sizing.

    Args:
        size: Preset name ("single", "double", "square", "wide") or
              a (width, height) tuple in inches.
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        **kwargs: Forwarded to plt.subplots().

    Returns:
        (fig, ax) or (fig, axes) — same as plt.subplots().
    """
    if isinstance(size, str):
        if size not in FIGURE_SIZES:
            raise ValueError(
                f"Unknown size '{size}'. Available: {sorted(FIGURE_SIZES.keys())}"
            )
        w, h = FIGURE_SIZES[size]
    else:
        w, h = size

    return plt.subplots(nrows=nrows, ncols=ncols, figsize=(w, h), **kwargs)


def save(fig, path=None, dpi=300, **kwargs):
    """Save figure with academic defaults and close it.

    Applies tight_layout, saves with bbox_inches='tight', then closes
    the figure to free memory.

    Args:
        fig: Matplotlib Figure object.
        path: Output file path. If None, saves to a temporary PNG file.
        dpi: Resolution (default 300).
        **kwargs: Forwarded to fig.savefig().

    Returns:
        The file path (useful when path is None for temp files, or for
        passing directly to paperops.slides.Image).
    """
    if path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        path = tmp.name

    fig.tight_layout()
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, dpi=dpi, **kwargs)
    plt.close(fig)
    return path


def colors(n=None):
    """Get colors from the current qualitative palette.

    Args:
        n: Number of colors to return. None returns the full palette.

    Returns:
        List of hex color strings.
    """
    palette = list(COLOR_PALETTE_QUALITATIVE)
    if n is None:
        return palette
    return palette[:n]


# Keep backward-compatible module-level config dicts for direct access
FONT_CONFIG = {k: v for k, v in THEMES["classic"].items() if k.startswith("font.") or "size" in k}
FIGURE_CONFIG = {k: v for k, v in THEMES["classic"].items() if k.startswith(("figure.", "savefig.", "axes."))}
TICK_CONFIG = {k: v for k, v in THEMES["classic"].items() if k.startswith(("xtick.", "ytick."))}
LEGEND_CONFIG = {k: v for k, v in THEMES["classic"].items() if k.startswith("legend.")}
BAR_CONFIG = {k: v for k, v in THEMES["classic"].items() if k.startswith("hatch.")}
LINE_CONFIG = {k: v for k, v in THEMES["classic"].items() if k.startswith("lines.")}
BOXPLOT_CONFIG = {k: v for k, v in THEMES["classic"].items() if k.startswith("boxplot.")}
ERRORBAR_CONFIG = {k: v for k, v in THEMES["classic"].items() if k.startswith("errorbar.")}
SCATTER_CONFIG = {k: v for k, v in THEMES["classic"].items() if k.startswith("scatter.")}
