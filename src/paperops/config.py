# -*- coding: utf-8 -*-
# Comprehensive plotting configuration for top-tier software engineering academic conferences.
# This configuration aims to provide a ready-to-use setup for creating publication-quality figures
# using Matplotlib, following best practices for clarity, readability, and aesthetics.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- General Style ---
# Use seaborn for a base style, which is generally more aesthetically pleasing than default matplotlib.
# 'paper' context is suitable for academic papers.
# 'ticks' style adds ticks on axes for better value tracking.
sns.set_context("paper")
sns.set_style("ticks")

# --- Font Configuration ---
# Using Times New Roman, as it's a classic and widely accepted font for academic publications (e.g., ACM, IEEE).
# If not available, common fallbacks like 'serif' are used.
# The font sizes are chosen for readability in a two-column paper format.
# Ensure Type 42 (TrueType) fonts are used in PDF/PS output for ACM/IEEE compliance.
FONT_CONFIG = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "pdf.fonttype": 42,  # Embed fonts in PDF
    "ps.fonttype": 42,  # Embed fonts in PS
}

# --- Color Palettes ---
# A colorblind-friendly palette is crucial for accessibility.
# This palette is derived from Paul Tol's notes and is well-regarded in the scientific community.
# It provides good contrast and is distinguishable by people with common forms of color blindness.

# Base 6 colors

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


# Keep original 6-color palette for backward compatibility
COLOR_PALETTE_QUALITATIVE = BASE_COLORS

# Grayscale palette for black-and-white publications.
GRAYSCALE_PALETTE = sns.color_palette("gray_r", n_colors=5)

# Recommended colormaps for heatmaps (perceptually uniform and colorblind-friendly).
# - 'viridis', 'plasma', 'inferno', 'magma', 'cividis' for sequential data.
# - 'coolwarm', 'bwr', 'seismic' for diverging data.
RECOMMENDED_CMAPS = {
    "sequential": "viridis",
    "diverging": "coolwarm",
}


# --- Figure and Axes Configuration ---
# These settings control the layout and appearance of the figure and axes.
FIGURE_CONFIG = {
    # Figure layout
    "figure.dpi": 300,  # High resolution for publication
    "figure.autolayout": False,  # Use tight_layout() manually for better control
    "savefig.dpi": 300,
    "savefig.format": "pdf",  # Vector format for scalability
    "savefig.bbox": "tight",  # Fit the saved figure tightly around the plot
    # Axes appearance
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "y",  # Horizontal grid lines often aid in reading values
    "grid.color": "gray",
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,  # Remove top and right spines for a cleaner look
    "axes.spines.right": False,
}

# --- Tick Configuration ---
# Controls the appearance of axis ticks.
TICK_CONFIG = {
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.major.size": 3,
    "ytick.minor.size": 1.5,
}

# --- Legend Configuration ---
# Settings for the plot legend.
LEGEND_CONFIG = {
    "legend.frameon": False,  # No frame around the legend for a cleaner look
    "legend.loc": "best",
}

# --- Bar Plot Specifics ---
# Settings for bar plots, including hatch patterns for black-and-white printing.
BAR_CONFIG = {
    "hatch.linewidth": 0.5,
}
HATCH_PATTERNS = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]

# --- Line Plot Specifics ---
# Define a cycle of markers and line styles for multi-line plots to ensure distinguishability.
LINE_CONFIG = {
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
}
MARKER_STYLES = ["o", "s", "v", "^", "D", "<", ">", "p", "*"]
LINE_STYLES = ["-", "--", "-.", ":"]

# --- Box Plot Specifics ---
# Enhance visibility of box plot components.
BOXPLOT_CONFIG = {
    "boxplot.boxprops.linewidth": 1.5,
    "boxplot.medianprops.linewidth": 1.5,
    "boxplot.whiskerprops.linewidth": 1.5,
    "boxplot.capprops.linewidth": 1.5,
    "boxplot.flierprops.markeredgecolor": "gray",
    "boxplot.flierprops.marker": ".",
}

# --- Error Bar Specifics ---
ERRORBAR_CONFIG = {
    "errorbar.capsize": 2,
}

# --- Scatter Plot Specifics ---
SCATTER_CONFIG = {
    "scatter.marker": "o",
}


def apply_plot_config(palette="color"):
    """
    Applies the comprehensive plotting configuration.
    Args:
        palette (str): The color palette to use. Options:
            - 'color': 6-color colorblind-friendly palette
            - '18tones': 18-color palette with 3 tones per base color
            - 'gray': grayscale palette
            - or any custom palette
    """
    plt.rcParams.update(FONT_CONFIG)
    plt.rcParams.update(FIGURE_CONFIG)
    plt.rcParams.update(TICK_CONFIG)
    plt.rcParams.update(LEGEND_CONFIG)
    plt.rcParams.update(BAR_CONFIG)
    plt.rcParams.update(LINE_CONFIG)
    plt.rcParams.update(BOXPLOT_CONFIG)
    plt.rcParams.update(ERRORBAR_CONFIG)
    plt.rcParams.update(SCATTER_CONFIG)

    if palette == "color":
        sns.set_palette(COLOR_PALETTE_QUALITATIVE)
    elif palette == "gray":
        sns.set_palette(GRAYSCALE_PALETTE)
    else:
        sns.set_palette(palette)

    print(
        f"Plotting configuration for academic publications applied (Palette: {palette})."
    )


# Example usage:
if __name__ == "__main__":
    apply_plot_config()

    # --- Example 1: Bar Plot ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = ["Group A", "Group B", "Group C", "Group D"]
    y1 = [20, 35, 30, 35]
    y2 = [25, 32, 34, 20]
    ax.bar(x, y1, label="Metric 1", width=0.4)
    ax.bar(x, y2, label="Metric 2", width=0.4, bottom=y1)
    ax.set_xlabel("Experiment Groups")
    ax.set_ylabel("Measured Value")
    ax.set_title("Example Bar Plot")
    ax.legend()
    ax.set_ylim(0)
    plt.tight_layout()
    fig.savefig("example_bar_plot.pdf")
    print("Example bar plot saved to example_bar_plot.pdf")
    plt.close(fig)

    # --- Example 2: Line Plot ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.linspace(0, 10, 10)
    for i in range(4):
        ax.plot(
            x,
            x + i,
            marker=MARKER_STYLES[i],
            linestyle=LINE_STYLES[i],
            label=f"Line {i + 1}",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Performance Metric")
    ax.set_title("Example Line Plot")
    ax.legend()
    plt.tight_layout()
    fig.savefig("example_line_plot.pdf")
    print("Example line plot saved to example_line_plot.pdf")
    plt.close(fig)

    # --- Example 3: Box Plot ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    data = [np.random.normal(0, std, 100) for std in range(1, 5)]
    sns.boxplot(data=data, ax=ax)
    ax.set_xlabel("Distribution")
    ax.set_ylabel("Value")
    ax.set_title("Example Box Plot")
    plt.tight_layout()
    fig.savefig("example_box_plot.pdf")
    print("Example box plot saved to example_box_plot.pdf")
    plt.close(fig)

    # --- Example 4: Grayscale Plot ---
    apply_plot_config(palette="gray")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for i in range(4):
        ax.bar(f"Bar {i + 1}", i + 2, label=f"Bar {i + 1}", hatch=HATCH_PATTERNS[i])
    ax.set_xlabel("Categories")
    ax.set_ylabel("Value")
    ax.set_title("Example Grayscale Bar Plot with Hatches")
    ax.legend()
    plt.tight_layout()
    fig.savefig("example_grayscale_plot.pdf")
    print("Example grayscale plot saved to example_grayscale_plot.pdf")
    plt.close(fig)
