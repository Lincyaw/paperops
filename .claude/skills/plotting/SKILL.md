---
name: plotting
description: "Use this skill whenever creating matplotlib/seaborn figures for academic papers, research reports, or any publication-quality plotting task using paperops.plotting. Trigger when the user mentions 'plot', 'figure', 'chart', 'graph', 'visualization' in a research/paper context, or wants to create figures that will appear in a PDF paper or latex document."
---

# Academic Plotting with paperops

Generate publication-quality figures for top-venue papers (ACM, IEEE, NeurIPS, ICML, etc.) using matplotlib/seaborn with paperops presets.

## Setup

```python
from paperops.plotting import (
    apply_plot_config, figure, save, colors,
    MARKER_STYLES, LINE_STYLES, HATCH_PATTERNS,
)
import matplotlib.pyplot as plt
import numpy as np

apply_plot_config("classic")  # "classic", "modern", or "grayscale"
```

| Theme | Font | Grid | Use case |
|-------|------|------|----------|
| `"classic"` | Serif | Y-axis dotted | ACM/IEEE, traditional venues |
| `"modern"` | Sans-serif | None | CS/ML venues, clean look |
| `"grayscale"` | Serif | Y-axis dotted | B&W printing, supplementary |

## Utilities

```python
fig, ax = figure("single")              # 3.5 x 2.5 — single column
fig, ax = figure("double")              # 7.0 x 3.5 — full width
fig, ax = figure("square")              # 3.5 x 3.5 — heatmap, confusion matrix
fig, ax = figure("wide")               # 7.0 x 2.5 — timeline, wide panel
fig, axes = figure("double", ncols=2)   # subplots
fig, ax = figure((4.0, 3.0))           # custom (w, h) in inches

path = save(fig, "results.pdf")         # tight_layout + save + close, returns path
path = save(fig)                        # save to temp PNG, returns path

c = colors()     # full palette (list of hex strings)
c = colors(4)    # first 4 colors
```

`save()` returns the file path — pass it directly to `paperops.slides.Image(path=...)` when embedding figures into presentations.

---

## Information Hierarchy in Figures

A paper figure is read in layers. Design each layer deliberately.

### Layer 1 — Instant read (< 2 seconds)

What the reader grasps at a glance. This is the **only layer most readers will engage with**.

- **Who wins** — which method/condition is best, and by how much
- **Trend direction** — going up, going down, inflection point
- **The one claim** this figure supports

How to amplify Layer 1:
- Use color contrast: "Ours" in a saturated color, baselines in muted/gray
- Use line weight: "Ours" at 2.5pt, baselines at 1.0pt
- Use spatial position: the best result should be visually highest/rightmost
- Add a reference line if it anchors the message (e.g., "human performance")

### Layer 2 — Careful read (5–10 seconds)

Available for readers who look closer. Must be accessible but must not compete with Layer 1.

- Specific values and rankings among all methods
- Error bars / confidence intervals — credibility signal
- Axis labels and units — what was measured and how
- Scale and sample size

How to support Layer 2:
- Clear axis labels with units (`Latency (ms)`, not just `Latency`)
- Error bars on every experimental data point
- Readable tick labels (not overlapping, not rotated 90° unless necessary)
- Legend that doesn't obscure data

### Layer 3 — Context from the paper

This information belongs in the caption or body text, **not in the figure**:

- Why a method wins (analysis)
- Experimental setup details (dataset, hyperparameters, hardware)
- Statistical test results (p-values, significance)
- Cross-references to other figures

### What to remove

Strip anything that doesn't serve Layer 1 or Layer 2:

- Top and right spines (already removed by themes)
- Dense grid lines — at most a subtle Y-axis dotted grid
- `ax.set_title()` — the caption handles this in papers
- Excessive value labels on every bar — the bar height IS the information
- 3D effects, shadows, gradients — never
- Redundant encodings — if position already distinguishes series, don't also vary marker shape

---

## Chart Type Selection

| Data relationship | Recommended | Avoid |
|------------------|-------------|-------|
| Method comparison (few methods × few metrics) | Grouped bar chart | Pie chart |
| Performance vs. parameter / scale | Line chart + markers | Bar chart (implies categories) |
| Distribution / variance | Box plot, violin plot | Bar chart with only mean |
| Correlation between two variables | Scatter plot | 3D plots |
| Ablation study | Grouped bar or table | Line chart (implies ordering) |
| Training curve / time series | Line chart (shaded CI band) | Dense scatter |
| Proportion / composition | Stacked bar chart | Pie chart |
| High-dimensional comparison | Radar/spider chart | Many overlapping bars |
| Matrix data (confusion, attention) | Heatmap | 3D surface |

### When to use a table instead of a figure

- Differences between methods are tiny (< 2%) — a figure amplifies visual differences that may not be meaningful
- You need to show exact numbers with many decimal places
- You have too many conditions to fit legibly in one figure (> 8 groups)

---

## Visual Rules

### Hard constraints

These are non-negotiable for top venues:

1. **Font embedding**: Type 42 fonts in PDF/PS (handled by themes)
2. **Minimum font size**: All text ≥ 8pt after scaling to column width (3.5")
3. **Color-blind safety**: Always use the provided palette (`colors()`). For line charts, **combine color + marker + linestyle** — the figure must be interpretable without color
4. **Vector output**: Save as PDF for the paper. Use PNG only for slides or supplementary
5. **Bar chart Y-axis starts at 0**: Always. No exceptions
6. **Error bars on experimental data**: Mean ± std across runs, or confidence intervals. Report n (number of runs)
7. **No figure titles**: Papers use `\caption{}`, not `ax.set_title()`
8. **Axis labels must include units**: `Throughput (req/s)`, `Memory (GB)`, `Time (s)`

### Distinguishing series without color

Every line/bar chart must be readable in grayscale. Use this progression:

```python
for i, (label, data) in enumerate(series):
    ax.plot(x, data,
            marker=MARKER_STYLES[i],
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            label=label)
```

For bar charts in grayscale contexts, add hatch patterns:

```python
for i, (label, values) in enumerate(groups):
    ax.bar(x + i * width, values, width,
           label=label, hatch=HATCH_PATTERNS[i])
```

### Highlighting "Ours"

A common and effective pattern: make your method visually dominant.

```python
ours_color = colors()[0]        # saturated color for "Ours"
baseline_color = "#AAAAAA"      # gray for baselines

# Line chart: thick line for ours, thin for baselines
ax.plot(x, ours_data, color=ours_color, linewidth=2.5, marker='o',
        label="Ours", zorder=10)
for i, (name, data) in enumerate(baselines):
    ax.plot(x, data, color=baseline_color, linewidth=1.0,
            marker=MARKER_STYLES[i + 1],
            linestyle=LINE_STYLES[i + 1], label=name)
```

---

## Statistical Rigor

- **Always run multiple times** and report mean ± std (or median + IQR for skewed data)
- **Error bars are mandatory** for any experimental measurement — a bar or dot without error bars claims unrealistic precision
- **Don't cherry-pick runs** — report aggregate statistics
- When differences between methods are small relative to variance, acknowledge this visually (overlapping error bars tell the story honestly)
- Consider whether the visual difference is statistically meaningful — if error bars overlap substantially, a table with significance tests may be more honest than a figure

---

## Layout & Typography

### Consistent sizing across a paper

All figures in one paper should use the same `apply_plot_config()` call and the same width:

```python
# At the top of your plotting script
apply_plot_config("classic")

# All single-column figures
fig, ax = figure("single")   # 3.5" wide

# All full-width figures
fig, ax = figure("double")   # 7.0" wide
```

### Subplot labeling

Multi-panel figures use (a), (b), (c) labels in the top-left corner:

```python
fig, axes = figure("double", ncols=3)
for i, ax in enumerate(axes):
    label = chr(ord('a') + i)
    ax.text(-0.15, 1.05, f"({label})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")
```

### Legend placement

Priority order:
1. Inside the plot in an empty region — `ax.legend(loc="lower right")`
2. Below the plot — `fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=N)`
3. To the right of the plot (for many series) — adjust figure width

The legend must **never** overlap data points.

### Tick labels

- Avoid rotation when possible — shorten labels instead
- If rotation is necessary, use 45° (not 90°) with `ha="right"`
- For large numbers, use SI prefixes or scientific notation

---

## Judgment Calls

These depend on context. Apply these heuristics:

| Situation | Guideline |
|-----------|-----------|
| **Value labels on bars** | Only add when: (1) exact numbers matter more than visual comparison, or (2) differences are too small to see. Otherwise omit — the bar height is the information |
| **Reference / baseline line** | Add when it anchors interpretation (e.g., "random guess = 50%", "human = 0.95"). Max 1–2 per figure |
| **Truncated Y-axis** | Acceptable for line charts when you need to show small differences, but: add a break marker (~//) and mention in caption. Never truncate bar charts |
| **Log scale** | Use when data spans > 1 order of magnitude. Always label clearly: "Latency (ms, log scale)" |
| **Subplot density** | ≤ 4 subplots per figure. Beyond that, split into separate figures |
| **Annotation arrows** | Use sparingly to draw attention to a specific data point. Max 1–2 per figure |

---

## Self-Check Checklist

Run through this after generating every figure:

- [ ] Shrink the figure to 3.5" wide on screen — is all text still readable?
- [ ] Convert to grayscale mentally — can you still distinguish every series?
- [ ] Does the bar chart Y-axis start at 0?
- [ ] Does every experimental data series have error bars?
- [ ] Is there a legend, and does it NOT overlap any data?
- [ ] Do axis labels include units?
- [ ] Is the figure saved as PDF (for paper) or PNG (for slides)?
- [ ] Is there no `ax.set_title()`? (caption is in LaTeX, not in the figure)
- [ ] Does the figure support exactly one claim? If it tries to say two things, split it
