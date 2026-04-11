---
name: plotting
description: "Use this skill whenever creating matplotlib/seaborn figures for academic papers, research reports, or publication-quality plotting tasks using paperops.plotting. Trigger when the user mentions plots, figures, charts, graphs, or visualizations in a research/paper context."
---

# Plotting

Use `paperops.plotting` to build publication-quality figures for papers and research reports.

## Use This Skill When

- the task is to create or revise figures that belong in a paper, report, poster, or appendix
- the user needs chart selection, figure hierarchy, or publication-oriented styling guidance
- a slide needs a real research figure rather than a schematic diagram

## Do Not Use This Skill When

- the task is to define a presentation-wide visual language or symbol system
- the task is to build the PPT layout itself
- the task is to review a rendered deck for clipping or layout issues

## Workflow

1. Lock the figure claim
   - identify the one conclusion the figure must support
   - choose the chart type that best exposes that conclusion
2. Set the figure system
   - call `apply_plot_config(...)`
   - choose a consistent size with `figure(...)`
   - use `colors()`, markers, line styles, and hatches deliberately
3. Build for layered reading
   - make the main result visible at a glance
   - keep exact values, error bars, and units readable without competing with the main claim
4. Enforce academic constraints
   - axis labels include units
   - bar charts start at zero
   - experimental data includes uncertainty where applicable
   - save vector output for papers and raster output only when the destination requires it
5. Handoff outputs
   - return figure file paths that can be embedded into `paperops.slides.Image(path=...)`

## Handoff

- if the figure is for a deck and the deck layout is the next task, hand off the saved asset to `slidecraft`
- if the user really needs a diagrammatic slide visual rather than a quantitative chart, hand off to `visual-language` or `slidecraft`
- if the user asks for codebase-wide checks after figure-related changes, hand off to `verify`

## References

- prefer the repo README and plotting API docs for exact function signatures
- keep the skill focused on figure decisions, not full API duplication
