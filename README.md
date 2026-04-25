# PaperOps

![Version](https://img.shields.io/badge/version-0.3.0-blue)
![Python](https://img.shields.io/badge/python-3.13+-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

PaperOps is a Python toolkit for two adjacent jobs:
- `paperops.plotting` for publication-quality academic figures
- `paperops.slides` for IR-first PowerPoint generation aimed at human authors and LLM agents

## Install

```bash
uv pip install -e ".[all]"
```

## Slides: 5-minute quickstart

The new `paperops.slides` flow is IR-first: JSON, MDX, and Python all compile into the same slide IR before style resolution, layout, autofit, and PPTX generation.

### Python

```python
from paperops.slides.dsl import Deck, Grid, Heading, KPI, Slide, Subtitle, Title

deck = Deck(theme="minimal", sheet="keynote")

deck += Slide(class_="cover")[
    Title("World-model RCA"),
    Subtitle("IR-first authoring in five minutes"),
]

deck += Slide()[
    Heading("Key metrics"),
    Grid(style={"cols": "1fr 1fr 1fr", "gap": "lg"})[
        KPI(label="Cases", value="9,152", delta="validated", trend="positive"),
        KPI(label="Signals", value="3", delta="metrics/logs/traces", trend="neutral"),
        KPI(label="Goal", value="Trust", delta="faithful reasoning", trend="positive"),
    ],
]

deck.render("quickstart_python.pptx")
```

### JSON

```json
{
  "$schema": "paperops-slide-1.0",
  "theme": "minimal",
  "sheet": "seminar",
  "slides": [
    {
      "type": "slide",
      "class": "cover",
      "children": [
        {"type": "title", "text": "World-model RCA"},
        {"type": "subtitle", "text": "One IR, many authoring fronts"}
      ]
    },
    {
      "type": "slide",
      "children": [
        {"type": "heading", "text": "Why IR-first"},
        {
          "type": "grid",
          "style": {"cols": "1fr 1fr", "gap": "md"},
          "children": [
            {"type": "text", "class": "card", "text": "Styles live in sheets or styles, not scattered inline."},
            {"type": "text", "class": "card", "text": "The same content can render as seminar, keynote, or whitepaper."}
          ]
        }
      ]
    }
  ]
}
```

Render it with:

```bash
uv run python - <<'PY'
from paperops.slides import render_json
render_json("quickstart_slides.json", out="quickstart_json.pptx")
PY
```

### MDX

```mdx
---
theme: minimal
sheet: keynote
---

# World-model RCA {.cover}

<Subtitle>Author once, render many variants</Subtitle>

---

## Why this API

<Grid cols="1fr 1fr" gap="md">
  <Text class="card">Use classes, sheets, and style keys instead of per-component absolute coordinates.</Text>
  <Text class="card">Switch `sheet` to produce seminar, keynote, whitepaper, or pitch variants.</Text>
</Grid>
```

Render it with:

```bash
uv run python - <<'PY'
from paperops.slides import render_json
render_json("quickstart_slides.mdx", out="quickstart_mdx.pptx")
PY
```

### Which front end should you choose?

- MDX: best default for LLM-authored talk decks with prose + components
- JSON: best for structured, tool-generated content
- Python: best when slide content is produced programmatically from data or pipelines

See `docs/quickstart-slides.md` for the full walkthrough, sheet guidance, and an LLM prompt fragment.

## Example decks

- `uv run python examples/4.15-talk/presentation.py`
- `uv run python examples/4.15-talk/pptx_restructure.py`
- `uv run python examples/fse26/presentation.py`
- `uv run python examples/communication_friction/build.py`
- `uv run python examples/self-intro/build.py`
- `uv run python examples/gallery/render_gallery_variants.py`

## Plotting

PaperOps also includes publication-ready plotting helpers for academic workflows.

```python
import pandas as pd
from paperops.core import AcademicPlotter

plotter = AcademicPlotter(layout="single", size="medium", color_scheme="nature")
data = pd.DataFrame({"x": [1, 2, 3], "method_a": [0.62, 0.71, 0.8], "method_b": [0.58, 0.69, 0.77]})
plotter.line_plot(
    data=data,
    x="x",
    y=["method_a", "method_b"],
    fig_name="Performance comparison",
    xlabel="Step",
    ylabel="Score",
    save_path="comparison.pdf",
)
```

## Development

- Run slides/plotting tests with `make verify`
- Use `uv run python` and `uv run pytest`
- Keep `.claude/skills/` and `.codex/skills/` aligned when editing mirrored skills
