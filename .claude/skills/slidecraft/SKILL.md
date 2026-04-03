---
name: slidecraft
description: "Use this skill whenever creating, modifying, or generating PowerPoint presentations (.pptx) using Python code with the paperops package. This includes: building slide decks programmatically, creating data visualizations in slides, laying out content with shapes/text/charts/tables, adding animations, theming presentations, or any task that involves paperops.slides. Trigger whenever the user mentions 'PPT,' 'slides,' 'presentation,' 'deck,' 'SlideCraft,' 'paperops,' or wants to generate a .pptx file using Python. If the user wants native PowerPoint shapes (not AI-generated images), this is the right skill."
---

# SlideCraft Toolkit

A Python toolkit for generating native PowerPoint presentations with declarative layouts, theme-aware styling, and click-to-advance animations. Install via `pip install paperops[slides]`.

## Quick Reference

| Task | How |
|------|-----|
| Create a presentation | `prs = Presentation(theme=themes.professional)` |
| Add a template slide | `prs.cover(...)`, `prs.content(...)`, `prs.comparison(...)` |
| Add a custom slide | `sb = prs.slide(title="...");  sb.layout(component)` |
| Add animations | `sb.animate([[group1_components], [group2_components]])` |
| Add speaker notes | `sb.notes("Speaker note text")` |
| Save | `prs.save("output.pptx")` |
| Validate layout | `report = prs.review()` |
| Preview to PNG | `paths = prs.preview(slides=[0, 2], output_dir="./preview")` |
| Check saved file | `from paperops.slides.preview import check_presentation; check_presentation("output.pptx")` |
| Integrated deck review | `prs.review_deck("output.pptx", render_preview=True, output_dir="./preview")` |

## Setup

Install: `pip install paperops[slides]` or `uv add paperops[slides]`

```python
from paperops.slides import (
    Presentation, themes, Direction, Align,
    HStack, VStack, Grid, Padding,
    Box, RoundedBox, Circle, Badge, Arrow, Line,
    TextBlock, BulletList, Table,
    Image, SvgImage,
    Callout, Flow,
    BarChart, RadarChart, Flowchart,
    LineChart, PieChart, HorizontalBarChart,
)
```

---

## Component Model

Everything is a **LayoutNode**. Nodes form a tree: containers hold children, leaves render content.

### Containers (layout)

| Container | Behavior | Key params |
|-----------|----------|------------|
| `HStack` | Left-to-right | `gap`, `children` |
| `VStack` | Top-to-bottom | `gap`, `children` |
| `Grid` | N-column grid | `cols`, `gap`, `children` |
| `Padding` | Inset wrapper | `child`, `all`/`left`/`right`/`top`/`bottom` |

### Leaves (content)

| Component | Purpose | Key params |
|-----------|---------|------------|
| `TextBlock` | Block of text | `text`, `font_size`, `color`, `bold`, `italic`, `align` |
| `BulletList` | Bullet points | `items` (strings or `(text, indent)` tuples) |
| `Table` | Data table | `headers`, `rows`, `header_color` |
| `Box` | Rectangle | `text`, `color`, `border`, `text_color`, `bold` |
| `RoundedBox` | Rounded rect | Same as Box + `radius` |
| `Circle` | Circle/oval | `text`, `color`, `text_color`, `radius` |
| `Badge` | Small label | `text`, `color`, `text_color` |
| `Arrow` | Connector arrow | `from_component`, `to_component`, `direction` |
| `Line` | Connector line | `from_component`, `to_component`, `dashed` |
| `Image` | Static image | `path` |
| `SvgImage` | SVG to PNG | `svg` (string or SvgCanvas), `scale` |
| `Callout` | Accent bar + title + body | `title`, `body`, `color` |
| `Flow` | Connected box chain | `labels`, `direction`, `colors` |
| `Flowchart` | Node-edge diagram | `nodes` (dict), `edges` (list), `direction` |
| `BarChart` | Grouped bar chart (SVG) | `groups`, `y_label`, `show_values` |
| `RadarChart` | Spider chart (SVG) | `dimensions`, `series` |
| `LineChart` | Multi-series line chart (SVG) | `series`, `x_labels`, `show_dots` |
| `PieChart` | Pie/donut chart (SVG) | `slices`, `donut`, `show_percentages` |
| `HorizontalBarChart` | Horizontal bars (SVG) | `items`, `show_values` |

All nodes accept `width`, `height`, `min_width`, `min_height` as keyword args to override sizing.

**For full API details including all parameters, data formats, and examples: read [references/api.md](references/api.md)**

**For the end-to-end production workflow (logic flow → visual vocabulary → layout → content): read [references/workflow.md](references/workflow.md)**

---

## Themes

Three built-in themes: `themes.professional`, `themes.minimal`, `themes.academic`.

Each theme defines:
- **Colors**: `primary`, `secondary`, `accent`, `positive`, `negative`, `highlight`, `warning`, `text`, `text_mid`, `text_light`, `bg`, `bg_alt`, `bg_accent`, `border`
- **Font sizes**: `title` (32pt), `subtitle` (24pt), `heading` (20pt), `body` (18pt), `caption` (14pt), `small` (11pt)
- **Font family**: Calibri (professional/minimal), Georgia (academic)

Colors can be specified as:
- Semantic name: `"primary"`, `"accent"`, `"text_mid"`
- Hex string: `"#FF6B35"`
- RGB tuple: `(255, 107, 53)`

Custom theme via `theme.override(colors={"primary": "#1E2761"}, fonts={"title": 36})`.

---

## Template Slides

Quick slides without manual layout:

```python
prs.cover("Title", subtitle="Subtitle", author="Author")
prs.section(1, "Part Title", subtitle="Subtitle")
prs.content("Slide Title", bullets=["Point 1", "Point 2", ("Sub-point", 1)])
prs.content("Slide Title", table=(["Col A", "Col B"], [["r1c1", "r1c2"], ["r2c1", "r2c2"]]))
prs.comparison("Title", left=("Left", ["a", "b"]), right=("Right", ["c", "d"]))
prs.quote("Quote text.", author="Author Name")
prs.transition("Bridge Text", sub_text="Optional subtext")
prs.end("Thank You", subtitle="Questions?")
```

---

## Custom Slides

For rich, varied layouts — **preferred over templates for interesting presentations**.

```python
sb = prs.slide(title="My Title", reference="Source: Paper 2024")

# Build a component tree
left_col = VStack(gap=0.2, children=[
    TextBlock(text="Key Insight", font_size="heading", bold=True, color="primary"),
    TextBlock(text="Supporting detail here.", font_size="body", italic=True),
])
right_col = Flowchart(
    nodes={"a": ("Input", "primary"), "b": ("Process", "secondary"), "c": ("Output", "positive")},
    edges=[("a", "b"), ("b", "c")],
    direction="down",
)
sb.layout(HStack(gap=0.5, children=[left_col, right_col]))

# Add animations: click 1 shows left, click 2 shows right
sb.animate([[left_col], [right_col]])
sb.notes("Explain the flow from input to output.")
```

---

## Animations

Click-to-advance: each group appears on one click (fade in). Later groups start hidden automatically.

```python
sb.animate([
    [component_a, component_b],   # Click 1: these appear together
    [component_c],                # Click 2: this appears
    [component_d, component_e],   # Click 3: these appear together
])
```

Group by logical progression (premise -> evidence -> conclusion). Every content slide with multiple groups should use animations.

---

## Design Philosophy

**A presentation is the flow of thinking, not the accumulation of content.**

Every slide deck is a chain of reasoning delivered visually. The audience should feel your logic unfold — premise by premise, step by step — not read a document projected on a wall. If a slide doesn't advance the argument, it doesn't belong. If a slide dumps information without guiding interpretation, it has failed.

This is the single principle behind every rule below.

## Review-Driven Workflow

For real deck work in this repo, do not stop at `prs.save(...)`.

Default loop:

1. Plan the deck's logic chain.
2. Build the deck in code.
3. Run `prs.review_deck(...)`.
4. Inspect the returned issue summary and preview PNGs.
5. Apply targeted fixes to content, layout, or sizing.
6. Rerun `prs.review_deck(...)` until the deck state is stable.

Use integrated review to decide whether a problem is:

- content density: too much visible text for the intended visual structure
- intrinsic sizing: text-bearing components are underestimating their natural width/height
- container negotiation: parent layout is compressing children too aggressively
- saved-file mismatch: preview and `.pptx` checks disagree

Trust intrinsic sizing for ordinary text-bearing components first. Add manual `width` / `height` only when the visual intent is genuinely fixed.

### 1. Design the Thinking Chain First

Before writing any code, design the **logical structure** of the entire deck. Every slide must answer two questions:

- **Why does this slide come after the previous one?** (logical dependency)
- **What must the audience understand here before the next slide?** (prerequisite)

If you can reorder slides without the audience noticing, there is no chain — just a pile of content.

Think in terms of logical progression:
- **Setup** (context → problem → why it matters) → **Development** (approach → evidence → analysis) → **Resolution** (findings → implications → what's next)
- Each slide = exactly one logical step. Not zero (padding), not two (overloaded).
- Section dividers (`prs.section()`) mark major turns in the argument — "we've established the problem, now here's our approach" — not arbitrary groupings.

**Plan the full slide sequence and its logical dependencies before touching any code.** The structure is the presentation. Everything else is rendering.

### 2. Animate the Reasoning, Not the Slide

Animation exists to pace the logic within a single slide. Each click reveals the next step in a local argument:

```python
# The slide's logic: "Here's a problem → here's evidence → here's what it means"
sb.animate([
    [problem_statement],      # Click 1: establish the claim
    [data_chart],             # Click 2: show the evidence
    [conclusion_callout],     # Click 3: deliver the interpretation
])
```

Rules:
- **Every content slide must have animation.** Showing everything at once lets the audience read ahead and stop listening. The presenter controls attention, not the audience's reading speed.
- **One click = one logical step.** Not one visual element — if two components together form one idea, they appear together.
- **Animation order = reasoning order.** Premise before evidence. Question before answer. Data before interpretation.
- Exceptions: cover slides, section dividers, and closing slides are simple enough to show all at once.

### 3. Encode Logic Through Structure, Not Text

The layout itself should express the relationship between ideas. If you need a sentence to explain how two things relate, you've chosen the wrong visual structure.

| Logical relationship | Visual encoding | NOT this |
|---------------------|----------------|----------|
| A causes/leads to B | `Flow`, `Arrow` chain | "A leads to B" in a bullet point |
| A vs B | Side-by-side `HStack`, contrasting colors | Two paragraphs describing each |
| Parts of a whole | `Grid` of `RoundedBox` | Bullet list of parts |
| Key insight | `Callout` with accent color | Bold text in a paragraph |
| Process / pipeline | `Flowchart` with directed edges | Numbered list of steps |
| Quantitative comparison | `BarChart`, `Table` | "X is 3x faster than Y" in text |
| Hierarchy / containment | Nested boxes, `Padding` | Indented bullets |

**The test**: if you removed all text from a slide and only saw shapes, colors, and arrows, could the audience still grasp the structure of the argument? If not, the visual structure isn't carrying enough weight.

For concrete guidance on when to use native shapes vs SVG icons vs charts, see [references/workflow.md](references/workflow.md) § Visual Vocabulary.

### 4. Minimize Text, Maximize Signal

Text on a slide is not prose — it is **anchor points** for spoken reasoning. The presenter's voice carries the explanation; the slide carries the structure.

Guidelines:
- **No full sentences.** If text reads like a paragraph, it belongs in a document.
- **Each TextBlock ≤ 15 words.** Beyond that, split into structure or cut.
- **BulletList items are keywords, not explanations.** "Latency reduced 3×" not "We observed that the latency was reduced by a factor of three compared to the baseline."
- **One emphasis per slide.** Identify the single most important phrase and give it `italic=True` + semantic color (`"primary"` for key insights, `"positive"` for results, `"negative"` for problems, `"highlight"` for concepts). Everything else is supporting context.

### 5. Every Slide Must Look Different

Repeating the same layout signals repetitive thinking. If three slides in a row are "title + bullet list," the audience assumes the content is equally uniform and disengages.

Mix layouts deliberately:
- `HStack` (side-by-side) → `VStack` (top-down) → `Grid` → `Flowchart` → `HStack` with `Callout`
- Prefer custom `slide.layout()` over template calls like `prs.content()` for anything beyond the simplest slides.
- When two slides share the same logical structure (e.g., comparing results for dataset A then dataset B), the same layout is acceptable — the repetition reinforces the parallel.

### 6. Density and Pacing

- **Target: 1–1.5 minutes per slide** for a typical talk. A 30-minute presentation ≈ 22–25 slides.
- **Merge related concepts** into fewer, richer slides rather than spreading thin ideas across many slides.
- **Cut slides that don't advance the argument.** A slide that only repeats what a visual already showed, or that exists only "for completeness," weakens the chain.
- **Breathing room matters.** Not every slide needs to be dense. A single `Callout` with one sentence can be the most powerful slide in the deck — if it's the right sentence at the right moment in the argument.

---

## QA Workflow

Two levels of validation: automated checks and visual review.

### Automated checks (always run)

```python
# 1. In-memory validation
report = prs.review()
print(f"Issues: {report['total_issues']}")
for iss in report['issues']:
    print(f"  [{iss['type']}] {iss['detail']}")

# 2. Post-save file check
from paperops.slides.preview import check_presentation
issues = check_presentation("output.pptx")
# Checks: off-slide shapes, text overflow, overlapping text

# 3. Generate preview PNGs
paths = prs.preview(output_dir="./preview")
```

### Visual review (delegate to `slide-reviewer` agent)

After generating preview PNGs, dispatch the **`slide-reviewer`** agent with:
- The `.pptx` file path
- The preview PNG paths
- The slide outline from Phase 1 (if available)

The reviewer examines every slide for layout issues, design principle violations, and logic flow coherence. Fix its findings, then re-run the automated checks.

Don't declare success until both automated checks and visual review pass cleanly.

---

## Complete Example

```python
from paperops.slides import (
    Presentation, themes,
    HStack, VStack, Grid, Padding,
    RoundedBox, Circle, Badge, Arrow,
    TextBlock, BulletList, Table,
    Callout, Flow, Flowchart, BarChart,
)

prs = Presentation(theme=themes.professional)

# Cover
prs.cover("Research Findings", subtitle="Q1 2026 Analysis", author="Team")

# Custom insight slide
sb = prs.slide(title="Key Finding")
insight = Callout(title="Discovery", body="Performance improved 3x after optimization.", color="positive")
detail = TextBlock(text="This was measured across 1000 trials with p < 0.01.", font_size="caption", italic=True, color="text_mid")
chart = BarChart(groups=[
    ("Before", [("Latency", 450, "negative")]),
    ("After", [("Latency", 150, "positive")]),
], y_label="ms")
left = VStack(gap=0.3, children=[insight, detail])
sb.layout(HStack(gap=0.5, children=[left, chart]))
sb.animate([[left], [chart]])
sb.notes("Highlight the 3x improvement, then show the chart.")

# Process slide
sb2 = prs.slide(title="Our Approach")
flow = Flowchart(
    nodes={"d": ("Data Collection", "primary"), "a": ("Analysis", "secondary"),
           "v": ("Validation", "accent"), "r": ("Report", "positive")},
    edges=[("d", "a"), ("a", "v"), ("v", "r")],
)
sb2.layout(flow)

prs.end("Thank You", subtitle="Questions?")
prs.save("output/research.pptx")

report = prs.review()
print(f"Slides: {report['total_slides']}, Issues: {report['total_issues']}")
```
