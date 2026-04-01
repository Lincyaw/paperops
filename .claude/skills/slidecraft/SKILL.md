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

A good presentation is not a document read aloud — it is a **narrative arc** delivered visually. These principles apply to every presentation.

### 0. Narrative Coherence Across the Entire Deck

Before touching any individual slide, design the **story** first. Every slide must earn its place in a logical chain: each one should answer "why am I here after the previous slide?" and "what does the audience need to know before the next slide?" If you can shuffle slides without the audience noticing, the narrative is broken.

Think in terms of:
- **Setup** (context, problem, motivation) → **Development** (evidence, analysis, method) → **Resolution** (findings, implications, call to action)
- Each slide advances the argument by exactly one step — no more, no less
- Transitions between slides should feel inevitable, not arbitrary
- Section dividers (`prs.section()`) mark major narrative beats, not arbitrary groupings

Plan the full slide sequence and its logical flow before writing any code. If the structure doesn't hold together as a story, no amount of visual polish will save it.

### 1. Every Slide Must Have Animation

Never present an entire slide at once. The audience will read ahead and stop listening to you. Use `sb.animate()` on **every content slide** to control pacing:

- Group by logical progression: premise → evidence → conclusion
- One click = one idea appears
- The presenter controls attention, not the audience's reading speed

The only exceptions are cover slides, section dividers, and closing slides — these are simple enough to show all at once.

```python
# Good: audience follows your reasoning step by step
sb.animate([
    [problem_statement],      # Click 1: "Here's the problem"
    [data_chart],             # Click 2: "Here's what we measured"
    [conclusion_callout],     # Click 3: "Here's what it means"
])
```

### 2. Visual Structure Over Text Walls

Every slide needs a **visual element** — not just text. If a slide is just TextBlocks and BulletLists, it belongs in a document, not a presentation. Express ideas through structure:

- **Processes/sequences** → `Flow`, `Flowchart`, or custom Arrow chains
- **Comparisons** → Side-by-side `HStack` with contrasting colors
- **Data** → `BarChart`, `RadarChart`, or `Table`
- **Key insights** → `Callout` with accent color
- **Hierarchies/relationships** → `Grid` of `RoundedBox` with `Arrow` connectors

**Rendering priority for visual content:**
1. **Native shapes first** — Box, RoundedBox, Circle, Arrow, Flowchart, Flow, Callout all render as real PPT shapes. They scale perfectly, stay crisp, and the audience can select/copy text from them.
2. **SvgCanvas for custom diagrams** — When native shapes can't express the visual (e.g., curved paths, custom geometries, complex spatial layouts), use `SvgCanvas` to draw it, then embed via `SvgImage(svg=canvas)`. This converts SVG→PNG at render time.
3. **BarChart/RadarChart** — These use SvgCanvas internally. Use them for data visualization.

Avoid SVG when native shapes suffice — SVG images can distort if the allocated region aspect ratio doesn't match.

### 3. Emphasis for Attention Guidance

Use `italic=True` and `color` to highlight key content. Color-match emphasis to meaning:
- `"negative"` (red) for warnings/problems
- `"positive"` (green) for results/solutions
- `"primary"` (blue) for key insights
- `"highlight"` (purple) for theoretical concepts

On every slide with >2 TextBlocks, identify the single most important phrase and give it `italic=True` + semantic color.

### 4. Layout Variety

Never repeat the same layout pattern on consecutive slides. Mix:
- HStack + VStack combinations
- Grid for multi-item layouts
- Callout for highlighted insights
- Flow/Flowchart for processes
- BarChart/RadarChart for data
- Custom combinations

Prefer custom `slide.layout()` over template calls like `prs.comparison()` for non-trivial content.

### 5. Content Density

For a 30-minute talk: ~22-25 slides (1-1.5 min/slide). Merge related concepts into fewer, richer slides. Cut redundant slides that repeat what a visual already shows.

---

## QA Workflow

Always validate after generating:

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

# 3. Visual preview (PIL-based)
paths = prs.preview(output_dir="./preview")
# Then read the PNG files to visually verify layout
```

Fix any issues found, then re-validate. Don't declare success until a full pass reveals no new issues.

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
