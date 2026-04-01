# Workflow: From Idea to Deck

Building a presentation is a four-phase process. Do not skip phases or do them out of order — each phase depends on the output of the previous one.

```
Phase 1: Logic Flow     → What does each slide argue?
Phase 2: Visual Vocab   → What recurring concepts need icons/identifiers?
Phase 3: Layout         → How is each slide spatially structured?
Phase 4: Content Fill   → What specific text, data, and components go where?
```

---

## Phase 1: Logic Flow

**Goal**: Define the chain of reasoning as an ordered list of slides, each with a one-line claim.

Before writing any code, produce a slide outline. Each entry has:
- A **slide number**
- A **one-line claim** — the single thing this slide argues or establishes
- A **logical link** — why it follows from the previous slide

Example:

```
1. [Cover] "Optimizing LLM Inference at Scale"
2. [Context] LLM deployment is bottlenecked by inference cost → audience understands the problem space
3. [Problem] Current batching strategies waste 40% of GPU cycles → quantifies the pain
4. [Section] Our Approach
5. [Key Idea] Predictive scheduling can fill idle GPU slots → introduces our core insight
6. [Method] Three-phase pipeline: predict → schedule → execute → shows how it works
7. [Experiment] Evaluated on 3 production workloads → grounds the claim in evidence
8. [Results] 2.7× throughput improvement, 35% cost reduction → delivers the payoff
9. [Ablation] Each phase contributes; removing prediction loses 60% of gains → proves the design
10. [End] Summary + open questions
```

**Validation**: Read the claims top-to-bottom. Does each one follow logically? Could you tell the story by reading just the claims aloud? If a slide feels out of place, move or cut it.

---

## Phase 2: Visual Vocabulary

**Goal**: Identify concepts that recur across multiple slides and design a consistent visual identity for each.

Scan the logic flow from Phase 1. Any concept that appears on ≥ 2 slides deserves a visual identifier — a consistent color, shape, or icon that the audience learns once and recognizes throughout.

### When to use native shapes vs SVG icons

| Situation | Use | Why |
|-----------|-----|-----|
| Abstract concepts (data, model, user) | **SVG icon** via `SvgCanvas` → `SvgImage` | An icon is instantly recognizable; a box with text "Data" requires reading |
| Structural elements (containers, groups) | **Native shapes** (`RoundedBox`, `Box`) | These are the frame, not the content — they should be invisible scaffolding |
| Status/category labels | **`Badge`** or **`Circle`** with color | Color + short text is enough for categorical distinction |
| Relationships (causes, feeds into) | **`Arrow`**, **`Line`**, **`Flow`** | Native connectors are crisp and semantically clear |
| Processes with ≥ 3 steps | **`Flowchart`** or **`Flow`** | Native, auto-laid-out, text stays selectable |
| Quantitative data | **Built-in charts** (`BarChart`, etc.) | Simple exhibition in slides; use matplotlib via `Image(path=save(fig))` for publication-quality figures |

### Designing SVG icons

When a concept needs an icon (e.g., "GPU", "database", "user", "cloud"), build it as a reusable `SvgCanvas` function:

```python
from paperops.slides import SvgImage
from paperops.slides.components.svg_canvas import SvgCanvas

def icon_gpu(theme, size=80):
    """GPU icon — consistent across all slides."""
    s = SvgCanvas(size, size, theme=theme)
    # Chip body
    s.rounded_rect(15, 20, 50, 40, fill="primary", stroke="none", rx=6)
    # Pins
    for y in [25, 35, 45]:
        s.rect(5, y, 10, 4, fill="primary", stroke="none")
        s.rect(65, y, 10, 4, fill="primary", stroke="none")
    # Label
    s.text(40, 42, "GPU", color="white", size=12, bold=True)
    return s

def icon_database(theme, size=80):
    """Database icon."""
    s = SvgCanvas(size, size, theme=theme)
    s.path("M 20 25 Q 40 15 60 25 Q 40 35 20 25 Z", fill="secondary", stroke="none")
    s.rect(20, 25, 40, 30, fill="secondary", stroke="none")
    s.path("M 20 55 Q 40 65 60 55", stroke="secondary", fill="none", stroke_width=2)
    return s

# Use in slides:
sb.layout(HStack(gap=0.3, children=[
    SvgImage(svg=icon_gpu(prs._theme), width=0.8, height=0.8),
    TextBlock(text="GPU Cluster", font_size="body", bold=True),
]))
```

**Key rule**: Define all icons before building any slides. They form the visual vocabulary the audience learns.

### Color as identity

Assign a consistent semantic color to each major concept or actor throughout the deck:

```python
# Example color assignments for an ML systems talk:
# "primary"   → Our system / method
# "secondary" → Baseline / comparison
# "accent"    → Infrastructure (GPU, network)
# "positive"  → Results / improvements
# "negative"  → Problems / bottlenecks
# "highlight" → Theoretical concepts
```

When the audience sees `"primary"` blue on slide 8, they should already associate it with "our method" from slide 5. Don't reassign colors to different meanings mid-deck.

---

## Phase 3: Layout

**Goal**: For each slide, decide the spatial structure before filling in content.

Work through each slide from the Phase 1 outline and decide:
1. **What components** does this slide need? (text blocks, charts, flowcharts, icons, images)
2. **What is the logical grouping?** Which components form one idea that appears together?
3. **What container** expresses the relationship? (HStack for comparison, VStack for sequence, Grid for enumeration)
4. **What is the animation order?** Which group appears first, second, third?

Write this as a structural sketch before coding:

```
Slide 6 "Three-phase pipeline"
├── Title: "Predictive Scheduling Pipeline"
├── Animation group 1:
│   └── HStack
│       ├── [GPU icon] + "Predict" (primary)
│       ├── Arrow →
│       ├── [Clock icon] + "Schedule" (secondary)
│       ├── Arrow →
│       └── [Gear icon] + "Execute" (accent)
├── Animation group 2:
│   └── Callout: "Key: prediction happens during previous batch's execution"
└── Notes: "Walk through each phase, then reveal the pipelining insight"
```

### Layout patterns by slide purpose

| Slide purpose | Typical structure |
|--------------|-------------------|
| Single key insight | `Callout` (centered or left) + supporting `TextBlock` |
| Process / pipeline | `Flow` or `Flowchart` (full width) |
| Comparison (A vs B) | `HStack` with two `VStack` columns, contrasting colors |
| Evidence / data | `HStack`: left = text context, right = chart |
| Enumeration (3–6 items) | `Grid` of `RoundedBox` with icons |
| Architecture diagram | `SvgCanvas` (complex) or nested `HStack`/`VStack` with `Arrow` (simple) |
| Result highlight | Big number `TextBlock` (heading size) + `Callout` for context |

---

## Phase 4: Content Fill

**Goal**: Implement the slides in code, filling in text, data, and components.

By this phase, every decision has been made. You are translating the structural sketch from Phase 3 into Python code.

### Execution order

1. **Icon functions** — Define all `icon_*()` functions from Phase 2
2. **Presentation + theme** — `prs = Presentation(theme=themes.professional)`
3. **Slides in sequence** — Build each slide following the Phase 3 sketch
4. **Animations** — Add `sb.animate()` following the grouping from Phase 3
5. **Speaker notes** — Add `sb.notes()` with what the presenter should say
6. **QA pass** — `prs.review()`, `prs.preview()`, `check_presentation()`

### Delegating to agents

For a large deck, you don't have to build every slide yourself. Delegate execution to specialized agents:

| Task | Agent | What to provide |
|------|-------|----------------|
| Build a single slide | **`slide-designer`** | Slide brief: claim, components, layout sketch, animation groups, visual vocabulary |
| Create a complex chart | **`chart-maker`** | Data, chart type, claim the figure supports, save path |

**Delegation workflow for a full deck:**

```
Main agent (you):
  1. Complete Phase 1–3 (logic flow, visual vocab, layout sketches)
  2. Write the deck scaffold: prs = Presentation(...), icon functions, prs.cover(...)
  3. For each content slide:
     → Dispatch slide-designer with the Phase 3 brief for that slide
     → If the slide needs a complex chart, slide-designer dispatches chart-maker
  4. Collect all slide code, assemble into final script
  5. Run QA pass on the complete deck
```

This lets multiple slides be built in parallel when they are independent.

### Charts: built-in vs delegation

| Need | Use | Why |
|------|-----|-----|
| Simple data exhibition (a few bars, a pie chart) | Built-in `BarChart`, `PieChart`, etc. | Native SVG, no external dependencies, fast |
| Publication-quality figure (error bars, CI bands, complex multi-panel) | Delegate to **`chart-maker` agent** | Uses `paperops.plotting` + matplotlib for full control |
| Figure that also appears in the paper | Delegate to **`chart-maker` agent** | Same figure, same style — save as PDF for paper, PNG for slides |

When delegating to `chart-maker`:
- Describe the data, the claim the figure should support, and the chart type
- The agent returns a file path — use it with `Image(path=...)`:

```python
# chart-maker agent saves to e.g. "figures/result.png"
from paperops.slides import Image
sb.layout(HStack(gap=0.5, children=[
    text_context,
    Image(path="figures/result.png", width=5.0, height=3.5),
]))
```

For simple inline cases where delegation is overkill, you can also use `paperops.plotting` directly:

```python
from paperops.plotting import apply_plot_config, figure, save
apply_plot_config("classic")
fig, ax = figure("single")
ax.bar(["A", "B"], [10, 20])
path = save(fig, "figures/comparison.png")
```
