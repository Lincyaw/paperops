# SlideCraft API Reference

## Table of Contents

1. [Presentation](#presentation)
2. [Themes](#themes)
3. [Layout Containers](#layout-containers)
4. [Text Components](#text-components)
5. [Shape Components](#shape-components)
6. [Composite Components](#composite-components)
7. [Chart Components](#chart-components)
8. [Image Components](#image-components)
9. [SvgCanvas](#svgcanvas)
10. [Slide Constants](#slide-constants)

---

## Presentation

Main entry point. All slides are created through this class.

```python
prs = Presentation(theme=themes.professional)
```

### Methods

| Method | Description |
|--------|-------------|
| `slide(title=None, reference=None)` | Create custom slide, returns `SlideBuilder` |
| `cover(title, subtitle="", author="")` | Cover slide |
| `section(num, title, subtitle="")` | Section divider ("Part N") |
| `content(title, bullets=None, table=None, reference=None)` | Content with bullets or table |
| `comparison(title, left, right, reference=None)` | Two-column comparison. left/right: `(column_title, items_list)` |
| `quote(text, author="", reference=None)` | Quote slide |
| `transition(text, sub_text="")` | Bridge/transition slide |
| `end(title, subtitle="")` | Closing slide |
| `save(path)` | Build and save to .pptx |
| `review()` | Returns `{"total_slides": N, "total_issues": N, "issues": [...]}` |
| `preview(slides=None, output_dir=None)` | Render to PNG, returns list of paths |

### SlideBuilder

Returned by `prs.slide()`. Chain methods:

```python
sb = prs.slide(title="Title", reference="Source")
sb.layout(component)        # Set content layout (any LayoutNode)
sb.notes("Speaker notes")   # Set speaker notes
sb.animate([[g1], [g2]])    # Click-to-advance animation groups
```

---

## Themes

```python
from paperops.slides import themes

# Built-in themes
themes.professional  # Blue/teal, Calibri
themes.minimal       # Gray/blue, Calibri
themes.academic      # Brown/green, Georgia

# Custom override
custom = themes.professional.override(
    colors={"primary": "#1E2761", "accent": "#FF6B35"},
    fonts={"title": 36},
    font_family="Arial",
)
```

### Professional Theme Colors

| Name | Hex | Usage |
|------|-----|-------|
| `primary` | `#3B6B9D` | Main accent, headers |
| `secondary` | `#4A8B7F` | Secondary accent |
| `accent` | `#C27C3E` | Warm highlights |
| `positive` | `#4A7C5F` | Good/success |
| `negative` | `#B85450` | Bad/warning |
| `highlight` | `#6B5B8D` | Purple emphasis |
| `warning` | `#C4A34D` | Caution |
| `text` | `#1E293B` | Primary text |
| `text_mid` | `#64748B` | Secondary text |
| `text_light` | `#94A3B8` | Subtle text |
| `bg` | `#FFFFFF` | White background |
| `bg_alt` | `#F7F8FA` | Alternating bg |
| `bg_accent` | `#EEF0F5` | Accent bg |
| `border` | `#C8CED8` | Borders/lines |

### Font Sizes

| Name | Points |
|------|--------|
| `title` | 32 |
| `subtitle` | 24 |
| `heading` | 20 |
| `body` | 18 |
| `caption` | 14 |
| `small` | 11 |

---

## Layout Containers

### HStack

Left-to-right layout. Children without explicit `width` share remaining space equally.

```python
HStack(gap=0.3, children=[child_a, child_b, child_c])
```

### VStack

Top-to-bottom layout. Children without explicit `height` share remaining space equally.

```python
VStack(gap=0.3, children=[child_a, child_b])
```

### Grid

N-column grid with auto-calculated rows.

```python
Grid(cols=3, gap=0.3, children=[a, b, c, d, e])  # 2 rows: [a,b,c] and [d,e]
```

### Padding

Wraps a single child with padding (in inches).

```python
Padding(child=component, all=0.3)
Padding(child=component, left=0.5, top=0.2, right=0.5, bottom=0.2)
```

### Sizing

All LayoutNodes accept keyword args:
- `width`: explicit width in inches
- `height`: explicit height in inches
- `min_width`: minimum width in inches
- `min_height`: minimum height in inches

Children with explicit width/height are sized first; remaining space is divided among flexible children.

---

## Text Components

### TextBlock

```python
TextBlock(
    text="Hello world",
    font_size="body",       # Semantic name or numeric pt
    color="text",           # Semantic name, hex, or RGB tuple
    align="left",           # "left", "center", "right"
    bold=False,
    italic=False,
    width=None,             # Optional explicit width (inches)
    height=None,            # Optional explicit height (inches)
)
```

### BulletList

```python
BulletList(
    items=[
        "First point",
        "Second point",
        ("Sub-point with indent", 1),   # (text, indent_level)
        ("Deep indent", 2),
    ],
    font_size="body",
    color="text",
)
```

---

## Shape Components

### Box

```python
Box(
    text="Label",
    color="bg_alt",         # Fill color
    border="border",        # Border color
    text_color="text",
    font_size="body",
    bold=False,
    align="center",         # Text alignment
    border_width=1.0,       # Points
)
```

### RoundedBox

Same as Box plus `radius=0.08` (corner radius).

### Circle

```python
Circle(
    text="1",
    color="primary",        # Fill color
    text_color="white",
    font_size="body",
    bold=True,
    radius=None,            # Explicit radius in inches
)
```

Circle forces equal width/height (uses the smaller dimension).

### Badge

Small colored label, auto-sized to text.

```python
Badge(text="NEW", color="primary", text_color="white", font_size="caption", bold=True)
```

### Arrow

Connector arrow between two components. Must set `from_component` and `to_component` to already-created component instances.

```python
box_a = RoundedBox(text="A")
box_b = RoundedBox(text="B")
arrow = Arrow(
    from_component=box_a,
    to_component=box_b,
    label=None,             # Not rendered yet (reserved)
    color="primary",
    width_pt=1.5,
    direction="horizontal", # "horizontal" or "vertical"
)
# Include all three in the same container:
HStack(children=[box_a, arrow, box_b])
```

### Line

Same as Arrow but without arrowhead. Supports `dashed=True`.

```python
Line(from_component=a, to_component=b, color="border", width_pt=1.0, dashed=False)
```

---

## Composite Components

Composites expand into trees of basic components at render time. They render as **native PPT shapes** (not SVG).

### Callout

Box with colored left accent bar, title, and body text.

```python
Callout(
    title="Important",
    body="This is the key takeaway from the analysis.",
    color="primary",        # Accent bar color
)
```

### Flow

Connected chain of boxes with arrows.

```python
Flow(
    labels=["Input", "Process", "Output"],
    direction="horizontal", # "horizontal" or "vertical"
    colors=["primary", "secondary", "positive"],  # Per-box colors (optional)
    arrow_color="primary",
)
```

### Flowchart

Node-edge diagram with explicit connections.

```python
Flowchart(
    nodes={
        "a": "Input",                      # Simple: label only
        "b": ("Process", "primary"),        # With color
        "c": ("Output", "positive"),
    },
    edges=[
        ("a", "b"),                         # Simple edge
        ("b", "c", "transforms"),           # Edge with label (label not rendered yet)
    ],
    direction="right",      # "right" or "down"
)
```

Nodes are topologically sorted based on edges. Renders as native RoundedBox + Arrow shapes.

---

## Chart Components

Charts render as SVG -> PNG images. Use only when native shapes can't represent the data.

### BarChart

Grouped bar chart.

```python
BarChart(
    groups=[
        ("Group A", [
            ("Series 1", 85, "primary"),    # (label, value, color_name)
            ("Series 2", 72, "secondary"),
        ]),
        ("Group B", [
            ("Series 1", 91, "primary"),
            ("Series 2", 88, "secondary"),
        ]),
    ],
    y_label="Score",        # Y-axis label
    show_values=True,       # Show value labels above bars
    max_value=None,         # Auto-calculated from data if None
)
```

### RadarChart

Spider/radar chart. Values are normalized 0-1.

```python
RadarChart(
    dimensions=["Speed", "Accuracy", "Cost", "Ease", "Scale"],
    series=[
        ("Ours", [0.9, 0.85, 0.7, 0.95, 0.8], "primary"),
        ("Baseline", [0.6, 0.9, 0.5, 0.7, 0.6], "secondary"),
    ],
)
```

---

## Image Components

### Image

Static image from file path.

```python
Image(path="/path/to/image.png", width=4.0, height=3.0)
```

### SvgImage

Render SVG string or SvgCanvas to PNG.

```python
SvgImage(svg="<svg>...</svg>", scale=3)
# Or with SvgCanvas:
SvgImage(svg=my_canvas, scale=3)
```

---

## SvgCanvas

Structured SVG builder for custom visualizations. Theme-aware color resolution.

```python
from paperops.slides.components.svg_canvas import SvgCanvas

svg = SvgCanvas(width=1200, height=600, theme=themes.professional, bg="bg")

# Definitions
svg.shadow_filter("shadow")
svg.arrow_markers()
svg.gradient("grad1", "primary", "secondary", direction="horizontal")

# Drawing
svg.rect(x, y, w, h, fill="bg_alt", stroke="border", rx=8)
svg.rounded_rect(x, y, w, h, text="Label", fill="primary", text_color="white", bold=True, filter_id="shadow")
svg.circle(cx, cy, r, text="1", fill="accent")
svg.text(x, y, "Text", color="text", size=16, bold=False, anchor="middle")
svg.line(x1, y1, x2, y2, color="border", width=1, dashed=True)
svg.arrow(x1, y1, x2, y2, color="primary", width=2)
svg.path(d="M 0 0 L 100 100", stroke="primary", fill="none")
svg.polygon([(x1,y1), (x2,y2), (x3,y3)], fill="accent", opacity=0.5)

# Groups (for transforms/opacity)
with svg.group(transform="translate(100, 50)", opacity=0.8) as g:
    g.rect(0, 0, 200, 100, fill="primary")
    g.text(100, 50, "Grouped", color="white")

# Render
svg_string = svg.render()
```

Use SvgCanvas when you need custom visualizations beyond what BarChart/RadarChart provide. Wrap with `SvgImage(svg=canvas)` to embed in a slide.

### SvgCanvas vs Raw SVG Strings

| Situation | Use | Why |
|-----------|-----|-----|
| Simple shapes (circles, rects, lines, text) | **SvgCanvas** | Theme-aware colors, type safety, consistent API |
| Complex paths, gradients, filters | **Raw SVG string** | Full SVG feature access, easier for complex graphics |
| Icons that need theme color changes | **SvgCanvas** | Semantic colors auto-update when theme changes |
| Static complex illustrations | **Raw SVG string** | Portability, can use any SVG editor |

**Using raw SVG strings:**

```python
from paperops.slides import SvgImage

# For complex graphics that SvgCanvas can't express easily
complex_svg = """
<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#3B6B9D;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#4A8B7F;stop-opacity:1" />
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <circle cx="100" cy="100" r="80" fill="url(#grad)" filter="url(#glow)"/>
  <path d="M 60 100 Q 100 60 140 100 Q 100 140 60 100" fill="white"/>
</svg>
"""

sb.layout(HStack(gap=0.3, children=[
    SvgImage(svg=complex_svg, width=1.5, height=1.5),
    TextBlock(text="Complex visualization", font_size="body"),
]))
```

**Trade-off:** SvgCanvas provides theme integration and type safety; raw SVG provides full feature flexibility. Choose based on complexity and whether you need theme-aware colors.

---

## Slide Constants

The slide is 16:9 widescreen (13.333" x 7.5").

| Constant | Value | Description |
|----------|-------|-------------|
| `SLIDE_WIDTH` | 13.333" | Slide width |
| `SLIDE_HEIGHT` | 7.5" | Slide height |
| `MARGIN_LEFT` | 0.8" | Left margin |
| `MARGIN_RIGHT` | 0.8" | Right margin |
| `MARGIN_TOP` | 0.5" | Top margin |
| `MARGIN_BOTTOM` | 0.4" | Bottom margin |
| `CONTENT_REGION` | 0.8, 1.5, 11.733, 5.6 | left, top, width, height — area below title |
| `TITLE_REGION` | 0.8, 0.4, 11.733, 0.85 | Title text area |

Content is automatically laid out within `CONTENT_REGION` by the layout engine. You don't need to calculate positions manually — just build the component tree and call `sb.layout(root)`.
