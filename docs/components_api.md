# Component API (Phase 3)

## Atomic Components

| Component | Default class | Props | Example |
|---|---|---|---|
| `text` | `text` | `text` | `{"type":"text","text":"Hello"}` |
| `prose` | `prose` | `text` | `{"type":"prose","text":"A short paragraph."}` |
| `title` | `title` | `text` | `{"type":"title","text":"Annual Report"}` |
| `subtitle` | `subtitle` | `text` | `{"type":"subtitle","text":"Q2 Update"}` |
| `heading` | `heading` | `text` | `{"type":"heading","text":"Highlights"}` |
| `box` | `box` | `text`, `style` | `{"type":"box","text":"Panel text"}` |
| `roundedbox` | `rounded-box` | `text` | `{"type":"roundedbox","text":"Chip"}` |
| `circle` | `circle` | `text` | `{"type":"circle","text":"1"}` |
| `badge` | `badge` | `text` | `{"type":"badge","text":"New"}` |
| `line` | `line` | `from`, `to`, `style`, `color`, `weight` | `{"type":"line","text":"divider"}` |
| `arrow` | `arrow` | `from`, `to`, `head`, `color`, `weight` | `{"type":"arrow","text":"Flow"}` |
| `divider` | `divider` | `orientation`, `thickness`, `length` | `{"type":"divider","props":{"orientation":"horizontal"}}` |
| `image` | `image` | `src`, `fit`, `width`, `height`, `alt` | `{"type":"image","props":{"src":"path/to/file.png"}}` |
| `svg` | `svg` | `src`, `body`, `fit`, `width`, `height` | `{"type":"svg","props":{"body":"<svg/>"}}` |
| `icon` | `icon` | `name`, `size`, `color` | `{"type":"icon","props":{"name":"star","size":"lg"}}` |
| `chart` | `chart` | `chart_type`, `data`, `title`, `labels`, `series`, `width`, `height` | `{"type":"chart","props":{"chart_type":"line"}}` |
| `table` | `table` | `headers`, `rows`, `header_color`, `header_text_color`, `font_size`, `row_height` | `{"type":"table","props":{"headers":["A","B"],"rows":[["1","2"]]}}` |
| `spacer` | `spacer` | `size`, `orientation`, `width`, `height` | `{"type":"spacer","props":{"size":"lg"}}` |

## Semantic Components

| Component | Default class | Props (required) | Example |
|---|---|---|---|
| `card` | `card` | *(none)* | `{"type":"card","children":[{"type":"text","text":"panel"}]}` |
| `kpi` | `kpi` | `label`, `value` / `delta` | `{"type":"kpi","props":{"label":"DAU","value":"125k","delta":"+56%"}}` |
| `callout` | `callout` | `kind`, `text` | `{"type":"callout","props":{"kind":"Insight","text":"A key observation"}}` |
| `quote` | `quote` | `text` / `author` | `{"type":"quote","props":{"text":"Data changes fast","author":"Ops Team"}}` |
| `pullquote` | `pullquote` | `text` / `author` | `{"type":"pullquote","props":{"text":"Important takeaway"}}` |
| `keypoint` | `keypoint` | `number`, `title`, `body` | `{"type":"keypoint","props":{"number":"01","title":"Setup","body":"Collect events"}}` |
| `stepper` | `stepper` | `steps` | `{"type":"stepper","props":{"steps":[{"label":"Collect"},{"label":"Analyze"},{"label":"Report"}]}}` |
| `timeline` | `timeline` | `items` | `{"type":"timeline","props":{"items":[{"date":"2026-01","title":"Pilot","desc":"Start"}]}}` |
| `figure` | `figure` | `src` / `body` / `chart_type`, `caption`, `source` | `{"type":"figure","props":{"chart_type":"bar","caption":"Recovery trend"}}` |
| `caption` | `caption` | `text` | `{"type":"caption","props":{"text":"Chart source: Internal"}}` |
| `spacer` | `spacer` | `size`, `orientation`, `width`, `height` | `{"type":"spacer","props":{"size":"lg"}}` |

> `note` is a semantic component in this phase so speaker-only text is captured by
> `codegen` into PPT speaker notes and does not create a visible shape.
