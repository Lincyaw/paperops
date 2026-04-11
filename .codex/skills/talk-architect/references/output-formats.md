# Output Formats

This skill should always produce two artifacts once the style is known.

## 1. `talk_master.md`

Required sections:
- `Talk Goal`
- `Audience Model`
- `Talk Style`
- `Core Thesis`
- `Why This Matters`
- `Story Arc`
- `Time/Page Budget`
- `Recurring Messages`
- `Slide-by-Slide Plan`
- `Backup Slides`
- `Audience Dropout Risks`
- `Open Risks / Missing Inputs`

### Slide-by-slide entry format

For each slide, include:
- `slide_no`
- `title_en`
- `message`
- `role` - one of `setup`, `problem`, `method`, `evidence`, `transition`, `takeaway`, `backup`
- `transition_from_previous`
- `visual_goal`
- `keep`
- `cut`
- `notes_zh`
- `time_sec`

`notes_zh` should sound like spoken delivery, not copied paper prose.

## 2. `slide_spec.md`

Use a compact structured representation for downstream slide generation.

Required top-level fields:
- `talk_style`
- `audience_type`
- `audience_distance`
- `duration_min`
- `language_policy`

Required per-slide fields:
- `slide_no`
- `title_en`
- `message`
- `visual_type`
- `build_order`
- `asset_or_figure_needs`
- `notes_zh`

Suggested `visual_type` values:
- `title`
- `claim`
- `comparison`
- `pipeline`
- `chart`
- `table-lite`
- `case-study`
- `transition`
- `takeaway`
- `backup`

Suggested `build_order` format:
1. claim or question
2. evidence or mechanism
3. interpretation or takeaway

## Style-Selection Prompt Behavior

Before writing either artifact, confirm style if needed.

Minimum clarification to ask when missing:
- What talk style do you want?
- Who is the audience?
- How long is the talk?

If the user is unsure, recommend a style and explain it briefly.

## Content Quality Checks

Before finalizing the artifacts, verify:
- every main slide has one primary message
- titles are active and informative
- timing fits the chosen talk style
- low-priority detail has been moved to backup
- the deck reads like one story rather than a paper summary dump
