# SlideCraft quickstart

This quickstart is the cutover guide for the IR-first `paperops.slides` API described in `docs/design/slide-dsl.md`.

## Mental model

Every authoring surface compiles to the same document shape:

1. Parse JSON / MDX / Markdown / Python builder into canonical IR
2. Resolve theme tokens + sheet rules + deck-local styles
3. Compute layout
4. Run overflow handling (`shrink`, `reflow`, `clip`, or `error`)
5. Emit `.pptx`

## Component shortlist for LLMs

Reach for these first:

- structure: `Slide`, `Grid`, `Flex`, `HStack`, `VStack`, `Layer`, `Padding`
- text: `Title`, `Subtitle`, `Heading`, `Text`
- semantics: `KPI`, `card`, `quote`, `callout`, `figure`, `note`
- assets: `Image`, `SvgImage`, `Table`

## Style keys that matter most

Prefer `style=` or stylesheet rules over bespoke inline component kwargs.

- spacing and layout: `padding`, `gap`, `cols`, `rows`, `grow`, `shrink`, `basis`
- typography: `font`, `font-weight`, `line-height`, `align`
- surfaces: `bg`, `color`, `border`, `radius`
- motion: `animate`, `animate-trigger`, `animate-group`, `stagger`, `delay`, `duration`
- overflow: `overflow` with `shrink`, `reflow`, `clip`, or `error`

## Built-in sheets

- `minimal`: lots of whitespace, clean defaults
- `academic`: denser reading rhythm and citation-friendly cards
- `seminar`: balanced text/data layout for research talks
- `keynote`: large emphasis blocks and high contrast
- `whitepaper`: report-like two-column reading lanes
- `pitch`: narrative rhythm and strong highlight moments

## Authoring choice guide

- use MDX when the deck is mostly narrative and the writer wants prose + components
- use JSON when another tool already knows the structure exactly
- use Python when data, loops, or conditionals generate deck content

A practical default for LLMs:
- start in MDX for exploratory deck drafts
- switch to JSON when an upstream planner/tool already outputs structured cards
- switch to Python when the deck depends on runtime data or shared helper functions

## LLM system prompt fragment

You can drop this into a slide-generation system prompt:

```text
You are authoring slides with the IR-first paperops.slides API.
Prefer MDX unless the task is strongly data-structured (JSON) or programmatic (Python).
Use semantic components and classes first; do not use absolute coordinates or legacy builder patterns.
Keep one core claim per slide, move repeated visual rules into `sheet` or `styles`, and pick an explicit `sheet` (`minimal`, `academic`, `seminar`, `keynote`, `whitepaper`, or `pitch`).
For text-heavy nodes, choose an overflow policy deliberately; default to `shrink` for titles and `reflow` for prose.
```

## Quick verification loop

```bash
uv run python examples/4.15-talk/presentation.py
uv run python examples/gallery/render_gallery_variants.py --output-dir /tmp/paperops-gallery
make verify
```
