# Slide Layout Patterns

Use these patterns as defaults before inventing a custom composition.

## Common Patterns

| Need | Preferred pattern | Notes |
|------|-------------------|-------|
| Big claim | `Callout` or large `TextBlock` + small support area | Keep the eye on one sentence first |
| Comparison | `HStack` with mirrored columns | Align headings and evidence density |
| Process | `Flow`, `Flowchart`, or linear `HStack`/`VStack` | Reveal steps in order |
| Evidence + interpretation | text on one side, figure on the other | Keep annotation close to the evidence |
| 3-6 key items | `Grid` of repeated cards | Repeat geometry and reduce wording |
| Timeline / staged reveal | stacked sections with explicit progression | Use animation to control the order |

## Container Heuristics

- `HStack` for side-by-side comparison or label/value pairings
- `VStack` for hierarchical reading from top to bottom
- `Grid` for repeated peers that should feel equivalent
- `Padding` for breathable whitespace around a strong focal component

## Density Heuristics

- if a slide cannot be summarized in one sentence, it likely holds more than one idea
- if the eye does not know where to look first, reduce simultaneous emphasis
- if both columns in a comparison require paragraph text, split the slide
