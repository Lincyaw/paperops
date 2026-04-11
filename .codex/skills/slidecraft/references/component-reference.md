# Component Reference

This reference is intentionally selective. It lists the components most useful for deck decisions; use code docs for exhaustive signatures.

## High-Frequency Components

| Component | Use it for | Avoid when |
|-----------|------------|------------|
| `TextBlock` | short claim text, labels, explanatory copy | the slide becomes a wall of prose |
| `BulletList` | compact ordered points | every bullet is a full sentence |
| `RoundedBox` / `Box` | cards, containers, labeled states | the box exists only to hold too much text |
| `Badge` | tiny categorical labels or tags | the label must carry the main meaning |
| `Image` | external figures or rendered assets | the asset should be a native diagram instead |
| `SvgImage` | reusable icons or compact custom diagrams | the same visual can be made with native nodes more simply |
| `Table` | small exact comparisons | the table is dense enough to belong in backup |

## Recommended Pairings

- `HStack` + two `VStack` columns for before/after or method/baseline comparisons
- `TextBlock` + `Image` for evidence slides
- `RoundedBox` + `Badge` for lightweight cards with state labels
- `Flowchart` + short side notes for pipeline explanations

## Review Warnings

- many nested boxes often signal missing visual hierarchy, not sophistication
- manual widths everywhere usually indicate the parent layout is wrong
- if a component must hold a paragraph, ask whether notes or a second slide would be better
