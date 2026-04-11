# Consuming A Visual-Language Brief

Use this reference when `visual-language` has already produced a style-system brief and `slidecraft` needs to turn it into deck code.

## Goal

Translate the style brief into repeatable implementation choices. Do not reinterpret the visual language from scratch on every slide.

## Field Mapping

| Brief field | SlideCraft implementation target |
|-------------|----------------------------------|
| `palette_roles` | theme override colors, emphasis usage, badge/callout/result-card semantics |
| `type_scale` | title/body/caption/numeric text choices and repeated text hierarchy |
| `spacing_scale` | default gaps, padding, and card spacing across recurring slide families |
| `layout_families` | reusable skeletons for claim/evidence/comparison/process/takeaway slides |
| `symbol_vocab` | repeated node shapes, card types, badges, and icon families |
| `icon_style_rules` | `SvgImage` style, stroke/fill choices, level of detail |
| `connector_rules` | arrow/line treatment and when each is allowed |
| `image_policy` | when to choose `Image`, native diagram nodes, table, or no asset |
| `emphasis_rules` | what gets strongest contrast, largest type, or strongest reveal order |
| `accessibility_baseline` | minimum readable text, label strategy, non-color-only distinctions |

## Translation Workflow

1. Build a deck-level mapping table
   - write down how each brief field maps into concrete theme/components/layout choices
2. Create slide families before slide instances
   - define the 2-4 recurring skeletons implied by `layout_families`
   - keep gaps, padding, and heading treatment stable within each family
3. Define reusable visual primitives
   - shared card treatment
   - shared badge treatment
   - shared icon family
   - shared connector style
4. Implement slides by selecting from the established family
   - only create a one-off pattern when the slide purpose truly differs
5. Review for drift
   - if a slide requires special styling, check whether it breaks the brief's palette, symbol, or emphasis logic

## Common Mapping Decisions

- if `palette_roles` reserves a warm accent for warnings, do not spend it on generic highlights
- if `layout_families` says evidence slides use text + chart, do not alternate randomly between chart-left and chart-right without a reason
- if `symbol_vocab` says systems are rounded blocks and artifacts are neutral cards, keep that distinction everywhere
- if `emphasis_rules` say only one focal point per slide, do not make both the chart and the side note equally loud

## Failure Modes

- treating the brief as inspiration instead of as an implementation contract
- introducing special-case colors for one local slide problem
- changing icon style midway because a new SVG looked attractive
- using screenshots or photos where the brief preferred native diagrams
