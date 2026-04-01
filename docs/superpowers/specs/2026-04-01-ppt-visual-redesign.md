# PPT Visual Redesign: Communication Friction Deck

## Context

Redesign the `examples/communication_friction/build.py` presentation to fix:
- Chaotic color usage (too many saturated colors competing on every slide)
- Over-reliance on RoundedBox making slides monotonous
- No cohesive visual identity

## Decisions

- **Audience**: Group sharing + teaching/training (relaxed, clear, guiding)
- **Style**: Warm, approachable, illustration-feel (Notion/Linear aesthetic)
- **Palette**: "晨雾蓝" — analogous blue-cyan-mint scheme
- **Shape language**: Icon-led + whitespace, minimal boxing

## Color System (60-30-10)

| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| Background 60% | Warm white | `#FAFBFC` | Slide base |
| Card area | Light blue-gray | `#F0F4F8` | Highlighted content areas |
| Layer: 呈现层 | Deep sea blue | `#2E5A88` | Presentation layer identity |
| Layer: 协作层 | Cyan stone | `#3D8B8B` | Collaboration layer identity |
| Layer: 探索层 | Mint green | `#6BC4A6` | Exploration layer identity |
| Accent (≤10%) | Coral orange | `#E8805C` | Warnings, friction, key emphasis |
| Text primary | Dark gray | `#2C3E50` | Body text, titles |
| Text secondary | Muted blue-gray | `#7B8FA3` | Captions, descriptions |
| Text light | Pale gray | `#B0BEC5` | Subtle labels |
| Border/divider | Soft edge | `#E2E8F0` | Thin lines, separators |

**Discipline**: Max 2 layer colors + 1 accent per slide. No more 5-color slides.

## Shape Language

1. **No box by default.** Text sits on whitespace. Only box when content needs visual grouping.
2. **SVG icons** are the primary visual anchors (1.5-2.0 inches), paired with text.
3. **Emphasis** via light background blocks (`#F0F4F8`) without borders, or left color bar (4px).
4. **Separation** via spacing and thin lines (`#E2E8F0`), not enclosing boxes.
5. **Contrast scenarios** (good vs bad) use subtle background tint, not saturated fill boxes.
6. **Callout** reserved for the single most important takeaway per slide.

## Icon Color Mapping

- Flask (探索层) → Mint `#6BC4A6`
- Bubbles (协作层) → Cyan stone `#3D8B8B`
- Stage (呈现层) → Deep blue `#2E5A88`
- Lightning (friction) → Coral `#E8805C`
- Robot, human, shuttle, dao → Dark gray `#2C3E50` with layer-color accents

## Typography

- Titles: left-aligned, dark gray `#2C3E50`, never colored
- Body: left-aligned, `#2C3E50`
- Supporting text: `#7B8FA3`, italic where appropriate
- No large colored Badge blocks; use small colored dots + text labels
- 40-50% whitespace per slide

## Slide-Level Changes

- **Slide 2**: Replace 4 RoundedBox with icon + text in 2x2 grid, spacing as separator
- **Slide 3**: Three large icons with text below, left color bars instead of filled boxes
- **Slide 5**: Three columns with icon headers, thin vertical lines as dividers
- **Slide 13-14**: Remove RoundedBox containers, use Callout only for key insight + bare text
- **Slide 16**: Enlarge robot/human icons as primary visual, text wraps around icons

## Scope

Single file change: `examples/communication_friction/build.py`. Also needs a custom theme override. No changes to the paperops library itself.
