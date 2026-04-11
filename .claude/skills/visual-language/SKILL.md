---
name: visual-language
description: "Use this skill when a PPT needs a coherent design language, symbol vocabulary, or diagram style. It defines the presentation's visual system before SlideCraft code is written or revised."
---

# Visual Language

Define the deck's visual system before implementing slide code.

## Use This Skill When

- the user asks for a PPT's design language, symbol language, illustration style, or diagram style
- multiple slides need a shared visual identity rather than page-by-page styling
- the deck lacks clear rules for hierarchy, iconography, imagery, or repeated motifs
- review feedback says the deck feels inconsistent, noisy, or visually ad hoc

## Do Not Use This Skill When

- the main task is to draft the talk story or speaker notes
- the visual system is already explicit and the task is pure deck implementation
- the task is a quantitative research figure that belongs in `plotting`
- the task is primarily diagnosing overflow or layout breakage in an existing render

## Core Output

Produce a style-system brief that `slidecraft` can implement directly. The brief must define:
- `style_name`
- `style_keywords`
- `audience_fit`
- `palette_roles`
- `type_scale`
- `spacing_scale`
- `layout_families`
- `symbol_vocab`
- `icon_style_rules`
- `connector_rules`
- `image_policy`
- `emphasis_rules`
- `accessibility_baseline`
- `drift_checks`

The goal is not to sound tasteful. The goal is to make another agent render a coherent deck without improvising the visual language slide by slide.

## Workflow

1. Read the talk intent
   - identify audience, venue, topic, duration, and desired academic tone
   - infer whether the deck needs to feel more conference-like, seminar-like, tutorial-like, or candidate-talk-like
2. Define system foundations
   - choose palette roles, type scale, spacing rhythm, and 2-4 recurring layout families
   - decide what must stay constant across the whole deck
3. Define symbol language
   - map recurring concepts to stable shape families and icon metaphors
   - keep the number of symbol families low enough to learn in one viewing
4. Define image policy
   - specify when to use native shapes, plots, tables, screenshots, photos, or no image at all
   - reject decorative imagery that does not advance the argument
5. Define emphasis and accessibility rules
   - specify how focus is created using size, contrast, position, whitespace, and reveal order
   - enforce readability, contrast, and non-color-only meaning
6. Package the brief for implementation and review
   - write the spec in concrete terms that `slidecraft` can apply
   - include drift checks that `slide-review` can use later

## Handoff

- hand the completed style-system brief to `slidecraft` for implementation
- if deck review shows palette drift, symbol drift, or inconsistent image use, return here before doing local layout tweaks
- if the user needs bitmap asset generation beyond the brief, escalate to a dedicated image-generation workflow outside this skill

## References

- `references/style-brief-template.md` for the required output shape
- `references/system-foundation.md` for palette, type, spacing, and layout-system decisions
- `references/iconography.md` for symbol and icon rules
- `references/image-policy.md` for asset selection decisions
- `references/emphasis-and-hierarchy.md` for focus and information-layering rules
- `references/accessibility-baseline.md` for readability and inclusive-design constraints
- `references/style-drift-checklist.md` for review-time consistency checks
