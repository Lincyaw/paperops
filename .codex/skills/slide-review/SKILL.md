---
name: slide-review
description: "Use this skill when reviewing an existing PPT or SlideCraft deck, diagnosing layout problems, inspecting preview PNGs, or iterating on slides based on current deck state rather than blindly editing code."
---

# Slide Review

Diagnose the rendered deck before editing authoring code.

## Review checklist

- class naming: do classes describe semantic role (`cover`, `card`, `kpi`, `summary`) instead of accidental styling?
- sheet fit: is the selected `sheet` helping the story, or fighting the density and emphasis pattern?
- style drift: are repeated values living in `styles` / sheet rules instead of being copied slide by slide?
- autofit policy: do titles use `shrink`, prose-heavy blocks use `reflow`, and error-prone dense shapes avoid silent clipping?
- container logic: is the problem a `Grid`/`Flex` relationship issue rather than a text issue?

## Workflow

1. build or regenerate the deck
2. inspect review output, preview PNGs, and slide-level issues
3. classify the failure as class naming, sheet mismatch, overflow policy, container negotiation, or content density
4. make the smallest targeted fix
5. rerender and compare again

## Typical fixes

- move repeated styling into `styles` or a built-in sheet override
- rename vague classes so selectors and intent are obvious
- swap `sheet` before inventing local one-off styles
- change `overflow` deliberately instead of shrinking every text box blindly
- reduce competing focal points so one slide carries one claim

## Handoff

- to `slidecraft` for implementation changes after diagnosis
- to `talk-architect` when density problems are really story problems
- to `visual-language` when inconsistency comes from an unstable visual system
- to `verify` for final repo-aware regression checks
