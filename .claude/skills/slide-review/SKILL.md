---
name: slide-review
description: "Use this skill when reviewing an existing PPT or SlideCraft deck, diagnosing layout problems, inspecting preview PNGs, or iterating on slides based on deck state rather than blindly editing code."
---

# Slide Review

Use this skill when the task is primarily about understanding what the current deck looks like and what should be fixed next.

Typical triggers:

- review this deck
- inspect the current PPT
- why is this slide clipped
- check the preview
- diagnose layout problems
- iterate on the deck

## Core Rule

Do not jump straight into editing slide code. First establish the current deck state through generated artifacts.

## Workflow

1. Build or regenerate the deck.
2. Run `prs.review_deck(...)` when available.
3. Inspect:
   - integrated issue summary
   - `top_problem_slides`
   - slide-level summaries
   - preview PNGs
4. Classify the problem source:
   - content density
   - intrinsic sizing
   - container negotiation
   - preview/render mismatch
5. Make the smallest targeted fix that addresses the diagnosed source.
6. Regenerate and rerun review.

## What To Look For

### Content density

Symptoms:

- too many visible text blocks
- multiple long body strings on one slide
- a slide summary reports high density even without hard overflow

Fix strategy:

- shorten on-slide wording
- move full phrasing into speaker notes
- simplify the structure before changing geometry

### Intrinsic sizing

Symptoms:

- labels like `Telemetry` or `Root cause label` are clipped
- badges, callouts, or rounded boxes are too narrow for their text
- crowding appears in fit-content elements

Fix strategy:

- improve component `preferred_size()`
- strengthen min width / padding logic
- avoid arbitrary deck-local width hacks unless intent is truly fixed

### Container negotiation

Symptoms:

- multiple children are uniformly compressed
- arrows and nodes consume space incorrectly
- side-by-side regions look balanced in code but not in preview

Fix strategy:

- inspect `HStack` / `VStack` sizing behavior
- adjust `size_mode_x`, `size_mode_y`, `grow`, `shrink`, `basis`, `wrap`
- treat the parent layout as the root cause, not each child independently

### Preview mismatch

Symptoms:

- preview looks worse than the saved `.pptx` checks suggest
- saved-file checker is clean but preview still looks unreadable

Fix strategy:

- inspect preview text wrapping and heuristic assumptions
- compare preview artifact behavior with saved-file checks before touching slide content

## Expected Outputs

A good review result should tell the user:

- which slides are most problematic
- what kind of issue each slide has
- whether the likely fix belongs in content, components, layout, or preview tooling
- what should be changed next

Avoid vague feedback like "adjust spacing" or "make the box bigger" unless you can identify the actual failure mode.

