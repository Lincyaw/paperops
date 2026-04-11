---
name: slide-review
description: "Use this skill when reviewing an existing PPT or SlideCraft deck, diagnosing layout problems, inspecting preview PNGs, or iterating on slides based on current deck state rather than blindly editing code."
---

# Slide Review

Diagnose the current deck state before changing slide code.

## Use This Skill When

- the user asks to review a deck, inspect the current PPT, or diagnose clipping/overflow
- preview PNGs, layout checks, or saved-file validation artifacts already exist or should be generated
- the next action depends on identifying whether the problem is content, component sizing, layout negotiation, preview mismatch, or style drift

## Do Not Use This Skill When

- the main task is to create a talk story from raw source material
- the deck does not exist yet and the primary job is first-pass generation
- the real issue is missing visual direction rather than a broken rendered deck

## Core Rule

Do not jump straight into editing slide code. First establish the current deck state through generated artifacts.

## Workflow

1. Build or regenerate the deck
2. Run integrated review such as `prs.review_deck(...)` when available
3. Inspect issue summaries, slide-level findings, and preview PNGs
4. Classify the likely source:
   - content density
   - intrinsic sizing
   - container negotiation
   - preview/render mismatch
   - style drift
5. Propose or apply the smallest targeted fix
6. Regenerate and rerun review until the issue stabilizes

## Handoff

- if the root cause is story or pacing, hand off to `talk-architect`
- if the root cause is visual inconsistency, palette drift, or symbol-language drift, hand off to `visual-language`
- if the issue is implementation/layout after diagnosis, hand off to `slidecraft`
- if the user asks for repository validation after fixes, hand off to `verify`

## References

- use `../slidecraft/references/workflow.md` for the expected build -> review loop
- use `../visual-language/references/style-drift-checklist.md` when the deck feels visually inconsistent
- keep recommendations concrete: identify the broken slide, failure mode, and likely fix surface
