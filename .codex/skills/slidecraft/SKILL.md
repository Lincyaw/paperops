---
name: slidecraft
description: "Use this skill whenever creating, modifying, or generating PowerPoint presentations (.pptx) with paperops.slides. This includes native PowerPoint slide composition, themes, layouts, animations, and deck implementation once the talk structure and visual language are ready."
---

# SlideCraft

Use `paperops.slides` to implement the deck once the story and visual system are sufficiently stable.

## Use This Skill When

- the user wants to generate or modify PPT code with `paperops.slides`
- the task is deck implementation, slide composition, or layout construction
- the story already exists or can be treated as stable enough to render

## Do Not Use This Skill When

- the user first needs a talk outline, talk style, or speaker-note plan
- the main need is defining the deck's visual language, symbol vocabulary, or diagram grammar
- the task is primarily to diagnose an already-rendered deck from review artifacts

## Entry Checklist

Before writing slide code, confirm or infer:
- slide sequence or story arc
- audience and duration
- presentation theme or desired visual direction
- required figures, tables, or external assets
- whether a style brief from `visual-language` is needed

## Workflow

1. Stabilize inputs
   - if the story is still moving, send the task to `talk-architect`
   - if the deck lacks a coherent visual system, get a brief from `visual-language`
2. Translate the brief into implementation choices
   - map palette roles into theme and emphasis usage
   - map layout families into reusable slide skeletons
   - map symbol vocabulary into repeated components, icons, and connector styles
3. Build the deck structure
   - choose the theme, slide sequence, and reusable slide patterns
   - decide where template slides are enough and where custom composition is needed
4. Compose slides in code
   - use containers to express relationships first, then fill them with content
   - keep each slide centered on one message and one visual focal point
   - add animations only when they reveal reasoning order
5. Integrate assets
   - embed plotting outputs, images, and diagrams only after the slide structure is clear
   - reuse recurring visual motifs instead of inventing a new one per slide
6. Validate the result
   - save the deck
   - run review/preview checks
   - route rendering problems into `slide-review`

## Handoff

- to `talk-architect` when story, pacing, or slide purpose is still unstable
- to `visual-language` when color roles, symbol choices, or diagram style are not locked
- to `slide-review` when the question becomes "what is broken in the current deck?"
- to `verify` when implementation changes need repo-aware checks

## References

- `references/workflow.md` for the generation loop and routing rules
- `references/layout-patterns.md` for common slide structures
- `references/component-reference.md` for high-frequency components and pairings
- `references/brief-consumption.md` for turning a visual-language brief into SlideCraft choices
- `references/handoff-rules.md` for skill boundaries and transitions
