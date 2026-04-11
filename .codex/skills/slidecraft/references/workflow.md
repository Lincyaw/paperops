# SlideCraft Workflow

Use this reference to decide the next step in deck production.

## Default Production Loop

1. Lock the story input
   - consume a talk plan from `talk-architect`, or confirm that the deck story is already stable
2. Lock the visual brief
   - consume a style/system brief from `visual-language` when the deck needs a coherent design language
3. Sketch slide intent
   - for each slide, record the message, visual role, and reveal order before writing code
4. Implement the slide tree
   - choose containers first, then assign leaves and assets
5. Save and review
   - build the deck, generate previews, and inspect the resulting state
6. Iterate through `slide-review`
   - when something looks wrong, diagnose before applying layout hacks

## Routing Rules

- unstable talk arc -> `talk-architect`
- unstable visual language -> `visual-language`
- stable plan + stable style -> `slidecraft`
- rendered deck with visible problems -> `slide-review`
- repo-level checks after changes -> `verify`

## Build Principles

- one main message per slide
- one dominant visual focus per reveal
- use templates for speed only when the slide still reads intentionally
- prefer reusable patterns over bespoke layout for every slide
- trust the component/layout system before reaching for arbitrary fixed dimensions
