---
name: slidecraft
description: "Use this skill whenever creating, modifying, or generating PowerPoint presentations (.pptx) with paperops.slides. This includes native PowerPoint slide composition, themes, layouts, animations, and deck implementation once the talk structure and visual language are ready."
---

# SlideCraft

Use `paperops.slides` to implement the deck once the story and visual system are stable enough to render.

## Core rule

Author with the IR-first API. Prefer `Deck` / `Slide` / `Title` / `Subtitle` / `Heading` / `Text` / `Grid` / `Flex` / `KPI` and render through the shared pipeline.

## Authoring defaults

- default to MDX for narrative decks with prose + components
- use JSON when the structure is already known exactly
- use Python when data, loops, or shared helper functions generate slide content
- choose an explicit `sheet`: `minimal`, `academic`, `seminar`, `keynote`, `whitepaper`, or `pitch`

## High-frequency components

Start with these before reaching for custom helpers:
- structure: `Slide`, `Grid`, `Flex`, `HStack`, `VStack`, `Layer`, `Padding`
- text: `Title`, `Subtitle`, `Heading`, `Text`
- semantics: `KPI`, `card`, `callout`, `quote`, `figure`, `note`
- assets: `Image`, `SvgImage`, `Table`

## Style rules

- put repeated visual decisions into `sheet` or deck-local `styles`
- use classes to communicate role: `cover`, `card`, `kpi`, `rail`, `summary`, `hero`
- prefer style keys such as `padding`, `gap`, `cols`, `bg`, `color`, `border`, `radius`, `font`, `font-weight`, `align`, and `overflow`
- use animation style keys only when they reveal reasoning order: `animate`, `animate-trigger`, `animate-group`, `stagger`, `delay`, `duration`

## Workflow

1. stabilize the story and visual brief
2. choose MDX / JSON / Python based on authoring pressure
3. choose the sheet and only then add deck-local styles
4. build the deck with semantic structure first, then fill in text and assets
5. render, review, and route remaining issues into `slide-review`

## LLM prompt fragment

Reference `docs/quickstart-slides.md` and reuse this guidance when prompting an LLM:
- prefer MDX unless the task is strongly structured or programmatic
- do not use coordinate-first absolute-position builder patterns
- keep one claim per slide
- pick an explicit sheet and overflow policy

## Handoff

- to `talk-architect` when the narrative or pacing is unstable
- to `visual-language` when palette, diagram grammar, or symbol vocabulary is still unclear
- to `slide-review` when the deck exists and the question is diagnosis rather than first-pass generation
- to `verify` when you need repo-aware checks before shipping
