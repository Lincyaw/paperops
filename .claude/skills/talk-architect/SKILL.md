---
name: talk-architect
description: "Use this skill whenever the user wants to turn one or more papers, notes, or source documents into an academic talk plan before building slides. Trigger for conference talks, seminars, research-line talks, job talks, speaker notes, slide scripts, talk outlines, or requests to organize source material into a PPT story. This skill plans the narrative, timing, audience framing, and slide-by-slide content; use slidecraft only after the talk plan is stable."
---

# Talk Architect

Design the talk before implementing the deck.

## Core rule

Choose the authoring surface only after the story is stable enough.

## Authoring DSL choice guide

- MDX: default for talk plans that mix prose, transitions, and semantic components
- JSON: best when an upstream planner already outputs structured slide objects
- Python: best when deck content depends on runtime data, loops, or reusable helper logic

## Planning workflow

1. lock audience, duration, and talk style
2. reduce the talk to a setup -> development -> evidence -> resolution arc
3. assign one claim per slide and write active titles
4. decide which slides want dense evidence versus keynote-style emphasis
5. hand the stable plan to `slidecraft` with an explicit authoring recommendation (MDX / JSON / Python)

## Deck handoff expectations

When handing off to `slidecraft`, include:
- recommended sheet (`minimal`, `academic`, `seminar`, `keynote`, `whitepaper`, or `pitch`)
- which slides are prose-heavy and may want `reflow`
- repeated class names or semantic roles worth encoding in `styles`
- any assets, figures, or tables that must appear

## Handoff

- to `slidecraft` when the narrative is locked and implementation can start
- to `visual-language` when the visual system needs a short brief before deck coding
- to `slide-review` when a rendered deck needs diagnosis instead of replanning
