---
name: talk-architect
description: "Use this skill whenever the user wants to turn one or more papers, notes, or source documents into an academic talk plan before building slides. Trigger for conference talks, seminars, research-line talks, job talks, speaker notes, slide scripts, talk outlines, or requests to organize source material into a PPT story. This skill plans the narrative, timing, audience framing, and slide-by-slide content; use slidecraft only after the talk plan is stable."
---

# Talk Architect

Design the talk before designing the deck.

## Use This Skill When

- the user wants a talk outline, slide-by-slide plan, speaker notes, or talk script
- the user has one or more papers/notes and needs an audience-aware presentation story
- the deck structure, pacing, or audience framing is still unstable

## Do Not Use This Skill When

- the story is already locked and the task is to implement PPT code or layout
- the main problem is visual consistency, symbol language, or diagram style
- the task is to review an existing deck for clipping, density, or layout regressions

## Core Rule

Do not draft talk content until the talk style is known.

If the user wants a talk but style is missing, confirm at least:
- talk style
- audience type
- duration

Recommend a default instead of silently guessing:
- `conference` for single-paper talks
- `seminar` for internal or lab talks
- `research-line` for multi-paper talks
- `job-talk` for candidate-style talks

## Workflow

1. Intake
   - gather the source documents, venue, duration, audience, language, must-cover points, and must-avoid points
   - classify the request as single-paper, multi-paper, or research-line
2. Lock the framing
   - define the audience's core question, the talk thesis, and the main promise
   - choose pacing based on talk style and audience distance
3. Compress the story
   - reorganize around audience value instead of paper section order
   - reduce the story to setup -> development -> evidence -> resolution
4. Plan slide by slide
   - one primary message per slide
   - active English title
   - explicit transition from the previous slide
   - Chinese speaker notes by default unless the user asks otherwise
5. Keep or cut aggressively
   - keep only the strongest evidence for each claim in the main flow
   - move tables, dense ablations, and low-priority detail to backup
6. Produce handoff artifacts
   - `talk_master.md` for the presenter
   - `slide_spec.md` or equivalent structured block for downstream deck generation

## Handoff

- if the story is stable and the next task is deck implementation, hand off to `slidecraft`
- if the visual system is underspecified, produce a short style brief and hand off to `visual-language` before `slidecraft`
- if the user later asks to diagnose an existing deck, hand off to `slide-review`

## References

- `references/style-taxonomy.md` for talk types and audience distance
- `references/output-formats.md` for required artifact structure
- `references/best-practices.md` for current talk-planning heuristics and rationale
