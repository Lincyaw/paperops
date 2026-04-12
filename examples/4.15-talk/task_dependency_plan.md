# 4.15 Talk Task Dependency Plan

## Purpose

This document decomposes the deck refinement into executable tasks with explicit
ordering, prerequisites, outputs, ownership boundaries, and review points. The
goal is to make the work parallelizable without losing system coherence.

## Dependency Model

The work is organized into seven layers:

- `L0` Baseline diagnosis and artifact freeze
- `L1` Deck-wide visual system specification
- `L2` Slide-family rules and page-by-page implementation spec
- `L3` Shared component library and visual primitives in code
- `L4` Slide implementation by act
- `L5` Backup slide implementation
- `L6` Whole-deck verification, diversity audit, and polish

Higher layers may start only when their required prerequisites are complete.
Some tasks within a layer can run in parallel if they have disjoint write scope.

## Task Graph Overview

### L0 - Baseline diagnosis and artifact freeze

#### T0.1 Baseline inventory
- Purpose: capture current files, preview assets, and implementation surface.
- Inputs: `presentation.py`, `visual_system.md`, `slide_visual_spec.md`, preview PNGs.
- Outputs: confirmed baseline state and target files list.
- Depends on: none.
- Blocks: all later tasks.

#### T0.2 Current deck diagnosis summary
- Purpose: write the diagnosis that motivates the redesign.
- Inputs: preview review, existing deck docs.
- Outputs: diagnosis section in planning docs.
- Depends on: T0.1.
- Blocks: T1.1, T2.1.

### L1 - Deck-wide visual system specification

#### T1.1 Rewrite visual system principles
- Purpose: define style positioning, layout families, shape families,
  anti-repetition rules, act rhythm map, and semantic vocabulary.
- Inputs: T0.2 diagnosis.
- Outputs: updated `visual_system.md`.
- Depends on: T0.2.
- Blocks: T2.1, T3.1.

#### T1.2 Define anti-repetition and diversity rules
- Purpose: make geometric diversity a hard implementation constraint.
- Inputs: T1.1 draft.
- Outputs: diversity rules embedded in `visual_system.md` and referenced by
  downstream implementation specs.
- Depends on: T1.1.
- Blocks: T2.1, T6.2.

### L2 - Slide-family rules and page-by-page implementation spec

#### T2.1 Rewrite deck-level slide visual spec
- Purpose: align `slide_visual_spec.md` with the new visual system.
- Inputs: T1.1, T1.2.
- Outputs: updated deck-level visual strategy section.
- Depends on: T1.2.
- Blocks: T2.2, T3.2, T4.*.

#### T2.2 Rewrite main-slide visual specs (01-32)
- Purpose: define page-level dominant geometry, layout intent, anti-patterns,
  and implementation surfaces for all main slides.
- Inputs: T2.1, `slide_spec.md`, `talk_master.md`.
- Outputs: updated main slide entries in `slide_visual_spec.md`.
- Depends on: T2.1.
- Blocks: T4.1-T4.5.

#### T2.3 Rewrite backup-slide visual specs (B1-B6)
- Purpose: define appendix support pages under the same visual system.
- Inputs: T2.1, `slide_spec.md` backup section.
- Outputs: updated backup-slide section in `slide_visual_spec.md`.
- Depends on: T2.1.
- Blocks: T5.1.

### L3 - Shared component library and visual primitives in code

#### T3.1 Design reusable component API inside `presentation.py`
- Purpose: map the visual system to reusable helper builders and SVG primitives.
- Inputs: T1.1, current `presentation.py` helpers.
- Outputs: agreed helper list and code insertion plan.
- Depends on: T1.1.
- Blocks: T3.2, T4.1-T4.5, T5.1.

#### T3.2 Implement shared builders and semantic primitives
- Purpose: add the reusable building blocks required by multiple acts.
- Inputs: T3.1, T2.1.
- Outputs: updated helper layer in `presentation.py`.
- Depends on: T3.1, T2.1.
- Blocks: T4.1-T4.5, T5.1.

### L4 - Main slide implementation by act

These tasks can run partly in parallel after T2.2 and T3.2 are complete, so long
as ownership boundaries are respected.

#### T4.1 Implement Act I slides (01-06)
- Purpose: build the opening, thesis, and audience-orientation slides.
- Inputs: T2.2, T3.2.
- Outputs: refined slides 01-06 in `presentation.py`.
- Depends on: T2.2, T3.2.
- Blocks: T6.1.
- Ownership: slide branches for 01-06 only.

#### T4.2 Implement Act II slides (07-15)
- Purpose: rebuild realism pages, first shock evidence, and first act close.
- Inputs: T2.2, T3.2.
- Outputs: refined slides 07-15.
- Depends on: T2.2, T3.2.
- Blocks: T6.1.
- Ownership: slide branches for 07-15 only.

#### T4.3 Implement Act III slides (16-21)
- Purpose: rebuild capability question, task contract, scale pressure, and gap slides.
- Inputs: T2.2, T3.2.
- Outputs: refined slides 16-21.
- Depends on: T2.2, T3.2.
- Blocks: T6.1.
- Ownership: slide branches for 16-21 only.

#### T4.4 Implement Act IV slides (22-29)
- Purpose: rebuild trustworthiness, process verification, and trust-gap slides.
- Inputs: T2.2, T3.2.
- Outputs: refined slides 22-29.
- Depends on: T2.2, T3.2.
- Blocks: T6.1.
- Ownership: slide branches for 22-29 only.

#### T4.5 Implement Act V slides (30-32)
- Purpose: rebuild synthesis, roadmap, and final close.
- Inputs: T2.2, T3.2.
- Outputs: refined slides 30-32.
- Depends on: T2.2, T3.2.
- Blocks: T6.1.
- Ownership: slide branches for 30-32 only.

### L5 - Backup implementation

#### T5.1 Implement backup slides (B1-B6)
- Purpose: rebuild backup pages under the updated system.
- Inputs: T2.3, T3.2.
- Outputs: refined backup slide branches.
- Depends on: T2.3, T3.2.
- Blocks: T6.1.
- Ownership: backup slide branches only.

### L6 - Whole-deck verification and polish

#### T6.1 Build and render full deck
- Purpose: verify the deck builds and previews after main/backup implementation.
- Inputs: T4.1-T4.5, T5.1.
- Outputs: regenerated PPTX and preview PNGs.
- Depends on: T4.1, T4.2, T4.3, T4.4, T4.5, T5.1.
- Blocks: T6.2, T6.3.

#### T6.2 Diversity audit
- Purpose: check the deck against anti-repetition constraints.
- Inputs: T6.1 preview outputs, T1.2 diversity rules.
- Outputs: pass/fail list of geometric repetition and layout-family issues.
- Depends on: T6.1, T1.2.
- Blocks: T6.4.

#### T6.3 Layout and readability review
- Purpose: detect clipping, density, weak hierarchy, and figure/text imbalance.
- Inputs: T6.1 outputs.
- Outputs: issue list grouped by severity and slide.
- Depends on: T6.1.
- Blocks: T6.4.

#### T6.4 Final polish pass
- Purpose: fix issues found in diversity audit and layout review.
- Inputs: T6.2, T6.3.
- Outputs: final deck candidate.
- Depends on: T6.2, T6.3.
- Blocks: T6.5.

#### T6.5 Final acceptance review
- Purpose: confirm that objectives and acceptance criteria from
  `deck_refinement_plan.md` are met.
- Inputs: T6.4.
- Outputs: accepted deck state or remaining issue list.
- Depends on: T6.4.
- Blocks: completion.

## Parallelization Strategy

### Parallel group A: documentation and specification
These tasks are the correct first wave because they unblock all later implementation.

- T1.1 Rewrite visual system principles
- T2.1 Rewrite deck-level slide visual spec
- T2.2 Main-slide visual spec rewrite
- T2.3 Backup-slide visual spec rewrite

Constraint:
- T1.1 must land before T2.* is finalized.
- T2.2 and T2.3 may proceed in parallel once T2.1 is stable.

### Parallel group B: implementation scaffolding
These tasks start only after visual spec is stable.

- T3.2 shared builders
- T4.1 Act I
- T4.2 Act II
- T4.3 Act III
- T4.4 Act IV
- T4.5 Act V
- T5.1 Backup slides

Constraint:
- T3.2 should land before act workers finish, because multiple acts depend on
  the shared component library.
- Act workers must have disjoint write ownership or merge in sequence.

### Parallel group C: verification and convergence
These tasks happen after implementation converges.

- T6.1 build and render
- T6.2 diversity audit
- T6.3 layout review
- T6.4 polish
- T6.5 final acceptance

## Recommended Subagent Ownership

### Worker 1 - Visual system owner
- Owns: `visual_system.md`
- Tasks: T1.1, T1.2
- Must not edit: `presentation.py`, `slide_visual_spec.md`

### Worker 2 - Slide visual spec owner
- Owns: `slide_visual_spec.md`
- Tasks: T2.1, T2.2, T2.3
- Must not edit: `presentation.py`, `visual_system.md`

### Worker 3 - Shared component owner
- Owns: shared helper layer inside `presentation.py`
- Tasks: T3.1, T3.2
- Must not edit: slide-specific branches outside helper infrastructure unless required

### Worker 4 - Main deck implementation owner
- Owns: act slide branches in `presentation.py`
- Tasks: T4.*
- Must coordinate slide-number ownership if parallelized by act

### Worker 5 - Backup owner
- Owns: backup branches in `presentation.py`
- Tasks: T5.1
- Must not modify main slide branches

### Reviewer / integrator (main agent)
- Owns: final sequencing, reviewing subagent outputs, conflict resolution,
  build/render validation, and acceptance
- Tasks: T0.*, T6.*, and all merge decisions

## Review Gates

### Gate G1 - System coherence gate
Reached after T1.* and T2.*.

Questions:
- Is the anti-repetition rule explicit enough to constrain implementation?
- Is each act visually differentiated at the spec level?
- Do slide specs stop defaulting to rounded-box patterns?

### Gate G2 - Shared primitive gate
Reached after T3.2.

Questions:
- Are the core primitives sufficient to support multiple acts?
- Did helper design drift back toward generic card builders?
- Are semantic SVGs reusable and state-consistent?

### Gate G3 - Per-act review gate
Reached after each T4.* or T5.1 delivery.

Questions:
- Does the act satisfy its intended rhythm?
- Are the revised slides visibly stronger than the old preview?
- Is there new local repetition inside the act?

### Gate G4 - Whole-deck acceptance gate
Reached after T6.4.

Questions:
- Is rounded-rectangle fatigue meaningfully reduced?
- Do section / results / mechanism / takeaway pages now look like different slide families?
- Does the deck feel like one coherent talk rather than disconnected redesign experiments?

## Immediate Execution Order

The first executable sequence should be:

1. Finish T1.1/T1.2 by updating `visual_system.md`.
2. Finish T2.1/T2.2/T2.3 by updating `slide_visual_spec.md`.
3. Finish T3.1/T3.2 by strengthening shared helper infrastructure in `presentation.py`.
4. Split implementation by act: T4.1-T4.5 and T5.1.
5. Run T6.1-T6.5.

This ordering maximizes coherence while still allowing later parallelism.
