# SlideCraft Layout And Review Design

**Goal:** Make PPT authoring in `paperops.slides` less manual by improving automatic layout negotiation and by adding a complete review loop that lets an agent reliably perceive the current state of a deck before iterating.

**Scope:** This spec covers SlideCraft layout semantics, deck review tooling, and skill documentation under `.claude/skills/`. It does not introduce high-level slide templates or change existing example deck content beyond what is needed to validate the new workflow.

## Problem

Recent PPT work exposed a structural gap in the toolchain:

1. Authors can generate decks, but the system still requires too much manual box sizing and local tweaking.
2. The layout engine, preview renderer, and saved-file checker are not yet a unified perception loop.
3. Agents can build slides, but they still lack a first-class workflow for understanding what the current deck actually looks like and where the iteration pressure is.

This creates a fragile authoring loop:

- code is edited
- `.pptx` is generated
- preview images are inspected manually
- sizing bugs are diagnosed ad hoc
- fixes are applied one slide at a time

The result is usable, but not yet self-reinforcing. The system should let an agent generate a deck, review the deck in a structured way, identify likely causes, and iterate with minimal manual control.

## Design Principles

### 1. Intrinsic sizing before manual dimensions

Every content component should be able to express what size it wants, what size it can shrink to, and when it should wrap. Manual `width` and `height` remain available, but should become the exception rather than the default authoring strategy.

### 2. Layout should negotiate, not just place

Containers should act more like frontend layout primitives. Authors should be able to say:

- this item should fit its content
- this item can grow to fill remaining space
- this item may shrink, but not below a readable minimum
- this row may wrap if the horizontal budget is exhausted

The current system mostly supports explicit dimensions plus `preferred_size()`. The new system should preserve that API surface but add clearer constraint semantics.

### 3. Deck review is part of authoring

The system should treat review artifacts as first-class outputs:

- in-memory layout report
- saved `.pptx` structural checks
- preview PNGs
- per-slide issue summaries

An agent should not need to manually stitch together `review()`, `preview()`, and `check_presentation()` to understand a deck.

### 4. Skill guidance should encode the loop

The `.claude/skills` guidance should explicitly push the workflow:

`plan slide logic -> build deck -> run integrated review -> inspect preview/state -> make targeted fixes`

This should become the default behavior for PPT work in the repo.

## Proposed Changes

## A. Layout Semantics Upgrade

### A1. Extend `LayoutNode` sizing semantics

Add lightweight constraint-style sizing properties to `LayoutNode`:

- `size_mode_x`: `fixed | fit | fill`
- `size_mode_y`: `fixed | fit | fill`
- `grow`: numeric weight, default `0`
- `shrink`: numeric weight, default `1`
- `basis`: optional preferred size override used during negotiation
- `wrap`: boolean for containers/components that support wrapping
- existing `width`, `height`, `min_width`, `min_height` remain supported

Intent:

- `fixed`: respect explicit width/height
- `fit`: size to intrinsic content
- `fill`: absorb available space
- `grow`: divide leftover space among eligible siblings
- `shrink`: define how aggressively a child may compress under pressure

This is intentionally smaller than CSS flexbox. The goal is not to reimplement web layout, only to give SlideCraft enough expressive power to stop depending on repeated manual box dimensions.

### A2. Improve intrinsic sizing contracts

Strengthen intrinsic sizing for core content components:

- `TextBlock`
- `BulletList`
- `Badge`
- `Callout`
- `RoundedBox`
- `Flow`
- `Flowchart` nodes

Each should expose reliable intrinsic size behavior derived from shared measurement logic, especially for:

- text wrapping
- inner padding
- minimum readable width
- multi-line height growth

This should continue using the shared text measurement utility, but with a clearer contract: the same sizing assumptions should drive layout estimation, saved-file overflow checks, and preview text wrapping.

### A3. Container negotiation improvements

Upgrade `HStack` and `VStack` from "place children sequentially" to "negotiate available space using intrinsic size + constraints".

Required behavior:

- allocate fixed-size children first
- compute intrinsic demand for `fit` children
- allocate remaining space to `fill` children by `grow`
- if over-constrained, shrink children by `shrink`, while respecting min sizes
- support optional row wrapping for horizontal stacks where requested

Non-goals:

- full CSS parity
- arbitrary nested alignment solver
- auto-pagination

### A4. Author ergonomics

Authors should be able to express common intent more directly. The API should support patterns like:

- content-fit badge next to a fill-width title block
- flow nodes that auto-size to labels while arrows stay narrow
- callout boxes that grow vertically with text but align with neighboring blocks

The concrete API names may vary, but the feature must reduce the need to hand-tune widths for ordinary research decks.

## B. Integrated Deck Review Pipeline

### B1. Add a unified review entry point

Introduce a single high-level deck review API, for example:

`prs.review_deck(...)`

The exact method name may vary, but it must produce a combined review artifact containing:

- build-time layout issues
- saved-file checks from `check_presentation()`
- preview image paths
- slide-level summary entries

The goal is to let an agent call one thing and get a coherent view of the current deck.

### B2. Define the review artifact shape

The returned structure should be machine-friendly and stable enough for agent iteration. At minimum:

- `total_slides`
- `layout_issue_count`
- `saved_issue_count`
- `preview_paths`
- `slides`: list of per-slide summaries

Each per-slide summary should include:

- `slide_number`
- `title` if known
- `layout_issues`
- `saved_file_issues`
- `preview_path` if generated
- heuristic metadata such as:
  - text shape count
  - likely dense regions
  - potential crowding risk

This is not meant to be perfect computer vision. It is a structured bridge between generation and inspection.

### B3. Improve perceptual checks

The review pipeline should incorporate heuristics useful for agent-driven iteration:

- identify slides with many small text boxes
- identify shapes with very low width-to-text ratios
- flag components likely to be label-crowded
- surface the top N problematic slides rather than only returning a flat issue list

These checks should stay deterministic and lightweight. The system does not need OCR or external vision models for this phase.

### B4. Preview outputs as first-class artifacts

Preview rendering should be treated as a review product, not a side helper. The integrated review output should always make it easy to locate the exact PNG for each slide.

The review pipeline should support:

- rendering all slides or a subset
- a stable output directory
- replacing stale preview assets when rerendering

## C. Skill Layer Updates

### C1. Expand `.claude/skills/slidecraft`

Update the existing skill so it documents the improved workflow:

1. Plan the logic chain of the deck.
2. Build the deck in code.
3. Run integrated review.
4. Inspect deck state via preview PNGs and issue summaries.
5. Make targeted fixes, not blind local tweaks.

The skill should explicitly teach:

- when to trust intrinsic sizing
- when to add manual size constraints
- how to interpret review artifacts
- how to debug whether a problem belongs to content, layout, preview, or saved-file rendering

### C2. Add a dedicated slide review skill

Create a new `.claude/skills` entry focused on reviewing existing decks and iterating on them.

Its job is not to explain SlideCraft API generally. Its job is to teach a review workflow:

- generate or load the current deck
- run the integrated review pipeline
- inspect preview outputs
- identify whether issues come from layout negotiation, text density, component sizing, or preview mismatch
- propose targeted fixes

This separates "build slides" from "diagnose deck state", which matches how agents actually work.

## D. Workflow Changes

### D1. Default PPT iteration loop

The target authoring loop becomes:

1. edit deck code
2. generate `.pptx`
3. run integrated deck review
4. inspect reported problematic slides
5. apply targeted layout/content changes
6. rerun integrated deck review

### D2. Minimize manual control

The system should prefer:

- intrinsic text sizing
- fit/fill container semantics
- component-level minimum sizes
- deck-level review guidance

The system should avoid requiring:

- arbitrary hand-written widths for ordinary labels
- repeated ad hoc slide-by-slide spacing fixes
- manual stitching of separate review commands

## File-Level Plan

Expected implementation areas:

- `.claude/skills/slidecraft/SKILL.md`
- create new skill under `.claude/skills/` for slide review
- `src/paperops/slides/layout/containers.py`
- `src/paperops/slides/layout/engine.py`
- `src/paperops/slides/layout/auto_size.py`
- `src/paperops/slides/components/text.py`
- `src/paperops/slides/components/shapes.py`
- `src/paperops/slides/components/composite.py`
- `src/paperops/slides/preview.py`
- `src/paperops/slides/build.py`

Examples may be updated only as needed to exercise the new workflow.

## Risks

### 1. Over-designing the layout model

If the new sizing model tries to become full CSS flexbox, implementation complexity will spike and behavior will become hard to reason about. Keep the feature set intentionally small.

### 2. Divergent measurement paths

If layout sizing, preview wrapping, and saved-file checks continue using subtly different assumptions, the system will regress into the same confusion. Shared measurement behavior is more important than perfect measurement accuracy.

### 3. Review output without prioritization

A large raw issue list is not enough. The integrated review result must help an agent focus on the slides most likely to need attention.

## Success Criteria

The work is successful if:

- common research-deck layouts require fewer hard-coded dimensions
- generated review artifacts let an agent identify problematic slides without manual guesswork
- preview outputs are easy to correlate with issue summaries
- the `.claude/skills` guidance makes this review-first workflow discoverable and repeatable

## Out Of Scope

- new high-level slide templates
- automatic rewriting of slide content
- image-based or LLM-based visual critique
- changing existing test scope purely for broader coverage

