# RCA Visual System

This document defines the implementation-facing visual system for
`examples/4.15-talk/`. It replaces ad hoc page styling with a coherent,
repeatable language that can be applied across the full deck.

The deck is already structurally sound at the story level. The redesign target is
presentation quality: stronger stage presence, cleaner hierarchy, richer diagram
language, and much lower dependence on repeated rounded-card layouts.

## Style Summary

- `style_name`: research-line keynote for RCA evaluation
- `style_keywords`: diagram-led, assertive, evidence-first, semantically visual,
  less card-centric, conference-grade, academically credible
- `audience_fit`: adjacent-field academic audience that can follow systems/SE/LLM
  concepts, but needs very rapid visual orientation during a live talk

The deck should feel like one coherent talk with a learned visual grammar, not a
sequence of paper-summary slides that happen to share a color palette.

## Core Intent

Within 1-2 seconds, the audience should be able to tell:

- what kind of object they are looking at
- whether the object is context, evidence, suspicion, or confirmed fault
- where the slide's main focus is
- whether the page is establishing stakes, showing mechanism, or delivering evidence

This system is not decorative. It is designed to reduce search cost and make the
speaker's reasoning easier to follow in real time.

## Palette Roles

Use existing semantic theme tokens, but apply them with stricter discipline.

- `primary`: our framework, trusted structure, stitched synthesis, final claims
- `secondary`: prior regime, baseline context, muted comparison state
- `accent`: method motion, transitions, gates, process scaffolding
- `positive`: validated evidence, retained cases, robust support
- `negative`: failure modes, hidden defects, trust gaps, root-cause emphasis
- `warning`: risk states, discarded cases, mismatch, fragile intermediate states
- `text` / `text_mid`: explanation, labels, secondary annotation
- `bg` / `bg_alt` / `bg_accent`: depth planes and chapter rhythm, not decorative tint

Color may encode state and confidence, but never identity by itself. Every major
semantic distinction must also have a shape, outline, icon, line style, texture,
or positional cue.

## Type Scale

Relative type behavior matters more than exact point sizes.

- `hero_title`: use for slides 01 and 32 only; large and decisive
- `section_question`: large but narrower than hero titles; used on section entries
- `slide_title`: default main-slide title hierarchy
- `claim`: single-sentence focal conclusion for takeaways and verdict strips
- `body`: compact explanatory text only where the visual cannot carry the claim
- `caption`: direct annotations, axis notes, data-source labels, slide-local footers
- `mono_protocol`: use sparingly for task schema, tuple output, or metric notation

Rules:
- prefer one large text anchor over many medium text regions
- keep supporting text close to the related visual
- avoid full-width explanatory paragraphs under diagrams

## Spacing Scale

- `tight`: local chip spacing, axis labels, telemetry bundles
- `medium`: default spacing inside support groups
- `wide`: major zone separation on mechanism and evidence slides
- `reset`: deliberate whitespace on section entries, takeaways, and closing slides

Rules:
- whitespace is part of emphasis, not empty remainder
- boundary slides must visibly breathe more than evidence slides
- dense technical backup slides may use tighter spacing, but not collapse hierarchy

## Layout Families

Every slide should commit to one of these families.

### 1. Hero
Use for title, stitched synthesis, and final close.

Characteristics:
- one dominant visual argument
- high whitespace
- strong thesis or final statement
- one supporting payoff or action region

Typical slides:
- 01, 30, 32

### 2. Diagnostic Contrast
Use for slides where the argument is based on a mismatch between two modes,
regimes, or reasoning paths.

Characteristics:
- asymmetry is preferred over mirrored columns
- one side may be shallower, faster, or more dangerous
- conclusion should be expressed as a verdict band or direct contrast label

Typical slides:
- 04, 13, 22, 23, 27

### 3. Evidence Stage
Use for numeric result pages.

Characteristics:
- one dominant chart or quantitative focal object
- one compact verdict region integrated with the figure
- direct labels and deltas preferred over detached summary cards

Typical slides:
- 08, 12, 13, 19, 20, 27, 28

### 4. System Diagram
Use for RCA task definitions, benchmark pipelines, propagation logic,
verification flows, and supervision structures.

Characteristics:
- semantic SVG or structured diagram is primary
- text explains interpretation, not mechanics the figure could show itself
- stage grouping should encode functional hierarchy, not just left-to-right order

Typical slides:
- 03, 10, 11, 17, 24, 25, 26, 31

### 5. Reset / Boundary
Use for section entry, act close, bridge, or agenda transitions.

Characteristics:
- low density
- one question, one sentence, or one bridge
- large whitespace and cleaner pacing rhythm

Typical slides:
- 06, 15, 16, 21, 29

## Shape Families

The deck must diversify geometry while keeping semantics stable.

### Ring / Circle
Use for:
- phase markers
- audit loops
- validation cycles
- gated synthesis anchors

Do not use for:
- generic content containers

### Capsule / Pill
Use for:
- tags
- compact KPI chips
- threshold labels
- small verdict fragments
- act markers

Do not use for:
- long-form explanatory containers

### Chevron / Arrow Block
Use for:
- flow progression
- falsification logic
- filter stages
- explicit dependency
- staged reasoning

Do not use for:
- decorative motion with no logical consequence

### Notched Panel
Use for:
- contract definitions
- methodology rules
- protocol statements
- compact interpretation notes

Do not use for:
- all-purpose text boxes

### Track / Brace / Band
Use for:
- grouped comparison
- roadmap sequencing
- evidence strips
- stitched act summary
- bridge logic between stages

Do not use for:
- large blocks of prose

### Semantic SVG Object
Use for:
- service nodes
- anomaly signals
- root-cause marks
- propagation paths
- telemetry evidence chips
- intervention points
- checkpoint supervision

### RoundedBox
Allowed only as a secondary support surface.

Rules:
- never make `RoundedBox` the default slide skeleton
- do not allow it to dominate consecutive pages
- prefer bands, notches, tracks, or semantic objects when a container is needed

## Anti-Repetition Rules

These are hard implementation constraints.

- No three consecutive slides may share the same dominant geometry.
- No three consecutive slides may be visually dominated by rounded rectangles.
- No equal-weight three-card composition should recur in back-to-back pages.
- Results slides may not all use the same chart-plus-summary-card template.
- Section entry, act close, and final close slides must each have distinct dominant structure.
- If a slide shows three peer concepts, only use the same geometry for all three when
  explicit equivalence is the message.
- Whenever a chart can carry a conclusion with direct annotation, do not move that
  conclusion into detached cards.

## Act Rhythm Map

### Act I - Stakes and evaluation contract
Message mode:
- why RCA matters
- why it is hard
- why current evaluation can mislead
- what questions structure the talk

Visual behavior:
- big statements
- strong contrast
- learned vocabulary introduced gradually
- less detail than later acts, but stronger hierarchy

Preferred structures:
- hero bands
- contrast boards
- gate ladders
- thesis strips

### Act II - Realism stress test
Message mode:
- benchmark falsification
- realism principle
- benchmark rebuild
- collapse under realistic conditions

Visual behavior:
- method plus first evidence shock
- more mechanical structure
- one or two hard evidence punches

Preferred structures:
- chevron probe logic
- filter / gate geometry
- triptych mechanism panels
- regime-shift evidence layouts

### Act III - Capability under real telemetry
Message mode:
- real task contract
- scale burden
- capability gap under realistic telemetry

Visual behavior:
- protocol-like clarity
- scale must feel heavy
- results need to read as operational capability limits

Preferred structures:
- contract map
- telemetry bundle
- large-number massing
- gain-vs-gap evidence stage

### Act IV - Trustworthiness and process verification
Message mode:
- correct labels can hide wrong reasoning
- outcome-only metrics are insufficient
- forward verification and step supervision add process scrutiny
- trust gap becomes measurable

Visual behavior:
- strongest semantic diagram language in the deck
- audit and verification motifs
- hidden defect exposure

Preferred structures:
- dual-path causal contrasts
- audit rings
- verification loops
- checkpoint tracks
- reasoning break markers

### Act V - Synthesis and agenda
Message mode:
- stitch three works into one research line
- show next agenda
- close on reliable AI operations

Visual behavior:
- cleaner and more conclusive
- stronger whitespace
- reduced detail, stronger consequence

Preferred structures:
- stitched ladder
- roadmap track
- manifesto close
- synthesis banding

### Backup slides
Message mode:
- support Q&A without turning into appendix sludge

Visual behavior:
- slightly denser than main deck
- same semantic language
- less theatrical staging, same design discipline

## RCA Semantic Vocabulary

### Service
Purpose:
- represent a software component in the system graph

Base form:
- module-like object with clear identity region, interface hints, and internal signal lane

States:
- `healthy`
- `suspect`
- `propagated`
- `root`
- `latent`

Rules:
- the same service should remain recognizable across states
- use overlays and accents to show state change rather than replacing the object class
- root cause must combine at least two encodings beyond color
- propagated must feel downstream, not origin-like

### Anomaly
Purpose:
- represent observed evidence of deviation

Rules:
- attach anomaly to service or path, not a standalone box
- use pulse, burst, threshold spike, or local badge treatment
- anomaly should read as observed symptom, not confirmed cause

### Root Cause
Purpose:
- represent the initiating fault

Rules:
- use epicenter ring, fracture notch, fault badge, or heavier outline
- root cause should read as source of cascade
- never represent it as only "the red box"

### Propagation
Purpose:
- represent causal movement from source to symptom

Rules:
- directional connector with pulse markers or flow accent
- `solid` = verified
- `dashed` = candidate
- `muted` = context
- allow short on-path annotation when the path itself is the claim

### Evidence
Purpose:
- represent modality-specific support

Glyph rules:
- `metrics`: sparkline, threshold band, or trend fragment
- `logs`: stacked event-line glyph
- `traces`: linked hop chain or span graph fragment

Rules:
- chips should be small, attachable, and readable from distance
- bundle evidence when heterogeneity itself is the point

### Validation Gate
Purpose:
- distinguish raw injections from retained, impact-validated failures

Rules:
- render as gate/filter throat rather than a normal pipeline box
- discarded cases must peel off onto a muted branch
- retained cases continue on the higher-contrast verified path

### Verification Loop
Purpose:
- express forward verification and audit logic

Rules:
- use loop, ring, or checkpoint cycle language
- show intervention and verdict as semantically distinct states
- process-aware checking should look deeper than answer-only evaluation

### Step-Wise Supervision
Purpose:
- show hop-level or path-level verification

Rules:
- each step gets a visible checkpoint or verification token
- verified path is visually more structured than answer-only supervision
- old shallow supervision should look less constrained and less auditable

## Connector Rules

- Use directional flow when causality or process order matters.
- Use line style to differentiate verified, candidate, and context paths.
- Use bands or braces for grouping rather than drawing many redundant borders.
- Do not let connectors become decorative wiring; every connector should express
  dependency, flow, or causal relation.

## Image Policy

- Redraw figures whenever only headline structure or a few values matter.
- Prefer native / deck-local visuals over screenshots for all main slides.
- Use paper figure screenshots only when exact geometry is itself evidence or redraw
  would distort the meaning.
- Any imported figure must be tightly cropped and locally annotated.
- No decorative stock imagery.

## Emphasis Rules

- Each slide gets one focal object.
- Context recedes via contrast and weight, not removal.
- Verdicts should appear close to the evidence region.
- Use scale and placement before using extra color.
- Do not create competing hot spots on the same page.
- Animation should reveal reasoning order, not decorate the page.

## Accessibility Baseline

- Avoid color-only meaning.
- Keep high contrast for all major text and focal marks.
- Keep chart labels legible in projected conditions.
- Prefer direct annotation over detached legends.
- Reduce prose density on main slides.
- Ensure diagrams remain verbally narratable.

## Immediate Refactor Targets

These are the first visible system upgrades expected in implementation.

- Replace remaining generic service rectangles with the module-style service object.
- Split anomaly and root-cause visuals consistently across the deck.
- Upgrade propagation from box-chain logic to causal-path logic.
- Replace repeated rounded-card KPI layouts with metric clusters, strips, tracks,
  or integrated chart annotations.
- Make section dividers and act-close slides distinct page families.
- Reserve rounded boxes for secondary support, not primary stage structure.

## Drift Checks

Use this list during implementation and review.

- Does each act feel visually different from the previous one?
- Are rounded rectangles still the easiest thing to notice on too many pages?
- Do charts carry their own verdicts?
- Do mechanism slides rely on semantic diagrams rather than repeated containers?
- Are section-entry slides and takeaway slides distinct from each other?
- Can a viewer tell evidence from conclusion quickly?
- Are anomaly, root cause, propagation, and verification visually separable?

## External Guidance

These references support the readability and accessibility assumptions behind the system.

- APA Convention accessibility guidance (2025):
  https://convention.apa.org/presenters/accessibility
- CSCW 2025 accessible presentation guidance:
  https://cscw.acm.org/2025/index.php/accessible-presentation-guidelines/
- UW-Madison presentation best practices (2025):
  https://brand.wisc.edu/content/uploads/2025/08/presentation-best-practices.pdf
- `SlideAudit` (UIST 2025):
  https://arxiv.org/abs/2508.03630
- `Attend to what I say` (2026):
  https://arxiv.org/abs/2601.10244
