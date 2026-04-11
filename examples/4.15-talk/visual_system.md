# RCA Visual System

This note defines the deck-wide visual language for `examples/4.15-talk/`.
It is meant to guide future SVG work so slides stop looking like generic flowcharts
and start reading like a coherent academic RCA presentation.

## Why This Matters

Recent academic talk guidance and slide-analysis work converge on the same point:
visuals should reduce search cost for the audience. In this deck, the audience should
be able to tell within 1-2 seconds:

- what is the entity (`service`, `metric`, `trace`, `root cause`)
- what is the state (`normal`, `suspect`, `propagated`, `confirmed`)
- what is the current focus (the object being explained right now)
- what is evidence versus what is conclusion

The goal is not decorative polish. The goal is semantic readability under live talk conditions.

## Best-Practice Principles for This Deck

- One focal object per slide. Context should remain visible but visually de-emphasized.
- Prefer direct labels on the visual over detached legends.
- Do not use color alone to encode meaning; pair color with outline, shape, icon, or texture.
- Keep object identity stable across states: the same service should look like the same service after it becomes suspicious or confirmed faulty.
- Use animation only to reveal reasoning order, never as decoration.
- Replace screenshots and stock icons with redrawn, deck-native visuals whenever possible.

## Semantic Tokens

### Service

Purpose: represent a software component in the system graph.

Shape rules:
- base shape is a module card, not a generic rounded box
- include a slim header strip to signal identity
- include small ports or terminals to suggest interfaces / dependencies
- body can include a tiny sparkline or telemetry slot when useful

States:
- `normal`: light fill, dark outline, low visual weight
- `suspect`: stronger outline, subtle glow or badge
- `propagated`: warm accent pulse on one side, but no fracture mark
- `root-cause`: fracture notch or fault badge plus epicenter ring

### Anomaly

Purpose: represent an observed deviation in telemetry.

Shape rules:
- anomaly should appear attached to a service, not as a standalone rectangle
- prefer pulse rings, wave bursts, threshold spikes, or local badges
- anomaly intensity can scale through halo radius or sparkline amplitude

Avoid:
- large solid warning rectangles
- making the anomaly look like a different entity than the service itself

### Root Cause

Purpose: represent the initiating fault.

Shape rules:
- combine at least two of: heavier outline, fracture mark, epicenter ring, fault chip
- root cause should look like the origin of a cascade, not simply a red node
- use a local badge such as `fault injected` or `trigger`

### Propagation

Purpose: represent causal travel from source to symptom.

Shape rules:
- use directional connectors with pulse nodes or flow accents
- `solid`: verified path
- `dashed`: candidate path / hypothesis
- `muted`: context only
- add one short takeaway directly on or near the path when the path itself is the point

### Evidence

Purpose: represent the modality supporting the causal story.

Glyph rules:
- `metrics`: sparkline, threshold band, or compact bar fragment
- `logs`: stacked text lines / event strips
- `traces`: linked hop nodes / span chain

Evidence chips should be small, attachable, and readable from a distance.

## Page-Level Usage Rules

### Motivation and setup slides
- one hero diagram max
- use strong title + one evidence diagram, not bullet-heavy explanation
- prefer a compact symbolic RCA panel over full service topology; motivation slides should show the relation between symptom, anomaly, cause, and time pressure in one glance

### Method / benchmark slides
- show process with semantic nodes, not generic SmartArt-style boxes
- highlight only the current stage and mute the rest

### Results slides
- annotate the winning or failing region directly on the chart
- suppress non-essential axes, legends, and borders
- pair one dominant chart with one interpretation strip; avoid dual-equal charts unless the slide's point is explicitly "two regimes"
- prefer delta badges and threshold lines over long result bullets
- gray context, color focus: old regime / weaker metric / non-winning models should visually recede

### Transition slides
- reduce density; preserve the same color semantics so the audience does not have to relearn the language
- use large whitespace, one core sentence, and one act marker; these pages should reset attention, not keep explaining

## Layout Families for Slides 09-32

- `Mechanism panel`: one SVG occupies 55-70% width, plus a short interpretation strip or claim box
- `Evidence slide`: one dominant chart on the right or center, plus one compact summary column with 2-3 evidence badges
- `Act close`: one claim card, one bridge strip, minimal supporting text
- `Section divider`: oversized question sentence, one recap badge, no explanatory body text

## Additional Symbol Rules for Later Acts

### Validation Gate

Purpose: distinguish "fault injected" from "operationally relevant case retained".

Shape rules:
- render as a gate or filter throat, not a normal pipeline box
- discarded cases should peel off into a muted branch with a `warning` chip
- retained cases continue as a higher-contrast verified path

### Telemetry Bundle

Purpose: represent metrics, logs, and traces as one analyzable evidence bundle.

Shape rules:
- use three attachable chips or mini cards with distinct glyphs
- bundle should read as "heterogeneous evidence arrives together", not three unrelated icons

### Step-Wise Verification

Purpose: show that propagation hops themselves are verified.

Shape rules:
- each hop gets a checkpoint token
- verified hops use solid connectors and check markers
- unverified or old-style supervision should look shallower and less structured

### Metric Semantics

Purpose: visually explain why `Pass@1` and `PR` are related but not identical.

Shape rules:
- use containment, overlap, or nested regions instead of purely textual definitions
- the audience should infer `PR <= Pass@1` in a glance before reading the sentence

## Immediate Refactor Targets

- Replace plain service rectangles in the cascade diagrams with service modules.
- Split `warning` visuals into separate `anomaly` and `root-cause` glyphs.
- Upgrade propagation chains from equal-weight boxes to causal-path diagrams.
- Standardize metrics / logs / traces as reusable evidence chips.
- Reuse the same root-cause mark across slides 02-04, 09-11, and 22-29.

## External Guidance Informing This System

These sources support the operating rules above:

- APA Convention accessibility guidance (2025): fewer words, higher contrast, avoid color-only meaning.
  https://convention.apa.org/presenters/accessibility
- CSCW 2025 accessible presentation guidance: readable text, verbalizable visuals, color-blind-safe distinctions.
  https://cscw.acm.org/2025/index.php/accessible-presentation-guidelines/
- UW-Madison presentation best practices (2025): reduce clutter, use purposeful animation, build visible hierarchy.
  https://brand.wisc.edu/content/uploads/2025/08/presentation-best-practices.pdf
- `SlideAudit` (UIST 2025): common slide failures cluster around hierarchy, layout, color, and clarity.
  https://arxiv.org/abs/2508.03630
- `Attend to what I say` (2026): aligning spoken explanation with visual focus improves comprehension.
  https://arxiv.org/abs/2601.10244
