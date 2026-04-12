# 4.15 Talk Deck Refinement Plan

## Summary

This document defines the full visual and structural refinement plan for
`examples/4.15-talk/`. The talk story is already broadly correct. The main
problem is presentation quality: too many slides rely on similar rounded-card
layouts, too many pages distribute visual weight evenly, and several evidence
or mechanism slides still read like adapted paper summaries instead of a polished
research-line talk.

The deck should be upgraded into a research-line presentation with the following
properties:

- the narrative stays stable, but every page is more intentional and more legible
- the visual system becomes stronger, more varied, and less card-centric
- RCA concepts are expressed through a reusable semantic graphic language
- results slides feel like conference-grade evidence pages
- mechanism slides feel like system diagrams, not generic flowcharts
- section and takeaway slides act as pacing resets instead of thin summary cards
- backup slides stay denser than the main talk, but still belong to the same
  visual system and avoid appendix-style default formatting

The user requested a stronger sense of stage presence and explicitly requested
less reliance on rounded rectangles. That is treated as a hard design constraint
rather than an optional refinement.

## Design Objectives

### Primary objectives

- Preserve the existing thesis, act structure, and slide count.
- Increase visual richness without making the deck decorative or noisy.
- Replace repeated rounded-box compositions with a broader, rule-driven shape
  and layout system.
- Make page families visibly distinct: hero, contrast, evidence, mechanism,
  section reset, takeaway close, roadmap, and appendix support.
- Build a reusable RCA symbol language for service, anomaly, root cause,
  propagation, telemetry evidence, intervention, and verification.
- Ensure each act has its own rhythm and does not feel like the same slide
  skeleton with different words.

### Secondary objectives

- Reduce detached bullet explanations by moving interpretation closer to charts
  and diagrams.
- Increase the amount of directly annotated evidence.
- Maintain accessibility and live-talk readability.
- Keep the deck implementable with the current SlideCraft / semantic SVG stack.

## Non-Objectives

- Do not rewrite the talk into a new story.
- Do not add significant new research content.
- Do not increase the main slide count beyond the existing 32 + 6 backup frame.
- Do not turn the deck into a product-marketing visual style.
- Do not introduce decorative imagery that does not help reasoning.

## Current Deck Diagnosis

### Structural strengths already present

- The research-line story is coherent.
- The A/B/C evaluation ladder already provides a strong organizing thesis.
- Many slides already have clear single-sentence messages.
- Existing semantic SVG work provides a good base for RCA-native visuals.

### Current weaknesses that must be corrected

- Too many slides use equal-weight card panels with similar rounded corners.
- Results pages often separate the figure from the interpretation too much.
- Mechanism pages sometimes still read like SmartArt or paper-summary flowcharts.
- Takeaway and section slides are currently too thin to create real pacing resets.
- Several pages have the correct information but not enough visual hierarchy.
- Visual diversity is under-specified, so implementation naturally falls back to
  `RoundedBox` as the default object.

### Slide types that need the strongest redesign

- Hero and title pages
- Section dividers
- Results slides with numeric shock or gap evidence
- Mechanism slides that explain realism / verification process
- Act-close takeaway slides
- Final synthesis and closing pages

## Deck-Level Visual System Decisions

### Style positioning

The deck should read as:

- research-line keynote
- diagram-led academic talk
- assertive but not flashy
- evidence-first
- semantically visual rather than card-heavy

### Layout families

Every slide must belong to one of the following layout families.
Implementation should not improvise new page structures unless the existing
families fail to express the message.

1. `Hero`
   - Use for title, major synthesis, and final close.
   - High whitespace, strong thesis statement, one large focal visual.
2. `Diagnostic Contrast`
   - Use for shortcut vs causal, old vs new, answer vs reasoning,
     outcome-only vs process-aware.
   - Non-symmetric contrast is preferred unless equal weighting is the point.
3. `Evidence Stage`
   - Use for chart-led slides.
   - One dominant evidence object plus a compact verdict or interpretation zone.
4. `System Diagram`
   - Use for pipelines, generation frameworks, task definitions, causal flows,
     verification logic, and supervision paths.
   - Prioritize semantic SVGs over generic containers.
5. `Reset / Boundary`
   - Use for section dividers, act closes, bridge slides, and agenda/roadmap.
   - Low density, strong rhythm, strong transition logic.

### Shape families

The deck must use a controlled but diverse set of shape families. Rounded boxes
can remain in the toolbox, but they are demoted to secondary support elements.

1. `Ring / Circle`
   - Phase gates, audit loops, closed validation logic, act markers.
2. `Capsule / Pill`
   - Status labels, metric chips, threshold bands, compact tags.
3. `Chevron / Arrow Block`
   - Progression, flow, dependency, filtering, staged reasoning.
4. `Notched Panel`
   - Contract statements, method notes, protocol definitions, rule boxes.
5. `Track / Brace / Band`
   - Comparison scope, evidence strips, roadmap rails, grouping.
6. `Semantic SVG Object`
   - Service nodes, anomaly bursts, root-cause marks, propagation chains,
     intervention points, verification checkpoints.
7. `RoundedBox`
   - Secondary support only; not the deck default.

### Anti-repetition rules

These are hard constraints, not style suggestions.

- No three consecutive slides may share the same dominant geometry.
- No three consecutive slides may use `RoundedBox` as the main visual container.
- No more than one large equal-weight card grid should appear in any four-slide window.
- Section slides, takeaway slides, evidence slides, and mechanism slides must each
  have visibly different dominant geometries.
- If a slide contains three parallel objects, they should only share the same
  geometry when the slide's argument depends on explicit equivalence.
- A chart slide may not rely on a chart plus detached summary cards if the same
  conclusion can be encoded inside the chart with direct annotation.

### Act rhythm map

#### Act I - Stakes and evaluation contract

Visual tone:
- big statements
- strong contrast
- audience orientation
- thesis anchoring

Preferred geometries:
- hero bands
- ladder / gate forms
- contrast boards
- evidence strips

#### Act II - Realism stress test

Visual tone:
- methodological challenge
- first shock evidence
- mechanism diagnosis

Preferred geometries:
- falsification tracks
- filter/gate diagrams
- regime-shift evidence layouts
- triptych mechanism panels

#### Act III - Capability under real telemetry

Visual tone:
- task protocol
- scale pressure
- measurable capability gap

Preferred geometries:
- contract map
- telemetry bundles
- large-scale evidence staging
- gain-vs-gap structures

#### Act IV - Trustworthiness and process verification

Visual tone:
- audit
- causal scrutiny
- process supervision
- hidden defect exposure

Preferred geometries:
- dual-path causal diagrams
- audit rings
- verification loops
- stepwise tracks
- reasoning break markers

#### Act V - Synthesis and agenda

Visual tone:
- stitched thesis
- clean conclusion
- forward-looking roadmap

Preferred geometries:
- stitched ladder
- roadmap track
- manifesto close
- summary band

#### Backup slides

Visual tone:
- denser technical support
- credible appendix pages
- same visual language, less theatrical staging

Preferred geometries:
- compact system diagrams
- table-lite grids
- detailed protocol panels
- evidence support layouts

## RCA Semantic Visual Language

### Service nodes

Purpose:
- represent software components as stable system objects rather than generic boxes

States:
- `healthy`
- `suspect`
- `propagated`
- `root`
- `latent / hidden`

Rules:
- service identity should stay stable across state transitions
- state changes should layer onto the same base object
- root cause must use at least two channels beyond color
- propagated should not look identical to root cause

### Anomaly

Purpose:
- represent observed deviation in telemetry

Rules:
- anomaly attaches to a service or path, not a standalone card
- use pulse rings, bursts, threshold spikes, or local halo treatment
- anomaly is not the same visual object as root cause

### Root cause

Purpose:
- represent the initiating fault

Rules:
- use fracture notch, epicenter ring, heavy outline, or fault badge
- root cause should read as an origin event
- root cause should not be encoded only as a red node

### Propagation

Purpose:
- represent causal travel or inferred dependency

Rules:
- directional paths with pulse markers
- `solid` = verified
- `dashed` = candidate
- `muted` = contextual background
- allow short direct annotation on path when the path is itself the point

### Evidence chips

Purpose:
- identify metrics, logs, and traces without heavy reading

Rules:
- metrics: sparkline / threshold / trend glyph
- logs: stacked event-line glyph
- traces: hop-chain glyph
- chips should be compact and attachable to service/path visuals

### Validation / verification

Purpose:
- express impact validation and later process auditing in a way distinct from
  standard pipelines

Rules:
- use gate / throat / checkpoint / audit loop forms
- distinguish retained cases from discarded cases via path split, not text alone
- process-aware verification should visually look deeper than answer-only checking

## Information Density and Evidence Rules

### Results slides

- One dominant evidence object per slide.
- Put the main verdict inside or directly against the chart.
- Suppress non-essential legend, border, and axis noise.
- Use direct labels and deltas where possible.
- If two charts appear, the slide message must genuinely require two regimes.
- Use color and emphasis to rank evidence importance.

### Mechanism slides

- Prefer semantic system diagrams to generic pipeline cards.
- Use one focus object and one interpretation zone.
- Group stages by function, not just order.
- Avoid repeated equal-width process boxes.

### Transition and takeaway slides

- They exist to reset attention, not add another mini-content page.
- Use whitespace deliberately.
- Show one large idea and one bridge or consequence.
- Make act-close slides feel different from section-entry slides.

## Slide-by-Slide Rewrite Strategy

### Slides 01-06: opening and framing

#### 01 - Building Trustworthy RCA Evaluation for LLM Agents
- Use `Hero` family.
- Left: thesis and research-line ladder.
- Right: audience payoff / why this matters.
- Milestones should become anchored timeline / research-line markers, not loose badges.
- Avoid summary-page sparsity.

#### 02 - RCA Failures Are Expensive and Cascading
- Use a three-layer composition: stakes strip, cascade scene, operator pressure.
- Make cost and MTTR pressure feel immediate.
- The cascade visual should be larger and more causal.
- Avoid four equal KPI cards.

#### 03 - RCA Requires Causal Multi-Modal Multi-Hop Reasoning
- Build a modality-to-burden map.
- Metrics, logs, and traces should use distinct shapes.
- The slide should show how evidence and causal hops interact.
- Avoid three identical modality cards.

#### 04 - Current Evaluation Often Rewards Shortcut Correlations
- Use `Diagnostic Contrast` family.
- Make shortcut path visually faster and shallower.
- Make causal path deeper and evidence-audited.
- End with a mismatch verdict strip.
- Avoid symmetric two-column card layout.

#### 05 - Our Evaluation Ladder Uses Three Questions
- Convert from motif repetition to research contract slide.
- A/B/C should behave like gates, not just labels.
- Show dependency semantics explicitly.
- Avoid duplicating slide 01's ladder.

#### 06 - What You Should Remember in 45 Minutes
- Make this a listening guide.
- Put decision rule first.
- Show three watch-for signals with act references.
- Avoid equal-width summary cards.

### Slides 07-15: realism act

#### 07 - If a Simple Heuristic Wins, the Benchmark Is Too Easy
- Show falsification logic using chevron or decision-track layout.
- Place fairness and falsification criteria in compact side bands.
- Avoid casual baseline framing.

#### 08 - SimpleRCA Matches or Beats SOTA on Public Benchmarks
- First hard shock slide.
- Chart dominates.
- Summary is compact and integrated.
- Exception case should be chart-annotated, not parked in text.

#### 09 - Legacy Benchmarks Are Oversimplified by Construction
- Use a triptych with three distinct visual mechanisms.
- Each mechanism should have a different micro-graphic.
- Avoid three same-shape info cards.

#### 10 - Realism Needs Impact-Validated Failures, Not Raw Injections
- Use filter/gate geometry.
- Show retain/discard branches explicitly.
- SLI impact should function as the actual decision gate.

#### 11 - Fault-Propagation-Aware Benchmarking at Scale
- Six-stage framework should feel like a construction system.
- Use grouped stage map with hierarchy.
- Put reproducibility signals in peripheral notes.
- Avoid classroom flowchart feel.

#### 12 - New Benchmark Stats Show a Different Difficulty Regime
- Present numbers as regime shift, not KPI scatter.
- Use metric clusters or strips instead of many independent cards.
- Add one strong explanatory panel.

#### 13 - Performance Collapses Under Realistic Conditions
- Use dual evidence panels with different internal geometry.
- Accuracy collapse and runtime escalation should feel like two linked shocks.
- Avoid chart plus detached explanation card.

#### 14 - Three Failure Modes Explain the Collapse
- Each failure mode gets a distinct scene or diagram metaphor.
- Keep examples brief but specific.
- Tie the three back together with a common verdict.

#### 15 - Takeaway A
- Low-density act close.
- One large statement, one bridge to capability.
- Do not let it read like another content card.

### Slides 16-21: capability act

#### 16 - Now We Ask the LLM Capability Question
- Section boundary slide.
- Use chapter band / question marker.
- Avoid a generic transition strip only.

#### 17 - OpenRCA Defines RCA as a Goal-Driven Task
- Present as a protocol / contract map.
- Query, telemetry, and output should feel like different logical layers.
- Use monospaced contract styling where useful.

#### 18 - OpenRCA Forces Reasoning Over Real Telemetry Scale
- Make size pressure visceral.
- Numbers should be large and carry mass.
- Use telemetry bundle or modality massing.
- Avoid small, evenly weighted KPI boxes.

#### 19 - Current LLMs Struggle on Core RCA Tasks
- Strong results stage layout.
- Let the chart carry the conclusion.
- Put failure diagnosis near the figure, not in a separate card stack.

#### 20 - Execution-Based RCA-Agent Helps but Does Not Close the Gap
- Present as gain -> ceiling -> residual gap.
- Do not frame as simple before/after bars plus summary card.

#### 21 - Takeaway B
- Reuse act-close rhythm, but shift the semantic center to capability gap.
- Keep it sparse and transitional.

### Slides 22-29: trustworthiness act

#### 22 - Correct Labels Can Still Hide Wrong Reasoning
- Use dual-path causal contrast.
- One path shows superficial correlation; the other causal faithfulness.
- The slide should make the hidden defect intuitive at a glance.

#### 23 - Outcome-Only Evaluation Misses Process Failures
- Use layered audit structure or evaluation funnel.
- Outcome-only should feel shallow.
- Process-aware should feel deeper and stricter.

#### 24 - FORGE Uses Forward Verification from Known Interventions
- Use intervention-centric verification logic.
- Make the intervention point the visual origin.
- Show forward check and verdict gates.

#### 25 - OpenRCA 2.0 Adds Step-Wise Causal Supervision
- Use a step track or checkpoint sequence.
- Distinguish answer-only supervision from path-aware supervision.
- Avoid a row of equal rounded tokens.

#### 26 - Process Metrics Separate Identification from Reasoning
- Use audit ring or center-periphery metric split.
- Make the separation between answer correctness and reasoning faithfulness visually self-evident.
- This is one of the deck's core explanatory slides and should be highly polished.

#### 27 - Best-Model Gap Shows Hidden Reasoning Defects
- Visualize the fracture between answer quality and reasoning quality.
- Gap itself should be the focal object.
- Avoid chart plus red summary card pattern.

#### 28 - The Trust Gap Widens Across 7 LLMs
- Use a faceted or grouped comparison layout.
- Integrate interpretation into the figure region.
- Make the population trend legible immediately.

#### 29 - Takeaway C
- Strong final act close.
- One sentence, one support strip, strong trust verdict.
- Distinct from slides 15 and 21.

### Slides 30-32: synthesis and close

#### 30 - One Research Line: Realism -> Capability -> Trust
- Use stitched ladder or gated synthesis chain.
- Summarize the dependency logic and outcome of all three questions.
- Should feel more consequential than a repeated motif.

#### 31 - Next Agenda: Process-Aware Data, Agents, and Training
- Use roadmap track, not a three-card future-work list.
- Three directions should read as milestones or workstreams.
- The page should feel forward-looking, not appendix-like.

#### 32 - Build Evaluations That Make Reliable AI Operations Possible
- Use `Hero` family again, but not a copy of slide 01.
- Strong statement, one final ladder or evidence band, one action line.
- Close with whitespace and confidence.

### Backup slides B1-B6

#### General backup rule
- Denser than the main deck.
- Same semantic language.
- Less theatrical, but never default appendix formatting.

#### B1 - Why SimpleRCA Wins on Legacy Benchmarks
- Single case mechanism support.
- Reuse realism contrast language.

#### B2 - Fault-Propagation-Aware Benchmark Construction Details
- Detailed construction system page.
- More thresholds and criteria are allowed.
- Avoid full prose or default tables.

#### B3 - OpenRCA Task Granularity and Metric Definition
- Protocol-heavy page.
- Taxonomy and metrics should be visually separated.

#### B4 - RCA-Agent Strengths and Failure Patterns
- Use gain-mechanism plus residual-failure structure.
- Avoid bullet-only narrative.

#### B5 - FORGE Verification Pipeline and Metric Math
- Use detailed verification logic plus formula panel.
- Formulas may appear, but should be framed and grouped visually.

#### B6 - Threats to Validity Across the Whole Research Line
- Use matrix-lite or structured risk board.
- Avoid a dense traditional table if a visual grid can do the job.

## Component Library Plan for `presentation.py`

The implementation should add or strengthen reusable deck-local primitives.

### Reusable structural builders
- `build_section_banner`
- `build_takeaway_strip`
- `build_metric_cluster`
- `build_regime_shift_panel`
- `build_triptych_panel`
- `build_diagnostic_contrast`
- `build_roadmap_track`
- `build_manifesto_close`

### Reusable RCA semantic SVG helpers
- service node with 5 visual states
- anomaly burst variants
- root-cause marker variants
- candidate vs verified propagation path
- telemetry evidence chips
- validation gate
- intervention point
- verification loop
- checkpoint / stepwise supervision track
- reasoning break marker

### Implementation rules
- One slide may use at most one dominant large-area container.
- If text needs a container, prefer bands, notches, tracks, or capsules before `RoundedBox`.
- Results pages should not duplicate the same chart-plus-summary-card structure.
- Section and takeaway builders should not share the same skeleton.

## Verification Workflow

After implementation, validate in the following order:

1. Build `talk_4_15.pptx` successfully from `presentation.py`.
2. Render full preview PNGs.
3. Run layout review and confirm no new severe overlap or clipping.
4. Conduct a manual visual pass by page family.
5. Conduct a dedicated diversity audit.

### Manual review checklist
- Does slide 01 feel like a hero opening?
- Do section dividers reset pacing?
- Do mechanism slides use semantic diagrams rather than generic process boxes?
- Do results slides integrate verdicts into the figure region?
- Do act-close slides create breathing room?
- Do backup slides feel technical but visually intentional?

### Diversity audit checklist
- Are there any 3-slide runs with the same dominant geometry?
- Are there any 3-slide runs dominated by rounded rectangles?
- Do results slides vary in internal composition?
- Are the three act-close slides visibly different?
- Are section entry slides and closing slides distinct from each other?
- Is the roadmap slide visibly different from earlier triptych or card structures?

## Acceptance Criteria

The refinement is successful when all of the following are true:

- The deck still supports the same story and timing.
- Rounded-box fatigue is visibly reduced across the full deck.
- Each act has a distinct visual rhythm.
- Charts and diagrams carry their own conclusions more directly.
- Mechanism slides look RCA-native rather than generic.
- Section and takeaway slides function as pacing tools.
- Backup slides remain aligned with the main visual system.
- The final deck feels like a polished research-line talk rather than a collection of paper-summary slides.
