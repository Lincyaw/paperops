# Slide Visual Spec

## Header

```yaml
talk_style: research-line
audience_type: academic audience
audience_distance: adjacent-field
duration_min: 45
language_policy: English slide titles/text + Chinese speaker notes
main_slide_count: 32
backup_slide_count: 6
```

## Deck-Level Visual Strategy

### Visual Positioning

- The deck should read as a research-line keynote rather than a stitched paper summary.
- Every slide must choose one dominant visual family and one clear focal object.
- Rounded rectangles are secondary support surfaces, not the default page grammar.
- Charts should carry their own verdicts whenever possible.
- Mechanism slides should use semantic RCA diagrams rather than generic process boxes.

### Allowed Layout Families

- `hero`: thesis, stitched synthesis, closing manifesto
- `diagnostic_contrast`: mismatch, shortcut vs causal, answer vs reasoning
- `evidence_stage`: chart-led slides with integrated verdict
- `system_diagram`: protocol, pipeline, propagation, verification, supervision
- `reset_boundary`: section entry, act close, bridge, roadmap reset

### Geometry Rotation Rules

- Do not repeat the same dominant geometry across three consecutive slides.
- Do not let `RoundedBox` dominate three consecutive slides.
- Avoid repeating equal-weight three-card layouts in adjacent slides.
- Vary results-page composition across the deck: bar-led, gap-led, strip-led, faceted,
  or chart-with-band layouts.

### Semantic Vocabulary Rules

- `service`: stable module object with state overlays
- `anomaly`: pulse / burst / threshold deviation attached to object or path
- `root`: epicenter or fracture-marked origin state
- `propagation`: directed candidate or verified path
- `evidence`: metrics / logs / traces chips with distinct glyphs
- `verification`: gate, loop, checkpoint, or track forms

### Evidence Rules

- Integrate 1-2 sentence conclusion into the evidence region.
- Use detached summary cards only when the chart cannot absorb the conclusion.
- Prefer threshold bands, delta chips, braces, or direct labels over extra small cards.
- Context should recede in gray or low-contrast tones.

### Transition and Takeaway Rules

- Section entries reduce density and reset attention.
- Takeaway slides are verdict pages, not mini-summary grids.
- The three act-close slides must not share the same skeleton.
- The final slide must feel like a closing statement, not a fourth summary page.

## Main Slide Visual Specs

```yaml
main_slides:
  - slide_no: "01"
    title_en: "Building Trustworthy RCA Evaluation for LLM Agents"
    visual_role: "hero opener"
    layout_pattern: "hero thesis left + audience payoff right + stitched ladder lower band"
    dominant_geometry: "banded hero + gated ladder"
    secondary_geometry: "capsule metadata"
    focus_object: "research-line ladder connecting realism, capability, and trust"
    evidence_mode: "thesis framing"
    anti_pattern: "sparse title slide or badge pile"
    implementation_surface: "native layout + custom ladder SVG"

  - slide_no: "02"
    title_en: "RCA Failures Are Expensive and Cascading"
    visual_role: "stakes builder"
    layout_pattern: "stakes strip top + cascade scene center + operator pressure side rail"
    dominant_geometry: "band + semantic cascade scene"
    secondary_geometry: "capsule KPI chips"
    focus_object: "incident cascade from symptom to costly MTTR pressure"
    evidence_mode: "operational stakes"
    anti_pattern: "four equal KPI cards and a tiny diagram"
    implementation_surface: "native layout + semantic SVG scene"

  - slide_no: "03"
    title_en: "RCA Requires Causal Multi-Modal Multi-Hop Reasoning"
    visual_role: "task hardness mechanism"
    layout_pattern: "modality map top + causal path center + reasoning burden row"
    dominant_geometry: "system diagram"
    secondary_geometry: "modality-specific chips"
    focus_object: "causal path tying metrics, logs, and traces to multi-hop reasoning"
    evidence_mode: "conceptual mechanism"
    anti_pattern: "three identical modality cards"
    implementation_surface: "semantic SVG + compact native labels"

  - slide_no: "04"
    title_en: "Current Evaluation Often Rewards Shortcut Correlations"
    visual_role: "problem contrast"
    layout_pattern: "asymmetric shortcut-vs-causal board + mismatch verdict band"
    dominant_geometry: "diagnostic contrast"
    secondary_geometry: "verdict band"
    focus_object: "difference between shallow ranking and causal verification"
    evidence_mode: "contrast reasoning"
    anti_pattern: "symmetric two-column card comparison"
    implementation_surface: "native contrast builder + semantic path icons"

  - slide_no: "05"
    title_en: "Our Evaluation Ladder Uses Three Questions"
    visual_role: "framework contract"
    layout_pattern: "gate ladder left + contract notes right"
    dominant_geometry: "gated ladder"
    secondary_geometry: "notched contract panels"
    focus_object: "A gates B gates C dependency"
    evidence_mode: "method contract"
    anti_pattern: "repeating slide 01 ladder with only text changes"
    implementation_surface: "custom ladder SVG + notched panels"

  - slide_no: "06"
    title_en: "What You Should Remember in 45 Minutes"
    visual_role: "listening guide"
    layout_pattern: "decision rule header + three watch-for signals + verdict footer"
    dominant_geometry: "reset boundary"
    secondary_geometry: "signal strips"
    focus_object: "single decision rule for judging progress claims"
    evidence_mode: "audience orientation"
    anti_pattern: "three equal summary cards"
    implementation_surface: "native typography + band system"

  - slide_no: "07"
    title_en: "If a Simple Heuristic Wins, the Benchmark Is Too Easy"
    visual_role: "falsification setup"
    layout_pattern: "probe claim top + hypothesis/probe/decision chevron flow + fairness rail"
    dominant_geometry: "chevron logic track"
    secondary_geometry: "notched fairness notes"
    focus_object: "SimpleRCA as diagnostic probe rather than proposed method"
    evidence_mode: "falsification design"
    anti_pattern: "casual baseline card page"
    implementation_surface: "native flow blocks + compact notes"

  - slide_no: "08"
    title_en: "SimpleRCA Matches or Beats SOTA on Public Benchmarks"
    visual_role: "first shock evidence"
    layout_pattern: "compact verdict rail + large chart stage"
    dominant_geometry: "evidence stage"
    secondary_geometry: "delta chips"
    focus_object: "benchmark comparison chart with direct implication labels"
    evidence_mode: "quantitative shock"
    anti_pattern: "chart plus detached stack of summary cards"
    implementation_surface: "redrawn chart + integrated annotations"

  - slide_no: "09"
    title_en: "Legacy Benchmarks Are Oversimplified by Construction"
    visual_role: "mechanism diagnosis"
    layout_pattern: "three-scene triptych + causal verdict strip"
    dominant_geometry: "triptych"
    secondary_geometry: "support labels"
    focus_object: "three oversimplification mechanisms"
    evidence_mode: "structural explanation"
    anti_pattern: "three same-shape rounded boxes"
    implementation_surface: "triptych builder + semantic SVG mini-scenes"

  - slide_no: "10"
    title_en: "Realism Needs Impact-Validated Failures, Not Raw Injections"
    visual_role: "principle slide"
    layout_pattern: "injection source -> validation gate -> retain/discard split"
    dominant_geometry: "filter gate"
    secondary_geometry: "branch labels"
    focus_object: "impact validation as the retention criterion"
    evidence_mode: "evaluation rule"
    anti_pattern: "linear pipeline of equal process boxes"
    implementation_surface: "semantic SVG flow"

  - slide_no: "11"
    title_en: "Fault-Propagation-Aware Benchmarking at Scale"
    visual_role: "construction method"
    layout_pattern: "grouped 2x3 stage map with hierarchy braces"
    dominant_geometry: "system diagram"
    secondary_geometry: "brace groups and stage chips"
    focus_object: "six-stage benchmark construction system"
    evidence_mode: "method architecture"
    anti_pattern: "classroom flowchart"
    implementation_surface: "native stage map + semantic arrows"

  - slide_no: "12"
    title_en: "New Benchmark Stats Show a Different Difficulty Regime"
    visual_role: "regime shift evidence"
    layout_pattern: "metric cluster left + regime explanation panel right"
    dominant_geometry: "metric strip cluster"
    secondary_geometry: "support panel"
    focus_object: "difficulty-regime shift rather than isolated KPIs"
    evidence_mode: "quantitative regime summary"
    anti_pattern: "five equal KPI cards"
    implementation_surface: "native metric cluster + compact support graphic"

  - slide_no: "13"
    title_en: "Performance Collapses Under Realistic Conditions"
    visual_role: "dual-shock evidence"
    layout_pattern: "accuracy collapse panel + runtime escalation track + implication band"
    dominant_geometry: "diagnostic contrast"
    secondary_geometry: "evidence band"
    focus_object: "drop in Top@1 together with runtime blow-up"
    evidence_mode: "quantitative collapse"
    anti_pattern: "two tiny charts plus detached explanation card"
    implementation_surface: "redrawn dual evidence panels"

  - slide_no: "14"
    title_en: "Three Failure Modes Explain the Collapse"
    visual_role: "failure taxonomy"
    layout_pattern: "three differentiated failure scenes + common verdict footer"
    dominant_geometry: "triptych"
    secondary_geometry: "failure badges"
    focus_object: "scalability, observability, and modeling bottlenecks"
    evidence_mode: "mechanism synthesis"
    anti_pattern: "uniform three-column cards"
    implementation_surface: "triptych builder + slide-local SVGs"

  - slide_no: "15"
    title_en: "Takeaway A: Benchmark Realism Determines What Progress Means"
    visual_role: "act close"
    layout_pattern: "single claim field + bridge arrow"
    dominant_geometry: "reset boundary"
    secondary_geometry: "bridge band"
    focus_object: "realism as the gate on meaningful progress"
    evidence_mode: "verdict"
    anti_pattern: "another content slide disguised as takeaway"
    implementation_surface: "takeaway builder"

  - slide_no: "16"
    title_en: "Now We Ask the LLM Capability Question"
    visual_role: "section entry"
    layout_pattern: "chapter band + large question + recap chip"
    dominant_geometry: "section banner"
    secondary_geometry: "capsule recap"
    focus_object: "shift from realism to capability"
    evidence_mode: "boundary reset"
    anti_pattern: "thin transition strip only"
    implementation_surface: "section banner builder"

  - slide_no: "17"
    title_en: "OpenRCA Defines RCA as a Goal-Driven Task"
    visual_role: "task contract"
    layout_pattern: "query -> telemetry bundle -> structured output contract"
    dominant_geometry: "protocol map"
    secondary_geometry: "notched output panel"
    focus_object: "goal-driven input-output task definition"
    evidence_mode: "protocol specification"
    anti_pattern: "generic pipeline boxes"
    implementation_surface: "semantic SVG + mono protocol block"

  - slide_no: "18"
    title_en: "OpenRCA Forces Reasoning Over Real Telemetry Scale"
    visual_role: "scale pressure"
    layout_pattern: "large numbers + telemetry mass panel + burden callout"
    dominant_geometry: "massed metric stage"
    secondary_geometry: "telemetry bundle"
    focus_object: "scale and heterogeneity burden"
    evidence_mode: "dataset scale framing"
    anti_pattern: "small evenly weighted KPI cards"
    implementation_surface: "native large-type layout + semantic chips"

  - slide_no: "19"
    title_en: "Current LLMs Struggle on Core RCA Tasks"
    visual_role: "capability evidence"
    layout_pattern: "large chart stage + embedded failure diagnosis"
    dominant_geometry: "evidence stage"
    secondary_geometry: "on-chart callouts"
    focus_object: "measured capability gap on core RCA tasks"
    evidence_mode: "quantitative capability result"
    anti_pattern: "bar chart plus separate red summary card"
    implementation_surface: "redrawn chart with integrated annotations"

  - slide_no: "20"
    title_en: "Execution-Based RCA-Agent Helps but Does Not Close the Gap"
    visual_role: "qualified gain evidence"
    layout_pattern: "gain block -> ceiling block -> residual-gap block"
    dominant_geometry: "three-phase track"
    secondary_geometry: "comparison bars"
    focus_object: "improvement that still leaves a capability ceiling"
    evidence_mode: "qualified improvement"
    anti_pattern: "simple before-after chart with detached explanation"
    implementation_surface: "hybrid chart + track layout"

  - slide_no: "21"
    title_en: "Takeaway B: Better Tasks Expose the True Capability Gap"
    visual_role: "act close"
    layout_pattern: "claim strip + capability gap anchor"
    dominant_geometry: "takeaway band"
    secondary_geometry: "gap marker"
    focus_object: "better tasks reveal real capability limits"
    evidence_mode: "verdict"
    anti_pattern: "copy of slide 15 with recolor only"
    implementation_surface: "takeaway builder variant"

  - slide_no: "22"
    title_en: "Correct Labels Can Still Hide Wrong Reasoning"
    visual_role: "trust problem contrast"
    layout_pattern: "dual-path causal diagram + contradiction band"
    dominant_geometry: "diagnostic contrast"
    secondary_geometry: "causal path labels"
    focus_object: "correct answer versus faithful reasoning path"
    evidence_mode: "conceptual trust failure"
    anti_pattern: "two rounded comparison boxes"
    implementation_surface: "semantic causal-path SVG"

  - slide_no: "23"
    title_en: "Outcome-Only Evaluation Misses Process Failures"
    visual_role: "evaluation-depth contrast"
    layout_pattern: "shallow outcome layer -> deep process audit layer"
    dominant_geometry: "audit funnel / layered board"
    secondary_geometry: "process tags"
    focus_object: "difference between shallow outcome scoring and deeper process audit"
    evidence_mode: "evaluation framework contrast"
    anti_pattern: "pipeline of generic rectangles"
    implementation_surface: "custom layered diagram"

  - slide_no: "24"
    title_en: "FORGE Uses Forward Verification from Known Interventions"
    visual_role: "verification method"
    layout_pattern: "intervention origin + forward-check path + verdict gate"
    dominant_geometry: "verification loop / gated path"
    secondary_geometry: "checkpoint chips"
    focus_object: "forward verification from intervention to causal verdict"
    evidence_mode: "method mechanism"
    anti_pattern: "three-box method summary"
    implementation_surface: "semantic SVG verification flow"

  - slide_no: "25"
    title_en: "OpenRCA 2.0 Adds Step-Wise Causal Supervision"
    visual_role: "supervision mechanism"
    layout_pattern: "answer-only baseline left + checkpoint path right"
    dominant_geometry: "step supervision track"
    secondary_geometry: "comparison panel"
    focus_object: "path-aware supervision over causal steps"
    evidence_mode: "training/data supervision mechanism"
    anti_pattern: "row of equal rounded step tokens"
    implementation_surface: "custom supervision track"

  - slide_no: "26"
    title_en: "Process Metrics Separate Identification from Reasoning"
    visual_role: "metric explanation core slide"
    layout_pattern: "audit ring center + metric decomposition panel right"
    dominant_geometry: "audit ring"
    secondary_geometry: "metric bars / strips"
    focus_object: "visual separation between answer correctness and reasoning faithfulness"
    evidence_mode: "metric semantics"
    anti_pattern: "textual definition slide with tiny diagram"
    implementation_surface: "semantic SVG ring + native decomposition panel"

  - slide_no: "27"
    title_en: "Best-Model Gap Shows Hidden Reasoning Defects"
    visual_role: "trust gap evidence"
    layout_pattern: "answer-vs-reasoning fracture chart + gap verdict"
    dominant_geometry: "gap-led evidence stage"
    secondary_geometry: "fracture marker"
    focus_object: "hidden defect gap between answer performance and reasoning performance"
    evidence_mode: "quantitative trust defect"
    anti_pattern: "bar chart plus side card"
    implementation_surface: "redrawn comparison chart + fracture annotation"

  - slide_no: "28"
    title_en: "The Trust Gap Widens Across 7 LLMs"
    visual_role: "population-level trust evidence"
    layout_pattern: "faceted comparison chart + embedded trend labels"
    dominant_geometry: "faceted evidence stage"
    secondary_geometry: "trend callouts"
    focus_object: "population-wide widening of trust gap"
    evidence_mode: "cross-model quantitative comparison"
    anti_pattern: "tiny grouped bars with detached explanatory paragraph"
    implementation_surface: "redrawn faceted chart"

  - slide_no: "29"
    title_en: "Takeaway C: Trustworthy RCA Needs Causal-Path Faithfulness"
    visual_role: "act close"
    layout_pattern: "verdict field + faithfulness support strip"
    dominant_geometry: "reset boundary"
    secondary_geometry: "support band"
    focus_object: "causal-path faithfulness as the missing layer"
    evidence_mode: "verdict"
    anti_pattern: "copy of slides 15 or 21"
    implementation_surface: "takeaway builder final-act variant"

  - slide_no: "30"
    title_en: "One Research Line: Realism -> Capability -> Trust"
    visual_role: "stitched synthesis"
    layout_pattern: "stitched ladder center + one-line synthesis rail"
    dominant_geometry: "hero synthesis chain"
    secondary_geometry: "capsule result anchors"
    focus_object: "three-question research line stitched into one dependency chain"
    evidence_mode: "synthesis"
    anti_pattern: "repeated motif without consequence"
    implementation_surface: "custom synthesis SVG"

  - slide_no: "31"
    title_en: "Next Agenda: Process-Aware Data, Agents, and Training"
    visual_role: "future roadmap"
    layout_pattern: "roadmap track with three workstream milestones"
    dominant_geometry: "roadmap track"
    secondary_geometry: "milestone nodes"
    focus_object: "future agenda as sequenced workstreams"
    evidence_mode: "forward agenda"
    anti_pattern: "three future-work cards"
    implementation_surface: "roadmap builder"

  - slide_no: "32"
    title_en: "Build Evaluations That Make Reliable AI Operations Possible"
    visual_role: "closing manifesto"
    layout_pattern: "closing statement field + compact ladder strip + action line"
    dominant_geometry: "hero close"
    secondary_geometry: "closing band"
    focus_object: "final build-evaluations message"
    evidence_mode: "final claim"
    anti_pattern: "summary page that rehashes slide 01"
    implementation_surface: "manifesto close builder"
```

## Backup Slide Visual Specs

```yaml
backup_slides:
  - slide_no: "B1"
    title_en: "Why SimpleRCA Wins on Legacy Benchmarks"
    visual_role: "mechanism appendix"
    layout_pattern: "single benchmark case + shortcut path reveal"
    dominant_geometry: "diagnostic contrast"
    secondary_geometry: "case annotation chips"
    focus_object: "concrete shortcut mechanism in one case"
    evidence_mode: "supporting mechanism"
    anti_pattern: "bullet-only appendix page"
    implementation_surface: "semantic SVG case board"

  - slide_no: "B2"
    title_en: "Fault-Propagation-Aware Benchmark Construction Details"
    visual_role: "method appendix"
    layout_pattern: "detailed stage map + criteria side panel"
    dominant_geometry: "system diagram"
    secondary_geometry: "criteria notched panel"
    focus_object: "construction details and filtering criteria"
    evidence_mode: "detailed method"
    anti_pattern: "dense prose or raw table"
    implementation_surface: "expanded stage map"

  - slide_no: "B3"
    title_en: "OpenRCA Task Granularity and Metric Definition"
    visual_role: "protocol appendix"
    layout_pattern: "task taxonomy board + metric-rule panel"
    dominant_geometry: "protocol map"
    secondary_geometry: "notched metric notes"
    focus_object: "task granularity and scoring assumptions"
    evidence_mode: "definition support"
    anti_pattern: "plain definition table"
    implementation_surface: "native structured board"

  - slide_no: "B4"
    title_en: "RCA-Agent Strengths and Failure Patterns"
    visual_role: "agent appendix"
    layout_pattern: "gain mechanism side + residual failure side"
    dominant_geometry: "diagnostic contrast"
    secondary_geometry: "failure tags"
    focus_object: "why the agent helps and where it still breaks"
    evidence_mode: "mechanism + failure support"
    anti_pattern: "bullet narrative only"
    implementation_surface: "hybrid chart + semantic diagrams"

  - slide_no: "B5"
    title_en: "FORGE Verification Pipeline and Metric Math"
    visual_role: "verification appendix"
    layout_pattern: "detailed verification flow + formula panel"
    dominant_geometry: "system diagram"
    secondary_geometry: "formula notched panel"
    focus_object: "verification logic plus metric formulation"
    evidence_mode: "detailed method support"
    anti_pattern: "full-slide equation dump"
    implementation_surface: "verification SVG + compact formula layout"

  - slide_no: "B6"
    title_en: "Threats to Validity Across the Whole Research Line"
    visual_role: "risk appendix"
    layout_pattern: "matrix-lite risk board"
    dominant_geometry: "structured grid"
    secondary_geometry: "impact chips"
    focus_object: "threat, expected impact, and mitigation relationship"
    evidence_mode: "limitations support"
    anti_pattern: "traditional dense table"
    implementation_surface: "custom grid board"
```
