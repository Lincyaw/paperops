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

### Recurring Visual Vocabulary

- Evaluation ladder: `Realism -> Capability -> Trust` (appears in 01, 05, 30, 32)
- RCA system view: telemetry modalities (`metrics`, `logs`, `traces`) + causal path chain (03, 17, 18)
- Benchmark workflow: injection, validation, propagation, annotation (10, 11, 24, 25)
- Evidence cards: key metrics and deltas (`Top@1`, `Pass@1`, `PR`, runtime) (12, 13, 19, 20, 27, 28)
- Takeaway cards: single-sentence claim anchors at act boundaries (06, 15, 21, 29, 30, 32)

### RCA Semantic SVG Language

- `Service`: render as a module card, not a plain rounded rectangle. Prefer a thin identity header, body surface, and small port/interface dots so the node reads as a system component.
- `Anomaly`: render as deviation around a component, not as the component itself. Use pulses, sparkline bursts, halos, or local warning badges.
- `Root cause`: encode with at least two channels beyond color, e.g. fracture notch, epicenter ring, fault badge, or heavier outline.
- `Propagation`: draw as directional causal flow with pulse markers and line-style differences (`solid = verified`, `dashed = candidate`, muted = context).
- `Evidence`: metrics, logs, and traces should have distinct micro-glyphs so the audience can identify modality without reading much text.
- `Context vs focus`: one foreground focal object per slide; surrounding nodes should drop to lower contrast rather than competing equally.

### Visual Hierarchy Rules

- Prefer direct annotations near the relevant node/path instead of detached legends.
- Use color for status and confidence, not object identity alone; pair color with shape, outline, icon, or texture.
- Keep the base object stable across states: normal `Service` -> suspect `Service` -> confirmed `Root cause` should look like the same entity with incremental overlays.
- Charts and diagrams should follow the same logic: gray context, colored focus, explicit takeaway label on the emphasized region.
- Animation should only reveal reasoning order: symptom -> propagation -> verification -> root cause.

### Color Semantics (use theme semantic tokens)

- `primary`: our evaluation framework and trusted conclusions
- `secondary`: baseline or prior benchmark regime
- `accent`: process/method pipelines and transitions
- `positive`: validated improvement or robust evidence
- `negative`: failure modes, capability gaps, trust gaps
- `warning`: risks, caveats, or mismatch states
- `text` / `text_mid`: explanatory text and labels

### Native vs SVG vs Chart vs Imported Figure

- Prefer native SlideCraft (`RoundedBox`, `Callout`, `Flowchart`, `HStack`, `VStack`, `Grid`) for structure-first slides.
- Use `SvgImage` for custom causal/path diagrams, act transitions, and compact schematic flows.
- Use native simple charts (or lightweight custom SVG charts) for small metric sets and headline comparisons.
- Use `Image(path=...)` from redrawn chart assets only when figure complexity exceeds native chart clarity.
- Avoid direct screenshots of paper figures in main deck; if needed for backup, crop tightly and annotate.

### Redraw vs Screenshot Policy

- Redraw when:
  - only 1-3 metrics are needed from a dense paper figure,
  - labels/styles in source are too small for talk settings,
  - structure should match deck vocabulary for continuity.
- Use screenshot only when:
  - exact geometric structure itself is evidence,
  - redraw risks misrepresenting nuanced multi-part relationships,
  - a citation-preserved original is explicitly requested.
- Any reused figure must include source citation and visual emphasis overlays.

## Main Slide Visual Specs

```yaml
main_slides:
  - slide_no: "01"
    title_en: "Building Trustworthy RCA Evaluation for LLM Agents"
    visual_story: "Open with a stronger hero that sells the audience payoff and anchors the full research line."
    layout_pattern: "Hero title + ladder left / payoff panel right + milestone row"
    primary_components: ["TextBlock", "RoundedBox", "Arrow", "Badge", "Callout"]
    animation_groups: ["G1 metadata + hero thesis", "G2 ladder and payoff panel", "G3 milestone row"]
    chart_or_diagram_plan: "Custom ladder diagram with A/B/C plus paper milestones."
    figure_source_strategy: "Redraw natively; no imported figure."
    implementation_notes: "Avoid a sparse title slide; make the opening feel like a confident top-talk opener."

  - slide_no: "02"
    title_en: "RCA Failures Are Expensive and Cascading"
    visual_story: "Connect quantified operational stakes to the mechanics of cascade-driven MTTR pressure."
    layout_pattern: "KPI strip on top + cascade/operations two-column body"
    primary_components: ["Grid", "HStack", "VStack", "Callout", "SvgImage", "BulletList"]
    animation_groups: ["G1 KPI strip", "G2 operator-burden explanation", "G3 cascade diagram + closing badge"]
    chart_or_diagram_plan: "KPI-card strip plus compact symbolic RCA panel; use one symptom chip, one observed-anomaly service, one root-cause service, and one MTTR timer chip."
    figure_source_strategy: "Redraw concept sketch; avoid incident screenshot."
    implementation_notes: "Keep the left diagram small and iconic. No inline long sentences inside the figure; the right column carries explanation."

  - slide_no: "03"
    title_en: "RCA Requires Causal Multi-Modal Multi-Hop Reasoning"
    visual_story: "Make the task hardness concrete by tying modalities to the reasoning burdens they create."
    layout_pattern: "Intro callout + modality cards + propagation chain + burden triad"
    primary_components: ["HStack", "Grid", "SvgImage", "RoundedBox", "Arrow", "TextBlock", "Callout"]
    animation_groups: ["G1 task-hardness framing", "G2 detailed modality cards", "G3 propagation chain", "G4 burden triad"]
    chart_or_diagram_plan: "Diagram-heavy slide with explicit modality-to-burden mapping."
    figure_source_strategy: "Redraw using icon set to keep style continuity."
    implementation_notes: "This slide should feel denser and more technical than slide 02 without turning into a wall of text."

  - slide_no: "04"
    title_en: "Current Evaluation Often Rewards Shortcut Correlations"
    visual_story: "Show that the benchmark problem is not just abstractly flawed, but structurally biased toward shortcut ranking."
    layout_pattern: "Two diagnostic columns + mismatch summary row"
    primary_components: ["HStack", "VStack", "RoundedBox", "Arrow", "Badge", "BulletList"]
    animation_groups: ["G1 shortcut column", "G2 causal column", "G3 benchmark-risk summary"]
    chart_or_diagram_plan: "Conceptual comparison with explicit evidence-use differences."
    figure_source_strategy: "Redraw conceptual contrast; no source figure needed."
    implementation_notes: "Make the comparison operational, not philosophical: what evidence is consumed, what is ignored, why that inflates scores."

  - slide_no: "05"
    title_en: "Our Evaluation Ladder Uses Three Questions"
    visual_story: "Turn the ladder into a methodology contract, not just a repeated motif."
    layout_pattern: "Dependency ladder on left + contract callouts on right + method badge"
    primary_components: ["Flowchart", "RoundedBox", "Arrow", "TextBlock", "Callout", "Badge"]
    animation_groups: ["G1 ladder backbone", "G2 question contracts", "G3 dependency/methodology badge"]
    chart_or_diagram_plan: "Ladder with explicit A gates B gates C semantics."
    figure_source_strategy: "Redraw natively."
    implementation_notes: "Must feel distinct from slide 01; this is the framework slide, not a title echo."

  - slide_no: "06"
    title_en: "What You Should Remember in 45 Minutes"
    visual_story: "Give the audience a professional listening guide, not generic summary cards."
    layout_pattern: "Anchor callout + three watch-for cards + decision rule footer"
    primary_components: ["Grid", "RoundedBox", "TextBlock", "Badge", "Callout"]
    animation_groups: ["G1 anchor callout", "G2 three A/B/C watch-for cards", "G3 decision-rule footer"]
    chart_or_diagram_plan: "No chart; hierarchy should come from emphasis structure."
    figure_source_strategy: "Native cards."
    implementation_notes: "This slide should teach the audience how to read the next evidence pages."

  - slide_no: "07"
    title_en: "If a Simple Heuristic Wins, the Benchmark Is Too Easy"
    visual_story: "Frame the probe as a proper falsification design rather than a casual baseline."
    layout_pattern: "Probe claim + hypothesis/probe/decision flow + fairness row"
    primary_components: ["HStack", "Callout", "Flow", "TextBlock", "Badge"]
    animation_groups: ["G1 probe claim", "G2 hypothesis/probe/decision flow", "G3 why-fair row", "G4 falsification banner"]
    chart_or_diagram_plan: "Process diagram with explicit diagnostic logic."
    figure_source_strategy: "Native redraw."
    implementation_notes: "The slide should make it obvious that SimpleRCA is a probe, not a proposed production method."

  - slide_no: "08"
    title_en: "SimpleRCA Matches or Beats SOTA on Public Benchmarks"
    visual_story: "Deliver the first hard evidence slide with a stronger chart and explicit benchmark diagnosis."
    layout_pattern: "Summary/evidence column left + large chart right + interpretation footer"
    primary_components: ["HStack", "TextBlock", "SvgImage", "Callout", "Badge", "BulletList"]
    animation_groups: ["G1 empirical-shock summary", "G2 redrawn grouped bars", "G3 interpretation footer"]
    chart_or_diagram_plan: "Large redrawn grouped bars plus summary KPIs (3/4 wins, +0.33 max gap, Eadro exception)."
    figure_source_strategy: "Redraw chart from headline values; avoid table screenshot."
    implementation_notes: "This should be the first unmistakably conference-grade evidence slide in the deck."

  - slide_no: "09"
    title_en: "Legacy Benchmarks Are Oversimplified by Construction"
    visual_story: "Explain mechanism behind easy benchmark scores."
    layout_pattern: "Three-column factor panel"
    primary_components: ["Grid", "RoundedBox", "SvgImage", "TextBlock"]
    animation_groups: ["G1 factor 1", "G2 factor 2", "G3 factor 3 + summary"]
    chart_or_diagram_plan: "Triad conceptual diagram (injection bias, shallow propagation, signal dominance)."
    figure_source_strategy: "Native redraw."
    implementation_notes: "Use one icon per factor for quick memory."

  - slide_no: "10"
    title_en: "Realism Needs Impact-Validated Failures, Not Raw Injections"
    visual_story: "Show injection-to-validation pipeline and filtering logic."
    layout_pattern: "Left-to-right pipeline"
    primary_components: ["Flow", "RoundedBox", "Arrow", "Badge", "TextBlock"]
    animation_groups: ["G1 injection stage", "G2 SLI validation stage", "G3 retained-case stage"]
    chart_or_diagram_plan: "Native flow diagram with filter gate."
    figure_source_strategy: "Redraw pipeline."
    implementation_notes: "Add `warning` badge on discarded silent faults."

  - slide_no: "11"
    title_en: "Fault-Propagation-Aware Benchmarking at Scale"
    visual_story: "Expose six-stage benchmark construction framework."
    layout_pattern: "2x3 stage grid with directional arrows"
    primary_components: ["Grid", "RoundedBox", "Arrow", "TextBlock", "Badge"]
    animation_groups: ["G1 first 3 stages", "G2 next 3 stages", "G3 full pipeline overlay"]
    chart_or_diagram_plan: "Framework diagram; no quantitative chart."
    figure_source_strategy: "Redraw from method description; no screenshot."
    implementation_notes: "Keep stage names short; details stay in notes."

  - slide_no: "12"
    title_en: "New Benchmark Stats Show a Different Difficulty Regime"
    visual_story: "Quantify benchmark scale and difficulty shift."
    layout_pattern: "KPI cards + mini support chart"
    primary_components: ["Grid", "RoundedBox", "TextBlock", "SvgImage"]
    animation_groups: ["G1 main KPIs (1430/9152/25)", "G2 supporting dimensions", "G3 takeaway sentence"]
    chart_or_diagram_plan: "KPI-card driven; optional tiny bar for old vs new case scale."
    figure_source_strategy: "Redraw from reported numbers."
    implementation_notes: "Primary visual should be large number cards, not dense axes."

  - slide_no: "13"
    title_en: "Performance Collapses Under Realistic Conditions"
    visual_story: "Show old-vs-new accuracy and runtime gap."
    layout_pattern: "Dual evidence panels (accuracy + runtime)"
    primary_components: ["HStack", "SvgImage", "Callout", "TextBlock"]
    animation_groups: ["G1 Top@1 drop panel", "G2 runtime escalation panel", "G3 implication callout"]
    chart_or_diagram_plan: "Two compact charts: old/new Top@1 and old/new runtime order-of-magnitude."
    figure_source_strategy: "Redraw from headline values; avoid benchmark table screenshot."
    implementation_notes: "Use consistent baseline color mapping across both charts."

  - slide_no: "14"
    title_en: "Three Failure Modes Explain the Collapse"
    visual_story: "Move from outcome drop to causal failure categories."
    layout_pattern: "Three-column failure mode panel"
    primary_components: ["Grid", "RoundedBox", "TextBlock", "Badge", "SvgImage"]
    animation_groups: ["G1 scalability", "G2 observability blind spots", "G3 modeling bottlenecks"]
    chart_or_diagram_plan: "Diagram + one micro-example bullet per column."
    figure_source_strategy: "Native redraw."
    implementation_notes: "Keep each mode to a short symptom + implication pair."

  - slide_no: "15"
    title_en: "Takeaway A: Benchmark Realism Determines What Progress Means"
    visual_story: "Close Question A and bridge to capability evaluation."
    layout_pattern: "Single claim card + bridge arrow"
    primary_components: ["Callout", "Arrow", "TextBlock", "Badge"]
    animation_groups: ["G1 takeaway claim", "G2 bridge to Question B"]
    chart_or_diagram_plan: "No chart; rhetorical close."
    figure_source_strategy: "Native."
    implementation_notes: "Reserve visual whitespace for a clean act boundary."

  - slide_no: "16"
    title_en: "Now We Ask the LLM Capability Question"
    visual_story: "Transition from realism findings to capability testing."
    layout_pattern: "Transition strip slide"
    primary_components: ["TextBlock", "RoundedBox", "Arrow", "Badge"]
    animation_groups: ["G1 recap badge", "G2 Question B focus strip"]
    chart_or_diagram_plan: "No chart."
    figure_source_strategy: "Native."
    implementation_notes: "Use section-divider style distinct from content slides."

  - slide_no: "17"
    title_en: "OpenRCA Defines RCA as a Goal-Driven Task"
    visual_story: "Formalize input-output task contract."
    layout_pattern: "Query -> Telemetry -> Structured Output pipeline"
    primary_components: ["Flow", "RoundedBox", "SvgImage", "TextBlock"]
    animation_groups: ["G1 query prompt", "G2 telemetry analysis block", "G3 output tuple (time/component/reason)"]
    chart_or_diagram_plan: "Pipeline schematic, no numeric chart."
    figure_source_strategy: "Redraw task schema instead of paper figure screenshot."
    implementation_notes: "Use monospaced text style for output tuple formatting."

  - slide_no: "18"
    title_en: "OpenRCA Forces Reasoning Over Real Telemetry Scale"
    visual_story: "Show scale and heterogeneity pressure."
    layout_pattern: "KPI cards + modality composition panel"
    primary_components: ["HStack", "Grid", "RoundedBox", "TextBlock", "Badge"]
    animation_groups: ["G1 335/3/68GB cards", "G2 modality heterogeneity panel", "G3 reasoning burden takeaway"]
    chart_or_diagram_plan: "Card-first design; optional small stacked modality bars."
    figure_source_strategy: "Redraw from benchmark stats."
    implementation_notes: "Avoid crowded charts; keep focus on scale intuition."

  - slide_no: "19"
    title_en: "Current LLMs Struggle on Core RCA Tasks"
    visual_story: "Show low scores under oracle and sampled telemetry settings."
    layout_pattern: "Single compact bar chart + interpretation callout"
    primary_components: ["HStack", "SvgImage", "Callout", "TextBlock"]
    animation_groups: ["G1 two-score chart (5.37, 3.88)", "G2 capability-gap callout"]
    chart_or_diagram_plan: "Two-bar chart with clear labels and delta."
    figure_source_strategy: "Redraw from headline scores."
    implementation_notes: "Large numeric labels are more important than axis detail."

  - slide_no: "20"
    title_en: "Execution-Based RCA-Agent Helps but Does Not Close the Gap"
    visual_story: "Show improvement and remaining distance to reliability."
    layout_pattern: "Before-after comparison with gap annotation"
    primary_components: ["HStack", "SvgImage", "Arrow", "Callout", "TextBlock"]
    animation_groups: ["G1 baseline", "G2 RCA-agent improved score", "G3 residual gap annotation"]
    chart_or_diagram_plan: "Before/after bars with explicit 11.34 marker."
    figure_source_strategy: "Redraw from reported values."
    implementation_notes: "Add visual 'still low' threshold line if space permits."

  - slide_no: "21"
    title_en: "Takeaway B: Better Tasks Expose the True Capability Gap"
    visual_story: "Close capability act with one-sentence conclusion."
    layout_pattern: "Takeaway card slide"
    primary_components: ["Callout", "Badge", "TextBlock"]
    animation_groups: ["G1 takeaway", "G2 bridge to trustworthiness"]
    chart_or_diagram_plan: "No chart."
    figure_source_strategy: "Native."
    implementation_notes: "Echo ladder color for Question B node."

  - slide_no: "22"
    title_en: "Correct Labels Can Still Hide Wrong Reasoning"
    visual_story: "Introduce label-vs-process tension."
    layout_pattern: "Binary contrast card"
    primary_components: ["HStack", "RoundedBox", "Arrow", "TextBlock", "Badge"]
    animation_groups: ["G1 correct label side", "G2 invalid path side", "G3 open Question C"]
    chart_or_diagram_plan: "Conceptual contrast diagram."
    figure_source_strategy: "Native redraw."
    implementation_notes: "Use `positive` for correct label and `warning/negative` for invalid path."

  - slide_no: "23"
    title_en: "Outcome-Only Evaluation Misses Process Failures"
    visual_story: "Formalize the blind spot in current RCA evaluation."
    layout_pattern: "Outcome-only vs process-aware two-column layout"
    primary_components: ["HStack", "VStack", "RoundedBox", "Badge", "TextBlock"]
    animation_groups: ["G1 outcome-only definition", "G2 process gap consequences", "G3 trust implication"]
    chart_or_diagram_plan: "No numeric chart; conceptual framework."
    figure_source_strategy: "Native redraw."
    implementation_notes: "Short labels; avoid metric formulas here."

  - slide_no: "24"
    title_en: "FORGE Uses Forward Verification from Known Interventions"
    visual_story: "Explain cause-to-effect verification workflow."
    layout_pattern: "Pipeline with verification checkpoints"
    primary_components: ["Flowchart", "RoundedBox", "Arrow", "Badge", "TextBlock"]
    animation_groups: ["G1 known intervention", "G2 forward checks", "G3 validated path output"]
    chart_or_diagram_plan: "Native flowchart for FORGE stages."
    figure_source_strategy: "Redraw method pipeline; no screenshot."
    implementation_notes: "Include one concise reason why forward direction is tractable."

  - slide_no: "25"
    title_en: "OpenRCA 2.0 Adds Step-Wise Causal Supervision"
    visual_story: "Show benchmark upgrade from outcome labels to path supervision."
    layout_pattern: "Before/after pipeline comparison"
    primary_components: ["HStack", "Flow", "RoundedBox", "TextBlock", "Badge"]
    animation_groups: ["G1 old labels", "G2 add step-wise annotations", "G3 benchmark upgrade claim"]
    chart_or_diagram_plan: "Diagram-only; side-by-side process depth contrast."
    figure_source_strategy: "Native redraw."
    implementation_notes: "Use consistent iconography with slide 24."

  - slide_no: "26"
    title_en: "Process Metrics Separate Identification from Reasoning"
    visual_story: "Define Pass@1 and PR as complementary metrics."
    layout_pattern: "Two-metric semantics panel"
    primary_components: ["HStack", "RoundedBox", "TextBlock", "Badge", "Arrow"]
    animation_groups: ["G1 Pass@1 meaning", "G2 PR meaning", "G3 PR <= Pass@1 relationship"]
    chart_or_diagram_plan: "No data chart needed; semantic metric panel."
    figure_source_strategy: "Native redraw."
    implementation_notes: "Reserve formulas for backup if needed."

  - slide_no: "27"
    title_en: "Best-Model Gap Shows Hidden Reasoning Defects"
    visual_story: "Quantify trust gap in top model."
    layout_pattern: "Single pair bar chart + implication box"
    primary_components: ["HStack", "SvgImage", "Callout", "TextBlock"]
    animation_groups: ["G1 bars 0.76 vs 0.63", "G2 gap interpretation"]
    chart_or_diagram_plan: "Two-bar comparison chart with delta annotation."
    figure_source_strategy: "Redraw from headline metrics."
    implementation_notes: "Keep bars large; no extra categories."

  - slide_no: "28"
    title_en: "The Trust Gap Widens Across 7 LLMs"
    visual_story: "Generalize gap from best model to population trend."
    layout_pattern: "Grouped bar chart + short takeaway"
    primary_components: ["HStack", "SvgImage", "TextBlock", "Badge"]
    animation_groups: ["G1 grouped bars across models or averages", "G2 average drop callout (0.52 -> 0.43)"]
    chart_or_diagram_plan: "Prefer average-focused bars; include per-model only if readable."
    figure_source_strategy: "Redraw from aggregate values; avoid dense leaderboard screenshot."
    implementation_notes: "Audience-level clarity over full model list completeness."

  - slide_no: "29"
    title_en: "Takeaway C: Trustworthy RCA Needs Causal-Path Faithfulness"
    visual_story: "Close Question C with trust criterion."
    layout_pattern: "Takeaway claim card"
    primary_components: ["Callout", "TextBlock", "Badge"]
    animation_groups: ["G1 trust criterion", "G2 bridge to synthesis"]
    chart_or_diagram_plan: "No chart."
    figure_source_strategy: "Native."
    implementation_notes: "Use same visual motif as slides 15 and 21."

  - slide_no: "30"
    title_en: "One Research Line: Realism -> Capability -> Trust"
    visual_story: "Merge all acts into one coherent program."
    layout_pattern: "Unified ladder summary with evidence anchors"
    primary_components: ["Flowchart", "RoundedBox", "Arrow", "Badge", "TextBlock"]
    animation_groups: ["G1 ladder backbone", "G2 attach key evidence to each node", "G3 final synthesis sentence"]
    chart_or_diagram_plan: "Diagram summary; no new chart."
    figure_source_strategy: "Native redraw."
    implementation_notes: "Reuse colors/icons to trigger memory from earlier slides."

  - slide_no: "31"
    title_en: "Next Agenda: Process-Aware Data, Agents, and Training"
    visual_story: "Show forward roadmap as three coordinated pillars."
    layout_pattern: "Three-pillar roadmap panel"
    primary_components: ["Grid", "RoundedBox", "SvgImage", "TextBlock", "Arrow"]
    animation_groups: ["G1 data pillar", "G2 agent pillar", "G3 training/supervision pillar"]
    chart_or_diagram_plan: "No numeric chart; roadmap diagram with arrows."
    figure_source_strategy: "Native redraw."
    implementation_notes: "One actionable phrase per pillar."

  - slide_no: "32"
    title_en: "Build Evaluations That Make Reliable AI Operations Possible"
    visual_story: "End with core thesis and Q&A handoff."
    layout_pattern: "Closing hero + Q&A marker"
    primary_components: ["TextBlock", "Callout", "Badge"]
    animation_groups: ["G1 final thesis sentence", "G2 Q&A prompt"]
    chart_or_diagram_plan: "No chart."
    figure_source_strategy: "Native."
    implementation_notes: "Keep minimal visual load for final verbal emphasis."
```

## Backup Slide Visual Specs

```yaml
backup_slides:
  - slide_no: "B1"
    title_en: "Why SimpleRCA Wins on Legacy Benchmarks"
    visual_story: "Provide one concrete shortcut-correlation case."
    layout_pattern: "Case-study comparison"
    primary_components: ["HStack", "SvgImage", "TextBlock", "Callout"]
    animation_groups: ["G1 case context", "G2 shortcut path trace", "G3 inflation implication"]
    chart_or_diagram_plan: "One annotated case diagram with minimal numbers."
    figure_source_strategy: "Redraw the case; avoid raw screenshot."
    implementation_notes: "Useful for fairness challenge in Q&A."

  - slide_no: "B2"
    title_en: "Fault-Propagation-Aware Benchmark Construction Details"
    visual_story: "Expand reproducibility details of six-stage pipeline."
    layout_pattern: "Detailed pipeline + threshold panel"
    primary_components: ["Flowchart", "RoundedBox", "Table", "TextBlock"]
    animation_groups: ["G1 full six stages", "G2 validation thresholds", "G3 reproducibility hooks"]
    chart_or_diagram_plan: "Pipeline plus compact table-lite for thresholds."
    figure_source_strategy: "Redraw from method text."
    implementation_notes: "Use small-font table only in backup context."

  - slide_no: "B3"
    title_en: "OpenRCA Task Granularity and Metric Definition"
    visual_story: "Clarify task decomposition and scoring assumptions."
    layout_pattern: "Taxonomy left + metric cards right"
    primary_components: ["HStack", "Table", "RoundedBox", "TextBlock"]
    animation_groups: ["G1 task taxonomy", "G2 scoring rules", "G3 fairness assumptions"]
    chart_or_diagram_plan: "Table-lite plus metric cards."
    figure_source_strategy: "Native redraw."
    implementation_notes: "Keep notation aligned with main slides 17 and 26."

  - slide_no: "B4"
    title_en: "RCA-Agent Strengths and Failure Patterns"
    visual_story: "Detail why tool use helps and where it still breaks."
    layout_pattern: "Workflow diagram + failure examples"
    primary_components: ["HStack", "Flowchart", "Grid", "RoundedBox", "TextBlock"]
    animation_groups: ["G1 agent workflow", "G2 two representative failures", "G3 design trade-off summary"]
    chart_or_diagram_plan: "Mostly diagrammatic; optional tiny comparison bars."
    figure_source_strategy: "Redraw from narrative evidence."
    implementation_notes: "Keep this slide modular for selective reveal during Q&A."

  - slide_no: "B5"
    title_en: "FORGE Verification Pipeline and Metric Math"
    visual_story: "Show full forward-verification steps and metric formula references."
    layout_pattern: "Pipeline + formula side panel"
    primary_components: ["HStack", "Flowchart", "TextBlock", "RoundedBox", "Badge"]
    animation_groups: ["G1 full FORGE pipeline", "G2 metric formulas", "G3 interpretation boundary notes"]
    chart_or_diagram_plan: "Diagram with compact formula text; no heavy chart."
    figure_source_strategy: "Redraw formulas/steps; no screenshot."
    implementation_notes: "If formulas are dense, use appendix-only font size and keep verbal explanation short."

  - slide_no: "B6"
    title_en: "Threats to Validity Across the Whole Research Line"
    visual_story: "Frame limits and mitigations for external validity."
    layout_pattern: "Threat matrix table-lite"
    primary_components: ["Table", "Callout", "TextBlock", "Badge"]
    animation_groups: ["G1 threats list", "G2 expected impact", "G3 mitigation direction"]
    chart_or_diagram_plan: "Table-lite with 3 columns (threat, impact, mitigation)."
    figure_source_strategy: "Native."
    implementation_notes: "Designed for fast response to generalization questions."
```
