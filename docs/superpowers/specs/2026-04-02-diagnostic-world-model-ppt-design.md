# Diagnostic World Model Deck Design

## Context

Build a 20-slide, 20-minute English presentation in SlideCraft for a research talk on diagnostic intelligence and world models.

The deck must follow a "vision first" narrative:

1. Vision: diagnostic intelligence should move from pattern matching to causal reasoning
2. Feasibility: the domain has atomicity, simulability, and verifiability
3. Achieved path: the PhD work already moves the field toward this goal
4. Next stage: train an agent, distill a world model, generalize into a universal expert
5. Closing: research philosophy and takeaways

The source outline is already fixed by the user. The implementation task is to convert that outline into a polished native PowerPoint deck using `paperops.slides`.

Reference materials for content grounding are located at:

- `/home/ddq/AoyangSpace/paperops/examples/self-intro/references`

## Audience And Goal

- Audience: research faculty / technical interview committee / PhD-level systems and ML audience
- Goal: make the audience believe the speaker has a coherent long-term research vision, has already delivered foundational contributions, and has a credible next-step agenda
- Tone: rigorous, ambitious, controlled, and explanatory rather than flashy

## Format Constraints

- Language: all English
- Duration: 20 minutes
- Slides: 20
- Medium: native PowerPoint shapes/components using SlideCraft, not image-only slides
- Editing target: a script in `examples/` that generates a `.pptx`

## Recommended Approach

Use a fully native SlideCraft deck with diagrams, stat blocks, structured comparisons, and limited on-slide text.

Why this approach:

- The argument is causal and sequential, so diagrams and staged reveals communicate better than dense prose
- Native components make revisions easier than rasterized visuals
- The audience needs both ambition and evidence, so the deck should sit between keynote minimalism and paper-style density

## Visual Direction

### Overall Style

- Presentation-first, not paper-first
- Minimal text, strong structural diagrams
- One dominant idea per slide
- Motion used to pace reasoning, not decorate

### Theme

Use a restrained professional-academic theme derived from `themes.professional`:

- Primary: deep blue for core concepts and structure
- Accent: warm orange for tension, failure, or benchmark shock
- Positive/highlight: teal or blue-green for validated causal reasoning and future roadmap
- Support colors: muted grays for background and secondary elements
- Background: light neutral for most slides, dark high-contrast opener on slide 1

### Typography

- All English
- Short phrases instead of full sentences
- Titles left aligned except the opener
- One emphasis phrase per slide
- No paragraph blocks

### Motion

- Most slides use 2-step or 3-step reveal
- Preferred sequence: claim -> evidence/structure -> interpretation
- Section divider style is not needed because the deck already has enough structural momentum

## Narrative Structure

### Part 1: Vision

Slides 1-2 establish the ambition and the contrast between current AI and a world-model-based diagnostic agent.

- Slide 1 should feel bold and conceptual
- Slide 2 should visually contrast black-box answering with causal diagnosis

### Part 2: Feasibility

Slides 3-5 answer "why this is possible now."

- Slide 3: finite atomic fault space
- Slide 4: fault injection makes the domain simulatable
- Slide 5: verifiable ground truth closes the training loop

These slides should make the audience accept the domain as learnable rather than open-ended.

### Part 3: Achieved Path

Slides 6-15 show the speaker's concrete research progression.

Sub-arc 1, problem redefinition:
- Slide 6: old RCA taxonomy vs goal-driven taxonomy
- Slide 7: effectiveness-data-cost triangle
- Slide 8: formalization from observations to propagation graph

Sub-arc 2, benchmark validation:
- Slide 9: benchmark crisis, simple baseline beats SOTA
- Slide 10: why existing datasets fail
- Slide 11: new benchmark and performance collapse under realistic evaluation

Sub-arc 3, process supervision:
- Slide 12: answer correctness vs reasoning correctness gap
- Slide 13: FORGE converts backward diagnosis into forward verification
- Slide 14: OpenRCA 2.0 as a process-level benchmark
- Slide 15: Triangle multi-agent triage system

### Part 4: From Agent To World Model

Slides 16-18 are roadmap slides.

- Slide 16: train the agent in a simulated loop
- Slide 17: distill world model from reasoning traces
- Slide 18: reuse world model across tasks

These slides should feel more synthetic and future-oriented than the middle evidence block.

### Part 5: Closing

Slides 19-20 should compress the whole talk into identity and memory.

- Slide 19: three durable research capabilities
- Slide 20: three takeaways and close

## Slide-Level Design Plan

### Slide 1: Title

- Structure: centered title composition on dark background
- Core elements: main title, subtitle, presenter name/affiliation, subtle causal-path motif
- Animation: optional 2-step reveal for title then subtitle

### Slide 2: The Vision

- Structure: side-by-side contrast
- Left: current AI black-box path
- Right: world-model diagnostic loop
- Key line: explainable diagnosis comes from causal inference plus verification
- Animation: left side first, right side second

### Slide 3: Premise 1: Atomic Faults

- Structure: atom library grid plus combination message
- Key visual: categories of atomic failures with a prominent "<100"-style boundedness cue
- Animation: finite atoms first, combinatorial complexity second

### Slide 4: Premise 2: Simulatable

- Structure: fault injection pipeline to case generation
- Key visual: injection -> workload -> observability -> cases
- Animation: mechanism first, data-generation consequence second

### Slide 5: Premise 3: Verifiable

- Structure: verification loop around ground truth
- Key visual: hypothesis -> verify -> reward/fail -> update
- Animation: ground truth anchor first, training loop second

### Slide 6: The Problem with RCA

- Structure: old taxonomy vs goal-driven taxonomy
- Key visual: fragmented modality buckets on the left, organized goal system on the right
- Animation: confusion first, reframing second

### Slide 7: Effectiveness-Data-Cost Triangle

- Structure: three-corner trade-off diagram
- Key visual: triangle with unavoidable trade-off in the center
- Animation: corners first, central interpretation second

### Slide 8: Ideal RCA Formalization

- Structure: large formal definition plus graph-producing process diagram
- Key visual: `F: O -> G`
- Emphasis: output is a propagation graph, not a point estimate
- Animation: formula first, graph mapping second

### Slide 9: The Benchmark Crisis

- Structure: compact comparison emphasizing simple baseline beating SOTA
- Key visual: result contrast using stat blocks or a minimalist chart
- Animation: surprising result first, implication second

### Slide 10: Root Cause Analysis

- Structure: three defect cards
- Defects: too few fault cases, oversimplified request structure, narrow fault diversity
- Animation: defects appear in sequence, then synthesis

### Slide 11: Our Solution

- Structure: left-side benchmark construction pipeline, right-side re-evaluation drop
- Key visual: `9,152 injections -> 1,430 cases -> 25 fault types` and `0.9+ -> 0.21`
- Animation: benchmark scale first, score collapse second

### Slide 12: The Process Gap

- Structure: dual-track comparison between answer correctness and path correctness
- Key visual: 76% vs 63%
- Emphasis: correct answers can still come from invalid reasoning
- Animation: metric contrast first, process-supervision diagnosis second

### Slide 13: FORGE: Forward Verification

- Structure: transformation diagram
- Key visual: backward ill-posed diagnosis rewritten into forward verification through intervention
- Animation: two problem formulations first, conversion mechanism second

### Slide 14: OpenRCA 2.0

- Structure: process-level benchmark split between benchmark spec and evaluation dimensions
- Key visual: process supervision rather than answer-only evaluation
- Animation: benchmark identity first, dimensions second

### Slide 15: Triangle: Multi-Agent Triage

- Structure: triangular collaborative system
- Key visual: Analyzer, Decider, Team Manager
- Animation: agents appear individually, collaboration loop closes last

### Slide 16: Stage 1: Train the Agent

- Structure: simulation-to-reward loop
- Key visual: simulated incidents -> hypothesis -> verify -> reward
- Animation: task loop first, learning objective second

### Slide 17: Stage 2: Distill World Model

- Structure: many traces compressed into one causal model
- Key visual: trace density on left, clean graph on right
- Animation: traces first, distilled model second

### Slide 18: Stage 3: Universal Expert

- Structure: one central model branching to downstream tasks
- Downstream tasks: root cause localization, fault prediction, repair suggestion
- Animation: central model first, task fan-out second

### Slide 19: My Research Philosophy

- Structure: three-column pillar slide
- Pillars: complex system modeling, rigorous validation, intelligent system design
- Animation: pillars first, synthesis statement second

### Slide 20: Three Takeaways

- Structure: three strong numbered takeaway blocks plus closing line
- Emphasis: memorable closure rather than recap overload
- Animation: takeaway 1 first, then 2 and 3 plus final line

## Component Strategy

Primary SlideCraft components to use:

- `Presentation`
- `TextBlock`
- `Callout`
- `HStack`, `VStack`, `Grid`, `Padding`
- `RoundedBox`, `Box`, `Circle`, `Badge`
- `Arrow`, `Line`
- `Flow`, `Flowchart`
- `BarChart` only where a quantitative contrast is stronger than prose

Avoid:

- Long `BulletList` slides
- Repeating the same boxy layout for multiple slides in a row
- Heavy tables except where they materially outperform a diagram

## Text Strategy

- Keep visible text to headline fragments and short labels
- Put fuller phrasing into speaker notes
- Preserve all major quantitative claims from the user's outline
- Make wording sound oral and presentation-native, not copied from a paper abstract

## Output Plan

Create:

1. A Python build script under `examples/`
2. A generated `.pptx`
3. Optional preview PNGs for selected slides if needed for QA

## Source Material Plan

Use the local materials in `/home/ddq/AoyangSpace/paperops/examples/self-intro/references` as the primary factual source for wording, terminology, and numeric claims.

Expected usage:

- Survey/reference material for the RCA framing slides
- Dataset/benchmark reference material for slides on the benchmark crisis and the new benchmark
- FORGE / process-supervision material for slides on forward verification and OpenRCA 2.0
- TRIANGLE material for the multi-agent triage slide

Implementation should avoid inventing new technical claims beyond what can be supported by those reference files and the user's outline.

## Verification Plan

Before completion:

1. Run the build script with `uv run python`
2. Save the generated presentation
3. Run presentation checks if available
4. Preview representative slides if rendering support works in the environment

## Scope Boundaries

In scope:

- One complete 20-slide English deck
- Native SlideCraft implementation
- Speaker-note-ready structure

Out of scope:

- Rewriting the user's research claims
- Adding external citations beyond what the outline already supports
- Building custom library features unless implementation uncovers a hard blocker

## Open Assumptions

- The speaker name and affiliation can be left as editable placeholders if not provided
- The deck should optimize for technical talks rather than investor-style storytelling
- Existing SlideCraft components are sufficient for all required visuals
