# Talk Style Taxonomy

Use this file to choose the talk style before writing content.

## 1. Conference

Best for:
- 8-20 minute paper talks
- venue talks where the audience expects the contribution quickly

Defaults:
- result-first
- minimal background
- strongest 1-3 findings in the main deck
- details, extra baselines, and limitations moved to backup
- pacing target: about 0.8-1.2 slides per minute

Preferred structure:
1. why this matters
2. what gap or problem exists
3. the core idea or approach
4. strongest evidence
5. takeaway and implication

Risks:
- too much setup before the contribution
- too many method details
- trying to preserve paper completeness

## 2. Seminar

Best for:
- group meetings
- internal talks
- invited research seminars with more time

Defaults:
- more audience onboarding
- more intuition, examples, and failure cases
- slower pacing than conference mode
- more room for method reasoning and discussion

Preferred structure:
1. motivation and context
2. problem framing and intuition
3. method or design choices
4. evidence and interpretation
5. open questions and discussion hooks

Risks:
- turning into a lecture
- letting the background crowd out the actual contribution

## 3. Research-Line

Best for:
- multi-paper talks
- thesis-style or agenda-style talks
- invited talks about a coherent research program

Defaults:
- start from one big problem
- unify multiple works under one thesis
- recurring messages must connect the projects
- explicitly avoid serial paper summaries

Preferred structure:
1. big problem and thesis
2. recurring challenges or gaps
3. project A as one step in the thesis
4. project B as the next step
5. project C or future direction
6. synthesis and agenda

Risks:
- "paper 1 / paper 2 / paper 3" fragmentation
- unclear throughline
- too many local details per project

## 4. Job-Talk

Best for:
- faculty or research candidate talks
- broad academic audiences with mixed expertise

Defaults:
- broad and accessible opening
- high clarity about your research identity
- balance expert credibility with non-expert accessibility
- connect completed work to future agenda

Preferred structure:
1. important problem area
2. your research thesis
3. 2-3 representative works
4. synthesis: what unifies them
5. future agenda and fit

Risks:
- too narrow for the room
- too many details before the audience understands your agenda
- no clear sense of future direction

## 5. Tutorial-Lite

Best for:
- education-heavy seminars
- onboarding talks for adjacent audiences
- mini tutorials with a research payload

Defaults:
- explanation-first
- stronger concept buildup
- fewer claims per unit time
- more examples and checkpoints for comprehension

Preferred structure:
1. concepts and why they matter
2. intuitive framing
3. method or system idea
4. worked example or evidence
5. practical takeaway

Risks:
- over-explaining basics
- losing the research novelty

## Audience Distance

Use this to tune density.

### Specialist
- assume terminology and baseline context are known
- move faster to the contribution
- keep deep detail only when it changes interpretation

### Adjacent-Field
- explain assumptions and vocabulary that your field treats as obvious
- use more intuition and framing
- keep examples close to the audience's mental model

### Broad-Academic
- minimize jargon early
- start with stakes, not implementation
- explain why the problem matters before how the method works

## Recommended Fallbacks

If the user has not chosen a style:
- default to `conference` for single-paper talks
- default to `research-line` for multi-paper talks
- default to `seminar` for internal group presentations

But still ask before writing the content if style is not explicit.
