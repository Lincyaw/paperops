# Talk Goal

Deliver a 45-minute research-line talk that unifies three works into one evaluation thesis:
we can only build trustworthy RCA LLM agents if evaluation improves along three axes:
benchmark realism, agent capability under realistic telemetry, and process-level causal faithfulness.

---

# Audience Model

- `audience_type`: academic audience (mixed systems/SE/LLM background)
- `audience_distance`: adjacent-field
- `assumed_knowledge`: knows LLM and software systems basics, not deep RCA benchmark details
- `what they need early`: why RCA is hard, why current metrics can mislead, why process supervision matters
- `language_policy`: English slide titles/text + Chinese speaker notes

---

# Talk Style

- `talk_style`: research-line
- `delivery_mode`: thesis-driven synthesis, not paper-by-paper reporting
- `pace`: 45 min main talk + Q&A with backup slides
- `slide_density_target`: 32 main slides / 45 min (~1.4 min per slide with staged reveals)

---

# Core Thesis

High leaderboard scores in RCA do not guarantee real diagnostic capability or trustworthy reasoning.
Evaluation must progress from realistic failure generation, to realistic LLM tasks, to process-level verification.

---

# Why This Matters

- Modern incidents are cascading and high-cost; wrong diagnosis wastes the recovery window.
- If benchmarks are easy, model design progress is overestimated.
- If labels are outcome-only, correct answers can still come from spurious reasoning.
- Reliable AI for operations needs not only correctness, but causal-path faithfulness.

---

# Story Arc

1. **Question A (Realism)**: Are we evaluating RCA on failures that are truly operationally hard?
2. **Question B (Capability)**: Can current LLM agents diagnose realistic RCA tasks from large heterogeneous telemetry?
3. **Question C (Trustworthiness)**: When models answer correctly, can we verify their causal reasoning path?
4. **Synthesis**: These are not three isolated studies; they are one evaluation ladder for trustworthy RCA agents.

---

# Time/Page Budget

- Act I - Stakes and Evaluation Contract: 6 slides / 8 min
- Act II - Realism Stress Test: 9 slides / 13 min
- Act III - LLM Capability Under Real Telemetry: 7 slides / 10 min
- Act IV - Process-Level Trustworthiness: 7 slides / 10 min
- Act V - Synthesis and Agenda: 3 slides / 4 min
- **Total**: 32 slides / 45 min

---

# Recurring Messages

1. `RCA is a causal, multi-hop, multi-modal reasoning task.`
2. `Benchmark realism determines whether reported progress is meaningful.`
3. `A correct root-cause label is necessary but not sufficient for trust.`

---

# Slide-by-Slide Plan

## Act I - Stakes and Evaluation Contract (Slide 01-06, 8 min)

### Slide 01
- `slide_no`: 01
- `title_en`: Building Trustworthy RCA Evaluation for LLM Agents
- `message`: This talk is one research line about evaluation foundations, not three disconnected papers.
- `role`: setup
- `transition_from_previous`: N/A
- `visual_goal`: One-line thesis + three-step ladder preview
- `keep`: thesis sentence, roadmap graphic
- `cut`: full publication metadata
- `notes_zh`: 开场直接定义主线：真实性、能力、可信性三层评测是同一条研究链。
- `time_sec`: 60

### Slide 02
- `slide_no`: 02
- `title_en`: RCA Failures Are Expensive and Cascading
- `message`: RCA quality directly affects MTTR and incident business impact.
- `role`: setup
- `transition_from_previous`: From talk thesis to operational stakes
- `visual_goal`: Incident impact panel + cascade sketch
- `keep`: downtime cost intuition, cascade propagation
- `cut`: long incident storytelling
- `notes_zh`: 让听众先认可问题价值：不是学术游戏，而是恢复时间和业务损失问题。
- `time_sec`: 75

### Slide 03
- `slide_no`: 03
- `title_en`: RCA Requires Causal Multi-Modal Multi-Hop Reasoning
- `message`: RCA is fundamentally harder than single-modal anomaly ranking.
- `role`: setup
- `transition_from_previous`: From business importance to technical hardness
- `visual_goal`: metrics/logs/traces to causal chain diagram
- `keep`: multi-modal, dependency reasoning, temporal reasoning
- `cut`: tool-specific implementation detail
- `notes_zh`: 解释难点本质：跨模态、跨服务、跨时间的因果推断，不是简单匹配。
- `time_sec`: 75

### Slide 04
- `slide_no`: 04
- `title_en`: Current Evaluation Often Rewards Shortcut Correlations
- `message`: Existing evaluation can over-credit methods that exploit easy symptom patterns.
- `role`: problem
- `transition_from_previous`: From task hardness to evaluation risk
- `visual_goal`: shortcut vs causal reasoning contrast
- `keep`: shortcut concern, measurement mismatch
- `cut`: dataset-specific clutter
- `notes_zh`: 把冲突抛出来：任务很难，但分数看起来很高，说明评测可能出了偏差。
- `time_sec`: 90

### Slide 05
- `slide_no`: 05
- `title_en`: Our Evaluation Ladder Uses Three Questions
- `message`: Question A/B/C form a dependency chain: realism first, then capability, then trustworthiness.
- `role`: transition
- `transition_from_previous`: Convert problem statement into research design
- `visual_goal`: A/B/C question ladder
- `keep`: three questions and dependencies
- `cut`: per-paper chronological list
- `notes_zh`: 这页只做研究问题框架：A/B/C 三问和依赖关系，不提前讲记忆点。
- `time_sec`: 90

### Slide 06
- `slide_no`: 06
- `title_en`: What You Should Remember in 45 Minutes
- `message`: Use one decision rule for the whole talk: reported gains are reliable only when all three layers are satisfied.
- `role`: transition
- `transition_from_previous`: Lock audience expectation before evidence
- `visual_goal`: three takeaways placeholder
- `keep`: three recurring messages
- `cut`: premature result details
- `notes_zh`: 这页只做记忆锚点和判断规则，作为后续所有证据的统一尺子。
- `time_sec`: 90

## Act II - Realism Stress Test (Slide 07-15, 13 min)

### Slide 07
- `slide_no`: 07
- `title_en`: If a Simple Heuristic Wins, the Benchmark Is Too Easy
- `message`: A probing baseline can reveal benchmark oversimplification.
- `role`: method
- `transition_from_previous`: Start Question A with falsifiable probe
- `visual_goal`: probe logic flow
- `keep`: probe rationale
- `cut`: baseline implementation micro-detail
- `notes_zh`: 用“反证法”建立可信度：先不用复杂模型，看简单规则能否打平。
- `time_sec`: 75

### Slide 08
- `slide_no`: 08
- `title_en`: SimpleRCA Matches or Beats SOTA on Public Benchmarks
- `message`: On several public datasets, simple rules approach or exceed SOTA accuracy.
- `role`: evidence
- `transition_from_previous`: Show probe outcome
- `visual_goal`: concise benchmark comparison
- `keep`: key trend and implication
- `cut`: full raw tables
- `notes_zh`: 强调含义而非逐项读数：如果简单方法都够用，评测区分度就不足。
- `time_sec`: 90

### Slide 09
- `slide_no`: 09
- `title_en`: Legacy Benchmarks Are Oversimplified by Construction
- `message`: Fault patterns, graph depth, and telemetry signals often expose root causes too directly.
- `role`: evidence
- `transition_from_previous`: Explain why simple methods work
- `visual_goal`: oversimplification factors triad
- `keep`: fault injection bias, shallow propagation, signal dominance
- `cut`: exhaustive taxonomy
- `notes_zh`: 解释机制：根因症状太显眼、传播太短，导致“相关性捷径”有效。
- `time_sec`: 90

### Slide 10
- `slide_no`: 10
- `title_en`: Realism Needs Impact-Validated Failures, Not Raw Injections
- `message`: Operationally relevant failures must show user-facing SLI degradation.
- `role`: method
- `transition_from_previous`: From diagnosis of old benchmarks to design principle
- `visual_goal`: injection -> validation -> retained cases pipeline
- `keep`: realism principle and retention rule based on SLI impact
- `cut`: internal tooling specifics
- `notes_zh`: 这页只讲原则：impact-validated 才算有效样本。
- `time_sec`: 90

### Slide 11
- `slide_no`: 11
- `title_en`: Fault-Propagation-Aware Benchmarking at Scale
- `message`: A systematic generation framework operationalizes the realism principle at scale.
- `role`: method
- `transition_from_previous`: Show how design principle is operationalized
- `visual_goal`: six-stage generation framework
- `keep`: six-stage construction framework and reproducible scaling
- `cut`: low-level platform config
- `notes_zh`: 这页只讲方法实现：如何把第10页原则系统化落地。
- `time_sec`: 90

### Slide 12
- `slide_no`: 12
- `title_en`: New Benchmark Stats Show a Different Difficulty Regime
- `message`: The new benchmark is larger and structurally harder (1,430 validated cases from 9,152 injections, 25 fault types).
- `role`: evidence
- `transition_from_previous`: Quantify realism change
- `visual_goal`: key numbers card
- `keep`: 1,430 / 9,152 / 25 / 6 / 50 and the implied difficulty-regime shift
- `cut`: decorative statistics
- `notes_zh`: 这页只讲结果态：构建后形成了新的难度分布。
- `time_sec`: 90

### Slide 13
- `slide_no`: 13
- `title_en`: Performance Collapses Under Realistic Conditions
- `message`: Re-evaluating 11 SOTA methods yields low Top@1 (avg 0.21, best 0.37) and much higher runtime.
- `role`: evidence
- `transition_from_previous`: Test model claims on realistic benchmark
- `visual_goal`: old-vs-new performance gap
- `keep`: avg 0.21, best 0.37, seconds->hours trend
- `cut`: method-by-method narration
- `notes_zh`: 这页是第一个“冲击证据”：过去的高分在新基准下显著下滑。
- `time_sec`: 90

### Slide 14
- `slide_no`: 14
- `title_en`: Three Failure Modes Explain the Collapse
- `message`: Scalability limits, observability blind spots, and modeling bottlenecks dominate hard cases.
- `role`: evidence
- `transition_from_previous`: Move from what happened to why
- `visual_goal`: three failure modes with one example each
- `keep`: three failure patterns and practical implication
- `cut`: long case walkthroughs
- `notes_zh`: 从“分数下降”转到“系统性瓶颈”，为后续 LLM 评测打地基。
- `time_sec`: 75

### Slide 15
- `slide_no`: 15
- `title_en`: Takeaway A: Benchmark Realism Determines What Progress Means
- `message`: Without realism, algorithmic progress claims are weak.
- `role`: takeaway
- `transition_from_previous`: Close Question A
- `visual_goal`: one-sentence conclusion + bridge arrow
- `keep`: realism-first claim
- `cut`: additional benchmarks
- `notes_zh`: 结论收口并桥接：先修正真实性合同，后续能力测量才有解释力。
- `time_sec`: 75

## Act III - LLM Capability Under Real Telemetry (Slide 16-22, 10 min)

### Slide 16
- `slide_no`: 16
- `title_en`: Now We Ask the LLM Capability Question
- `message`: Only after fixing the realism contract can Question B validly test whether LLMs can diagnose RCA.
- `role`: transition
- `transition_from_previous`: From realism contract closure to capability measurement
- `visual_goal`: Question B framing
- `keep`: explicit dependency of Question B on Question A
- `cut`: repeated benchmark history
- `notes_zh`: 明确桥接句：先修正真实性合同，再进入能力测量。
- `time_sec`: 75

### Slide 17
- `slide_no`: 17
- `title_en`: OpenRCA Defines RCA as a Goal-Driven Task
- `message`: RCA output is structured around time, component, and reason from natural-language queries.
- `role`: method
- `transition_from_previous`: Introduce task contract
- `visual_goal`: query -> telemetry -> structured answer
- `keep`: goal-driven framing and output elements
- `cut`: prompt engineering detail
- `notes_zh`: 强调任务定义升级：不是只猜组件，而是时间、组件、原因三元目标。
- `time_sec`: 90

### Slide 18
- `slide_no`: 18
- `title_en`: OpenRCA Forces Reasoning Over Real Telemetry Scale
- `message`: 335 failures, three enterprise systems, and 68GB telemetry create realistic long-context heterogeneity.
- `role`: evidence
- `transition_from_previous`: Quantify task difficulty
- `visual_goal`: dataset scale panel
- `keep`: 335 cases, 3 systems, 68GB, logs/metrics/traces
- `cut`: secondary dataset preprocessing details
- `notes_zh`: 让听众感知“真实规模压力”：上下文长、数据异构、噪声高。
- `time_sec`: 90

### Slide 19
- `slide_no`: 19
- `title_en`: Current LLMs Struggle on Core RCA Tasks
- `message`: Reported performance is low under both oracle and sampled telemetry settings.
- `role`: evidence
- `transition_from_previous`: Move from dataset to model outcomes
- `visual_goal`: compact score comparison
- `keep`: 5.37 (oracle) and 3.88 (sampled) for Claude 3.5 baseline settings
- `cut`: full matrix of all model-task combinations
- `notes_zh`: 这里要讲“能力边界”：在真实条件下，现有模型远未达到可用水平。
- `time_sec`: 90

### Slide 20
- `slide_no`: 20
- `title_en`: Execution-Based RCA-Agent Helps but Does Not Close the Gap
- `message`: Tool-augmented reasoning improves best result to 11.34, but the gap to practical reliability remains large.
- `role`: evidence
- `transition_from_previous`: Show value and limit of agentization
- `visual_goal`: baseline vs RCA-agent delta
- `keep`: RCA-agent idea and 11.34 headline
- `cut`: long agent implementation internals
- `notes_zh`: 客观评价 agent：有提升，但不是“问题已解决”，仍是低可用区间。
- `time_sec`: 90

### Slide 21
- `slide_no`: 21
- `title_en`: Takeaway B: Better Tasks Expose the True Capability Gap
- `message`: Realistic benchmarks reveal a substantial diagnostic gap for current LLM agents.
- `role`: takeaway
- `transition_from_previous`: Close Question B
- `visual_goal`: capability gap summary
- `keep`: capability gap statement
- `cut`: repeated numbers
- `notes_zh`: 收束第二问：不是模型没进步，而是离可靠诊断还有明显距离。
- `time_sec`: 90

### Slide 22
- `slide_no`: 22
- `title_en`: Correct Labels Can Still Hide Wrong Reasoning
- `message`: A model can output the correct root-cause label yet still rely on an invalid causal story.
- `role`: transition
- `transition_from_previous`: Open Question C
- `visual_goal`: correct answer vs valid causal path contrast
- `keep`: intuitive tension with one fast contrast example
- `cut`: early metric definitions
- `notes_zh`: 这页偏直觉冲击：先让听众接受“答对不等于推理对”。
- `time_sec`: 75

## Act IV - Process-Level Trustworthiness (Slide 23-29, 10 min)

### Slide 23
- `slide_no`: 23
- `title_en`: Outcome-Only Evaluation Misses Process Failures
- `message`: Formally, outcome-only labels evaluate what answer is given, but not whether the causal derivation process is valid.
- `role`: problem
- `transition_from_previous`: Move from intuitive tension to formal evaluation diagnosis
- `visual_goal`: outcome-only limitation graphic
- `keep`: evaluation contract gap: outcome check without process verification
- `cut`: broad LLM alignment discussion
- `notes_zh`: 这页偏形式化诊断：不是直觉例子，而是评测合同缺口。
- `time_sec`: 75

### Slide 24
- `slide_no`: 24
- `title_en`: FORGE Uses Forward Verification from Known Interventions
- `message`: Given injected causes, FORGE reconstructs causal propagation by checking cause-to-effect consistency.
- `role`: method
- `transition_from_previous`: Introduce solution principle
- `visual_goal`: forward verification pipeline
- `keep`: intervention-informed verification logic
- `cut`: proof-level formalism
- `notes_zh`: 强调 FORGE 的核心优势：把难做的逆向推断变成可操作的正向验证。
- `time_sec`: 90

### Slide 25
- `slide_no`: 25
- `title_en`: OpenRCA 2.0 Adds Step-Wise Causal Supervision
- `message`: Process-level annotations upgrade RCA evaluation beyond outcome labels.
- `role`: method
- `transition_from_previous`: From method to benchmark artifact
- `visual_goal`: outcome label vs step-wise path annotation
- `keep`: 500 evaluated instances with causal paths
- `cut`: data cleaning minutiae
- `notes_zh`: 讲清“新增了什么监督信号”：每一步传播链都可核对。
- `time_sec`: 90

### Slide 26
- `slide_no`: 26
- `title_en`: Process Metrics Separate Identification from Reasoning
- `message`: Use two complementary metrics: Pass@1 asks whether the root cause was identified; PR asks whether that claimed cause can reach symptoms through a valid path.
- `role`: method
- `transition_from_previous`: Define evaluation axes
- `visual_goal`: metric semantics chart
- `keep`: plain-language metric meanings plus inequality PR <= Pass@1
- `cut`: complete metric derivations
- `notes_zh`: 低术语解释：Pass@1 看“找没找对”，PR 看“路径是否成立”。
- `time_sec`: 90

### Slide 27
- `slide_no`: 27
- `title_en`: Best-Model Gap Shows Hidden Reasoning Defects
- `message`: Even the top model drops from Pass@1 0.76 to PR 0.63, exposing unsupported diagnoses.
- `role`: evidence
- `transition_from_previous`: Apply metrics to best case
- `visual_goal`: single-model gap visualization
- `keep`: 0.76 vs 0.63 and one-in-six intuition
- `cut`: unrelated model leaderboard details
- `notes_zh`: 重点句：最高分模型也存在明显过程缺口，不能只看最终命中率。
- `time_sec`: 90

### Slide 28
- `slide_no`: 28
- `title_en`: The Trust Gap Widens Across 7 LLMs
- `message`: Average quality drops from Pass@1 0.52 to PR 0.43, showing systematic process weakness.
- `role`: evidence
- `transition_from_previous`: Generalize from best model to population trend
- `visual_goal`: cross-model Pass@1 vs PR bars
- `keep`: 7 LLMs, avg 0.52 vs 0.43, hallucinated edges notion
- `cut`: per-model anecdotal digressions
- `notes_zh`: 从个例到总体：过程失真是普遍现象，不是某个模型偶然失误。
- `time_sec`: 90

### Slide 29
- `slide_no`: 29
- `title_en`: Takeaway C: Trustworthy RCA Needs Causal-Path Faithfulness
- `message`: Future RCA agents must be evaluated on both outcomes and verifiable propagation processes.
- `role`: takeaway
- `transition_from_previous`: Close Question C
- `visual_goal`: trustworthiness principle card
- `keep`: faithfulness requirement
- `cut`: repeated benchmark descriptions
- `notes_zh`: 第三问收束：可托付诊断必须“答对且推理链可验证”。
- `time_sec`: 75

## Act V - Synthesis and Agenda (Slide 30-32, 4 min)

### Slide 30
- `slide_no`: 30
- `title_en`: One Research Line: Realism -> Capability -> Trust
- `message`: The three studies form one coherent evaluation program for reliable RCA agents.
- `role`: takeaway
- `transition_from_previous`: Integrate A/B/C conclusions
- `visual_goal`: unified ladder with evidence anchors
- `keep`: realism-capability-trust dependency
- `cut`: paper chronology
- `notes_zh`: 再次强调“不是三篇拼盘”，而是同一评测框架的连续推进。
- `time_sec`: 75

### Slide 31
- `slide_no`: 31
- `title_en`: Next Agenda: Process-Aware Data, Agents, and Training
- `message`: Future progress needs harder incidents, better tool-using agents, and process-aware supervision.
- `role`: takeaway
- `transition_from_previous`: Move from synthesis to forward-looking agenda
- `visual_goal`: three future work pillars
- `keep`: concurrent faults, partial observability, process supervision
- `cut`: speculative roadmap without evaluation tie-in
- `notes_zh`: 未来方向要和评测闭环挂钩：数据更真、agent更强、训练更重过程。
- `time_sec`: 90

### Slide 32
- `slide_no`: 32
- `title_en`: Build Evaluations That Make Reliable AI Operations Possible
- `message`: Evaluation quality is the bottleneck and the lever for trustworthy RCA automation.
- `role`: takeaway
- `transition_from_previous`: Final close
- `visual_goal`: closing sentence + Q&A entry
- `keep`: final thesis restatement
- `cut`: new technical content
- `notes_zh`: 结尾句收在“先把评测做对，可靠运维智能体才可能成立”。
- `time_sec`: 90

---

# Backup Slides

### B1 - Why SimpleRCA Wins on Legacy Benchmarks
- `purpose`: Show shortcut correlation mechanism with one concrete case.
- `when_to_use`: If asked whether probe baseline is unfair.

### B2 - Fault-Propagation-Aware Benchmark Construction Details
- `purpose`: Expand six-stage pipeline and impact-driven filtering criteria.
- `when_to_use`: If audience asks about data generation reproducibility.

### B3 - OpenRCA Task Granularity and Metric Definition
- `purpose`: Clarify seven tasks and evaluation assumptions.
- `when_to_use`: If audience questions scoring fairness.

### B4 - RCA-Agent Strengths and Failure Patterns
- `purpose`: Show why execution-based agents improve but still fail.
- `when_to_use`: If audience asks about agent design trade-offs.

### B5 - FORGE Verification Pipeline and Metric Math
- `purpose`: Detail forward verification steps and PR/Edge metrics.
- `when_to_use`: If audience wants formal process-level evaluation details.

### B6 - Threats to Validity Across the Whole Research Line
- `purpose`: Discuss representativeness, oracle assumptions, and external validity.
- `when_to_use`: If audience challenges generalization.

---

# Audience Dropout Risks

1. **Risk**: opening is too operations-specific for adjacent-field listeners.  
   **Mitigation**: use one simple cascade diagram and one cost intuition, then move quickly to evaluation problem.
2. **Risk**: Act II may feel like dataset engineering detail overload.  
   **Mitigation**: keep only numbers that change interpretation (1,430 / 9,152 / 25 / 0.21 / 0.37).
3. **Risk**: listeners may treat low OpenRCA scores as “LLMs are useless” conclusion.  
   **Mitigation**: explicitly separate current gap from long-term potential and show RCA-agent improvement.
4. **Risk**: process metrics in Act IV may feel abstract.  
   **Mitigation**: explain with one sentence: “PR asks whether the claimed root cause reaches symptoms through a valid path.”
5. **Risk**: research-line synthesis could still be perceived as three-paper stitching.  
   **Mitigation**: repeat A/B/C question ladder at transitions and summary.

---

# Open Risks / Missing Inputs

- Exact venue constraints (total slot, mandatory Q&A reservation, formatting rules) are not yet specified.
- Audience composition ratio (systems vs LLM vs software engineering) is unknown; this affects how much method detail to retain.
- Presenter preference on technical depth for FORGE formalism is unspecified.
- Final visual assets (figures to reuse vs redraw) are not selected yet.
