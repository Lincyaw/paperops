# Slide Spec

## Header

```yaml
talk_style: research-line
audience_type: academic audience
audience_distance: adjacent-field
duration_min: 45
language_policy: English slide titles/text + Chinese speaker notes
main_slide_count: 32
backup_slide_count: 6
source_docs:
  - FSE_26_RCA_dataset_study.pdf
  - 13411_OpenRCA_Can_Large_Langua.pdf
  - ICML_Bench.pdf
```

## Main Slides

```yaml
main_slides:
  - slide_no: "01"
    title_en: "Building Trustworthy RCA Evaluation for LLM Agents"
    message: "This talk is one research line about evaluation foundations, not three disconnected papers."
    visual_type: "title"
    build_order: ["State one-line thesis and audience payoff", "Reveal Realism->Capability->Trust ladder", "Show three milestone cards and A/B/C framing"]
    asset_or_figure_needs: ["Hero ladder diagram", "Research-line metadata badges", "Three milestone cards for realism/capability/trust"]
    notes_zh: "开场直接定义主线：真实性、能力、可信性三层评测是同一条研究链。"

  - slide_no: "02"
    title_en: "RCA Failures Are Expensive and Cascading"
    message: "RCA quality directly affects MTTR and incident business impact."
    visual_type: "claim"
    build_order: ["Lead with quantified operational stakes", "Show cascade diagram and MTTR pressure", "Explain why RCA is the recovery bottleneck"]
    asset_or_figure_needs: ["KPI strip with downtime/on-call/incident-cost signals", "Microservice cascade diagram", "MTTR explanation callouts"]
    notes_zh: "让听众先认可问题价值：不是学术游戏，而是恢复时间和业务损失问题。"

  - slide_no: "03"
    title_en: "RCA Requires Causal Multi-Modal Multi-Hop Reasoning"
    message: "RCA is fundamentally harder than single-modal anomaly ranking."
    visual_type: "pipeline"
    build_order: ["Tie each telemetry modality to a distinct inference burden", "Show a realistic propagation chain", "Summarize the three reasoning burdens"]
    asset_or_figure_needs: ["Detailed metrics/logs/traces cards", "Causal propagation chain", "Three burden cards: fuse signals / trace hops / order events"]
    notes_zh: "解释难点本质：跨模态、跨服务、跨时间的因果推断，不是简单匹配。"

  - slide_no: "04"
    title_en: "Current Evaluation Often Rewards Shortcut Correlations"
    message: "Existing evaluation can over-credit methods that exploit easy symptom patterns."
    visual_type: "comparison"
    build_order: ["Contrast shortcut ranking with causal verification", "Show what evidence each path uses", "State why high scores can still mislead"]
    asset_or_figure_needs: ["Two-column reasoning comparison", "Evidence-use bullets", "Mismatch summary strip"]
    notes_zh: "把冲突抛出来：任务很难，但分数看起来很高，说明评测可能出了偏差。"

  - slide_no: "05"
    title_en: "Our Evaluation Ladder Uses Three Questions"
    message: "We evaluate RCA progress with realism, capability, and trustworthiness in sequence."
    visual_type: "pipeline"
    build_order: ["Show the three-question ladder", "Make the A->B->C dependency explicit", "Translate each question into an evaluation contract"]
    asset_or_figure_needs: ["Dependency ladder graphic", "Three contract callouts for A/B/C", "Methodology contract badge"]
    notes_zh: "明确方法论：先确认题目是否真实，再测能力，最后测推理是否可信。"

  - slide_no: "06"
    title_en: "What You Should Remember in 45 Minutes"
    message: "Reported gains are reliable only when all three layers are satisfied."
    visual_type: "takeaway"
    build_order: ["State the three anchors for the audience", "Show what evidence to watch for under A/B/C", "Lock the decision rule for the rest of the talk"]
    asset_or_figure_needs: ["Anchor callout", "Three watch-for cards", "Decision-rule footer"]
    notes_zh: "先给听众“记忆钩子”，后面所有证据都回扣这三句话。"

  - slide_no: "07"
    title_en: "If a Simple Heuristic Wins, the Benchmark Is Too Easy"
    message: "A probing baseline can reveal benchmark oversimplification."
    visual_type: "claim"
    build_order: ["Define the probe as a diagnostic instrument", "State the falsification logic", "Explain why this probe is fair before showing results"]
    asset_or_figure_needs: ["Hypothesis-probe-decision flow", "Why-fair callout", "Falsifiable criterion banner"]
    notes_zh: "用“反证法”建立可信度：先不用复杂模型，看简单规则能否打平。"

  - slide_no: "08"
    title_en: "SimpleRCA Matches or Beats SOTA on Public Benchmarks"
    message: "On several public datasets, simple rules approach or exceed SOTA accuracy."
    visual_type: "chart"
    build_order: ["Lead with the first empirical shock", "Show redrawn benchmark comparison chart", "Summarize the win pattern and its benchmark implication"]
    asset_or_figure_needs: ["Redrawn grouped bar chart for SimpleRCA vs SOTA", "Summary KPI cards: 3/4 wins, +0.33 max gap, Eadro as exception", "Interpretation badge"]
    notes_zh: "强调含义而非逐项读数：如果简单方法都够用，评测区分度就不足。"

  - slide_no: "09"
    title_en: "Legacy Benchmarks Are Oversimplified by Construction"
    message: "Fault patterns, graph depth, and telemetry signals often expose root causes too directly."
    visual_type: "comparison"
    build_order: ["List oversimplification factors", "Map each factor to shortcut effect", "Bridge to benchmark redesign"]
    asset_or_figure_needs: ["Three-factor panel: injection/graph/signal"]
    notes_zh: "解释机制：根因症状太显眼、传播太短，导致“相关性捷径”有效。"

  - slide_no: "10"
    title_en: "Realism Needs Impact-Validated Failures, Not Raw Injections"
    message: "Operationally relevant failures must show user-facing SLI degradation."
    visual_type: "pipeline"
    build_order: ["Inject faults", "Validate with SLI impact", "Retain operationally relevant cases only"]
    asset_or_figure_needs: ["Injection->validation->retention pipeline"]
    notes_zh: "点出关键原则：并非注入就算故障，必须对用户指标产生影响才有评测价值。"

  - slide_no: "11"
    title_en: "Fault-Propagation-Aware Benchmarking at Scale"
    message: "Systematic generation creates more realistic and diverse RCA difficulty."
    visual_type: "pipeline"
    build_order: ["Show six-stage framework", "Emphasize dynamic workload + fault diversity", "Emphasize hierarchical labels"]
    asset_or_figure_needs: ["Six-stage framework graphic"]
    notes_zh: "这页强调“可复现的大规模构建能力”，不是一次性手工造数据。"

  - slide_no: "12"
    title_en: "New Benchmark Stats Show a Different Difficulty Regime"
    message: "The new benchmark is larger and structurally harder (1,430 validated cases from 9,152 injections, 25 fault types)."
    visual_type: "chart"
    build_order: ["Present key counts", "Map counts to difficulty shift", "Summarize why this changes evaluation meaning"]
    asset_or_figure_needs: ["KPI cards for 1430/9152/25/6/50"]
    notes_zh: "只保留改变结论的数字，让听众看到“难度分布已经变了”。"

  - slide_no: "13"
    title_en: "Performance Collapses Under Realistic Conditions"
    message: "Re-evaluating 11 SOTA methods yields low Top@1 (avg 0.21, best 0.37) and much higher runtime."
    visual_type: "chart"
    build_order: ["Show old vs new performance", "Highlight avg/best Top@1 drop", "Highlight seconds-to-hours runtime jump"]
    asset_or_figure_needs: ["Old-vs-new Top@1 and runtime comparison chart"]
    notes_zh: "这页是第一个“冲击证据”：过去的高分在新基准下显著下滑。"

  - slide_no: "14"
    title_en: "Three Failure Modes Explain the Collapse"
    message: "Scalability limits, observability blind spots, and modeling bottlenecks dominate hard cases."
    visual_type: "comparison"
    build_order: ["Introduce failure mode triad", "Give one representative symptom each", "Tie to research bottlenecks"]
    asset_or_figure_needs: ["Three-column failure mode panel"]
    notes_zh: "从“分数下降”转到“系统性瓶颈”，为后续 LLM 评测打地基。"

  - slide_no: "15"
    title_en: "Takeaway A: Benchmark Realism Determines What Progress Means"
    message: "Without realism, algorithmic progress claims are weak."
    visual_type: "takeaway"
    build_order: ["State realism-first conclusion", "Pin to Question A", "Bridge to capability question"]
    asset_or_figure_needs: ["Conclusion card + bridge arrow"]
    notes_zh: "结论收口：先把题目做对，后续谈模型能力才有意义。"

  - slide_no: "16"
    title_en: "Now We Ask the LLM Capability Question"
    message: "With a realistic task definition, we can test whether LLMs can actually diagnose RCA."
    visual_type: "transition"
    build_order: ["Recap realism outcome", "Introduce Question B", "State evaluation contract for capability"]
    asset_or_figure_needs: ["Act transition strip with Question B"]
    notes_zh: "过渡语义：不是“LLM 万能论”，而是在真实任务上做能力测量。"

  - slide_no: "17"
    title_en: "OpenRCA Defines RCA as a Goal-Driven Task"
    message: "RCA output is structured around time, component, and reason from natural-language queries."
    visual_type: "pipeline"
    build_order: ["Show query input", "Show telemetry analysis step", "Show structured output (time/component/reason)"]
    asset_or_figure_needs: ["Query->telemetry->answer schema diagram"]
    notes_zh: "强调任务定义升级：不是只猜组件，而是时间、组件、原因三元目标。"

  - slide_no: "18"
    title_en: "OpenRCA Forces Reasoning Over Real Telemetry Scale"
    message: "335 failures, three enterprise systems, and 68GB telemetry create realistic long-context heterogeneity."
    visual_type: "chart"
    build_order: ["Present dataset scale", "Explain heterogeneity challenge", "Connect scale to reasoning burden"]
    asset_or_figure_needs: ["Scale panel for 335/3/68GB and modality mix"]
    notes_zh: "让听众感知“真实规模压力”：上下文长、数据异构、噪声高。"

  - slide_no: "19"
    title_en: "Current LLMs Struggle on Core RCA Tasks"
    message: "Reported performance is low under both oracle and sampled telemetry settings."
    visual_type: "chart"
    build_order: ["Show oracle setting score", "Show sampled setting score", "Interpret as capability gap"]
    asset_or_figure_needs: ["Score chart emphasizing 5.37 and 3.88"]
    notes_zh: "这里要讲“能力边界”：在真实条件下，现有模型远未达到可用水平。"

  - slide_no: "20"
    title_en: "Execution-Based RCA-Agent Helps but Does Not Close the Gap"
    message: "Tool-augmented reasoning improves best result to 11.34, but the gap to practical reliability remains large."
    visual_type: "comparison"
    build_order: ["Introduce execution-based idea", "Show gain to 11.34", "State remaining reliability gap"]
    asset_or_figure_needs: ["Before/after chart for base vs RCA-agent"]
    notes_zh: "客观评价 agent：有提升，但不是“问题已解决”，仍是低可用区间。"

  - slide_no: "21"
    title_en: "Takeaway B: Better Tasks Expose the True Capability Gap"
    message: "Realistic benchmarks reveal a substantial diagnostic gap for current LLM agents."
    visual_type: "takeaway"
    build_order: ["State capability takeaway", "Tie back to Question B", "Bridge to trustworthiness question"]
    asset_or_figure_needs: ["Capability gap summary card"]
    notes_zh: "收束第二问：不是模型没进步，而是离可靠诊断还有明显距离。"

  - slide_no: "22"
    title_en: "Correct Labels Can Still Hide Wrong Reasoning"
    message: "Outcome accuracy alone cannot guarantee trustworthy diagnosis."
    visual_type: "transition"
    build_order: ["State label-vs-reasoning tension", "Show quick contrast example", "Open Question C"]
    asset_or_figure_needs: ["Contrast graphic: correct label vs invalid path"]
    notes_zh: "关键转折页：引出第三问，为什么“答对”不等于“可托付”。"

  - slide_no: "23"
    title_en: "Outcome-Only Evaluation Misses Process Failures"
    message: "Existing labels usually check what the answer is, not how the answer was derived."
    visual_type: "comparison"
    build_order: ["Define outcome-only check", "Expose process blind spot", "State consequence for trust"]
    asset_or_figure_needs: ["Outcome-only limitation diagram"]
    notes_zh: "把问题说清：缺的是过程可验证性，不是再多一个最终标签。"

  - slide_no: "24"
    title_en: "FORGE Uses Forward Verification from Known Interventions"
    message: "Given injected causes, FORGE reconstructs causal propagation by checking cause-to-effect consistency."
    visual_type: "pipeline"
    build_order: ["Start from known intervention", "Run forward verification", "Recover validated causal path"]
    asset_or_figure_needs: ["FORGE forward verification pipeline"]
    notes_zh: "强调 FORGE 的核心优势：把难做的逆向推断变成可操作的正向验证。"

  - slide_no: "25"
    title_en: "OpenRCA 2.0 Adds Step-Wise Causal Supervision"
    message: "Process-level annotations upgrade RCA evaluation beyond outcome labels."
    visual_type: "pipeline"
    build_order: ["Show old outcome-only labels", "Add step-wise path annotations", "State benchmark upgrade"]
    asset_or_figure_needs: ["Outcome-only vs step-wise supervision diagram"]
    notes_zh: "讲清“新增了什么监督信号”：每一步传播链都可核对。"

  - slide_no: "26"
    title_en: "Process Metrics Separate Identification from Reasoning"
    message: "Pass@1 and Path Reachability measure different properties and should be reported together."
    visual_type: "comparison"
    build_order: ["Define Pass@1", "Define Path Reachability", "Show why dual reporting is required"]
    asset_or_figure_needs: ["Metric semantics panel with PR <= Pass@1"]
    notes_zh: "把指标解释成“会不会找对”和“能不能讲对因果链”两层能力。"

  - slide_no: "27"
    title_en: "Best-Model Gap Shows Hidden Reasoning Defects"
    message: "Even the top model drops from Pass@1 0.76 to PR 0.63, exposing unsupported diagnoses."
    visual_type: "chart"
    build_order: ["Plot Pass@1 vs PR for best model", "Quantify gap", "Interpret trust implication"]
    asset_or_figure_needs: ["Two-bar chart for 0.76 vs 0.63"]
    notes_zh: "重点句：最高分模型也存在明显过程缺口，不能只看最终命中率。"

  - slide_no: "28"
    title_en: "The Trust Gap Widens Across 7 LLMs"
    message: "Average quality drops from Pass@1 0.52 to PR 0.43, showing systematic process weakness."
    visual_type: "chart"
    build_order: ["Show cross-model averages", "Highlight drop from 0.52 to 0.43", "Conclude systematic weakness"]
    asset_or_figure_needs: ["Cross-model grouped bars for Pass@1 and PR"]
    notes_zh: "从个例到总体：过程失真是普遍现象，不是某个模型偶然失误。"

  - slide_no: "29"
    title_en: "Takeaway C: Trustworthy RCA Needs Causal-Path Faithfulness"
    message: "Future RCA agents must be evaluated on both outcomes and verifiable propagation processes."
    visual_type: "takeaway"
    build_order: ["State trustworthiness conclusion", "Tie to Question C closure", "Bridge to full-line synthesis"]
    asset_or_figure_needs: ["Trust principle card"]
    notes_zh: "第三问收束：可托付诊断必须“答对且推理链可验证”。"

  - slide_no: "30"
    title_en: "One Research Line: Realism -> Capability -> Trust"
    message: "The three studies form one coherent evaluation program for reliable RCA agents."
    visual_type: "takeaway"
    build_order: ["Merge A/B/C outcomes", "Show dependency arrows", "Restate unified thesis"]
    asset_or_figure_needs: ["Unified ladder summary graphic"]
    notes_zh: "再次强调“不是三篇拼盘”，而是同一评测框架的连续推进。"

  - slide_no: "31"
    title_en: "Next Agenda: Process-Aware Data, Agents, and Training"
    message: "Future progress needs harder incidents, better tool-using agents, and process-aware supervision."
    visual_type: "claim"
    build_order: ["List data agenda", "List agent agenda", "List training/supervision agenda"]
    asset_or_figure_needs: ["Three-pillar future agenda panel"]
    notes_zh: "未来方向要和评测闭环挂钩：数据更真、agent更强、训练更重过程。"

  - slide_no: "32"
    title_en: "Build Evaluations That Make Reliable AI Operations Possible"
    message: "Evaluation quality is the bottleneck and the lever for trustworthy RCA automation."
    visual_type: "takeaway"
    build_order: ["Restate final thesis", "Link to practical reliability", "Open Q&A"]
    asset_or_figure_needs: ["Closing statement layout + Q&A marker"]
    notes_zh: "结尾句收在“先把评测做对，可靠运维智能体才可能成立”。"
```

## Backup Slides

```yaml
backup_slides:
  - slide_no: "B1"
    title_en: "Why SimpleRCA Wins on Legacy Benchmarks"
    message: "Show the shortcut-correlation mechanism with one concrete case."
    visual_type: "backup"
    build_order: ["Present one legacy case", "Trace shortcut path", "State why this inflates scores"]
    asset_or_figure_needs: ["One representative legacy benchmark case graphic"]
    notes_zh: "用于回应“SimpleRCA 是否不公平”的问题。"

  - slide_no: "B2"
    title_en: "Fault-Propagation-Aware Benchmark Construction Details"
    message: "Expand the six-stage pipeline and impact-driven filtering criteria."
    visual_type: "backup"
    build_order: ["Detail six stages", "Detail validation thresholds", "Show reproducibility hooks"]
    asset_or_figure_needs: ["Detailed generation pipeline figure", "Validation criteria table-lite"]
    notes_zh: "用于回应“数据如何构建、是否可复现”的问题。"

  - slide_no: "B3"
    title_en: "OpenRCA Task Granularity and Metric Definition"
    message: "Clarify task decomposition and scoring assumptions."
    visual_type: "backup"
    build_order: ["List task units", "List scoring rules", "Clarify fairness assumptions"]
    asset_or_figure_needs: ["Task taxonomy card", "Metric definition table-lite"]
    notes_zh: "用于回应“任务定义和打分是否公平”的问题。"

  - slide_no: "B4"
    title_en: "RCA-Agent Strengths and Failure Patterns"
    message: "Show why execution-based agents improve performance but still fail in hard cases."
    visual_type: "backup"
    build_order: ["Show gain mechanism", "Show typical failure patterns", "State design trade-offs"]
    asset_or_figure_needs: ["Agent workflow schematic", "Failure pattern examples"]
    notes_zh: "用于回应“agent 设计为何有效、为何仍失败”的问题。"

  - slide_no: "B5"
    title_en: "FORGE Verification Pipeline and Metric Math"
    message: "Detail forward verification steps and process-level metric formulation."
    visual_type: "backup"
    build_order: ["Step through forward verification", "Define PR/Edge-style metrics", "Explain interpretation boundary"]
    asset_or_figure_needs: ["FORGE detailed pipeline", "Metric formula panel"]
    notes_zh: "用于回应“过程指标如何定义与计算”的问题。"

  - slide_no: "B6"
    title_en: "Threats to Validity Across the Whole Research Line"
    message: "Discuss representativeness, oracle assumptions, and external validity limits."
    visual_type: "backup"
    build_order: ["List major threats", "Explain expected impact", "Give mitigation direction"]
    asset_or_figure_needs: ["Threat matrix table-lite"]
    notes_zh: "用于回应“结论是否可泛化、有哪些限制”的问题。"
```
