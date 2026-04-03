from __future__ import annotations

from pathlib import Path

from paperops.slides import (
    Badge,
    Callout,
    Circle,
    Flow,
    Flowchart,
    Grid,
    HStack,
    Padding,
    Presentation as BasePresentation,
    RoundedBox,
    TextBlock,
    VStack,
    themes,
)


def make_theme():
    return themes.professional.override(
        font_family="Liberation Sans",
        colors={
            "primary": "#5C82AD",
            "secondary": "#8FAAC8",
            "accent": "#D69A68",
            "positive": "#7FA195",
            "negative": "#C98A78",
            "highlight": "#7396BE",
            "warning": "#D8A86A",
            "text": "#223043",
            "text_mid": "#647489",
            "text_light": "#94A3B6",
            "bg": "#FBFCFE",
            "bg_alt": "#F1F5FA",
            "bg_accent": "#EAF0F7",
            "border": "#D8E0EA",
        },
    )


def stat_card(label: str, value: str, tone: str = "primary"):
    return VStack(
        gap=0.10,
        children=[
            Badge(text=label, color=tone),
            TextBlock(
                text=value,
                font_size="heading",
                color="text",
                bold=True,
            ),
        ],
    )


SLIDE_TITLES = [
    "Building World Models for Diagnostic Intelligence",
    "The Vision",
    "Premise 1: Atomic Faults",
    "Premise 2: Simulatable",
    "Premise 3: Verifiable",
    "The Problem with RCA",
    "Effectiveness-Data-Cost Triangle",
    "Ideal RCA Formalization",
    "The Benchmark Crisis",
    "Why Existing Benchmarks Fail",
    "Our Solution",
    "The Process Gap",
    "FORGE: Forward Verification",
    "OpenRCA 2.0",
    "Triangle: Multi-Agent Triage",
    "Stage 1: Train the Agent",
    "Stage 2: Distill World Model",
    "Stage 3: Universal Expert",
    "My Research Philosophy",
    "Three Takeaways",
]
IMPLEMENTED_SLIDE_COUNT = len(SLIDE_TITLES)

OUTPUT_FILE = Path(__file__).with_name("diagnostic_world_model.pptx")


class Presentation(BasePresentation):
    @property
    def slides(self):
        return self._pptx.slides


def _speaker_notes(*segments: str) -> str:
    return " ".join(segment.strip() for segment in segments if segment.strip())


def _dark_stat_card(label: str, value: str, tone: str = "secondary"):
    return VStack(
        gap=0.10,
        children=[
            Badge(text=label, color=tone),
            TextBlock(
                text=value,
                font_size="heading",
                color="white",
                bold=True,
            ),
        ],
    )


def _closing_takeaway(number: str, message: str, tone: str = "accent"):
    return VStack(
        gap=0.08,
        width=3.45,
        children=[
            TextBlock(
                text=number,
                font_size=30,
                color=tone,
                bold=True,
            ),
            TextBlock(
                text=message,
                font_size="body",
                color="text",
                bold=True,
            ),
        ],
    )


def _build_slide_1(prs: Presentation):
    sb = prs.slide(background="bg_accent")

    hero_badge = Badge(text="Research Vision", color="accent")
    hero_title = TextBlock(
        text="Building World Models for\nDiagnostic Intelligence",
        font_size=28,
        color="primary",
        bold=True,
    )
    hero_subtitle = TextBlock(
        text="From outcome labels to causal process supervision in root cause analysis.",
        font_size="body",
        color="text_mid",
    )
    hero_claim = TextBlock(
        text="Move RCA from point finding to graph building.",
        font_size="body",
        color="accent",
        bold=True,
    )
    hero_stack = VStack(
        gap=0.18,
        width=7.2,
        children=[hero_badge, hero_title, hero_subtitle, hero_claim],
    )

    metric_stack = VStack(
        gap=0.24,
        width=3.4,
        children=[
            stat_card("Problem", "Oversimplified RCA", tone="negative"),
            stat_card("Method", "Forward verification", tone="secondary"),
            stat_card("Goal", "Trusted diagnosis", tone="positive"),
        ],
    )
    footer = TextBlock(
        text="Trusted diagnosis needs simulatable failures, verifiable reasoning, and explicit propagation structure.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(
        VStack(
            gap=0.40,
            children=[
                HStack(gap=0.60, children=[hero_stack, metric_stack]),
                footer,
            ],
        )
    )
    sb.notes(
        _speaker_notes(
            "Open by reframing RCA as a world-modeling problem rather than a label prediction problem.",
            "Transition by stating that the next slides unpack the three premises that make this view practical.",
        )
    )
    sb.animate(
        [
            [hero_badge, hero_title],
            [hero_subtitle, hero_claim],
            [metric_stack, footer],
        ]
    )
    return sb


def _build_slide_2(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[1], reference="Goal-driven RCA survey")

    left_badge = Badge(text="Black-box RCA", color="negative")
    left_flow = Flow(
        labels=["Telemetry", "LLM guess", "Root cause label"],
        colors=["bg_alt", "negative", "bg_alt"],
        arrow_color="text_light",
    )
    left_callout = Callout(
        title="Current pattern",
        body="Optimize for a correct answer, even if the causal path is hidden or hallucinated.",
        color="negative",
    )
    left_col = VStack(
        gap=0.18,
        width=5.5,
        children=[left_badge, left_flow, left_callout],
    )

    right_badge = Badge(text="World-model RCA", color="positive")
    right_flow = Flow(
        labels=["Telemetry", "Hypothesis", "Verify", "Graph + diagnosis"],
        colors=["bg_alt", "secondary", "accent", "positive"],
        arrow_color="text_light",
    )
    right_callout = Callout(
        title="Target behavior",
        body="Build and check a causal loop so the answer and the reasoning are both inspectable.",
        color="positive",
    )
    right_col = VStack(
        gap=0.18,
        width=5.5,
        children=[right_badge, right_flow, right_callout],
    )

    bridge = TextBlock(
        text="Black-box RCA predicts an answer. World-model RCA closes a causal loop.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.28, children=[HStack(gap=0.40, children=[left_col, right_col]), bridge]))
    sb.notes(
        _speaker_notes(
            "Contrast today's black-box symptom-to-label flow with a world-model loop that hypothesizes and verifies propagation.",
            "Transition by asking what assumptions must be true for that loop to be feasible.",
        )
    )
    sb.animate([[left_col], [right_col], [bridge]])
    return sb


def _build_slide_3(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[2], reference="OpenRCA 2.0 / FORGE")

    premise = Callout(
        title="Atomic premise",
        body="Reason in terms of finite atomic fault types, then study how they compose into larger incidents.",
        color="secondary",
    )
    structure = Flow(
        labels=["Finite", "Bounded", "Composable"],
        colors=["secondary", "bg_alt", "accent"],
        arrow_color="text_light",
    )
    fault_grid = Grid(
        cols=3,
        gap=0.16,
        children=[
            RoundedBox(text="CPU stress", color="bg_alt", height=0.78),
            RoundedBox(text="Memory leak", color="bg_alt", height=0.78),
            RoundedBox(text="Packet loss", color="bg_alt", height=0.78),
            RoundedBox(text="Delay spike", color="bg_alt", height=0.78),
            RoundedBox(text="Config drift", color="bg_alt", height=0.78),
            RoundedBox(text="Code regression", color="bg_alt", height=0.78),
        ],
    )
    conclusion = TextBlock(
        text="Large incidents are combinations of local faults, not an unbounded alphabet of new failure modes.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.24, children=[premise, structure, fault_grid, conclusion]))
    sb.notes(
        _speaker_notes(
            "State the first premise: the intervention space is broad but still bounded enough to model.",
            "Transition by noting that bounded atomic faults only help if we can replay them inside a system.",
        )
    )
    sb.animate([[premise], [structure, fault_grid], [conclusion]])
    return sb


def _build_slide_4(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[3], reference="OpenRCA 2.0 / FORGE")

    pipeline = Flowchart(
        nodes={
            "inject": ("Fault injection", "accent"),
            "workload": ("Dynamic workload", "secondary"),
            "observe": ("Observability", "primary"),
            "cases": ("Validated cases", "positive"),
        },
        edges=[
            ("inject", "workload"),
            ("workload", "observe"),
            ("observe", "cases"),
        ],
        direction="right",
        height=1.00,
    )
    evidence = Grid(
        cols=3,
        gap=0.18,
        children=[
            Callout(
                title="Replayable",
                body="Known interventions make the incident generation process reproducible.",
                color="secondary",
            ),
            Callout(
                title="Operationally relevant",
                body="Keep only failures that produce observable user-facing impact.",
                color="positive",
            ),
            Callout(
                title="Causal evidence",
                body="The intervention gives a concrete starting point for tracing propagation.",
                color="accent",
            ),
        ],
    )
    takeaway = TextBlock(
        text="Simulation turns RCA data collection from passive observation into controlled causal experimentation.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.28, children=[pipeline, evidence, takeaway]))
    sb.notes(
        _speaker_notes(
            "Introduce the second premise: if we can inject faults under workload, we can watch failures propagate instead of merely reading symptoms after the fact.",
            "Transition by saying replay alone is insufficient unless we can score whether a reasoning path matches reality.",
        )
    )
    sb.animate([[pipeline], [evidence], [takeaway]])
    return sb


def _build_slide_5(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[4], reference="OpenRCA 2.0 / FORGE")

    loop = Flow(
        labels=["Hypothesis", "Verify", "Reward / fail", "Update"],
        colors=["bg_alt", "secondary", "accent", "positive"],
        arrow_color="text_light",
    )
    truth = Callout(
        title="Ground truth anchor",
        body="Verified propagation tells us whether the reasoning path is faithful, not just whether the final answer is lucky.",
        color="positive",
        height=1.15,
    )
    bridge = TextBlock(
        text="Ground truth closes the learning loop: hypotheses are checked against the incident path, then revised.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.32, children=[loop, truth, bridge]))
    sb.notes(
        _speaker_notes(
            "Make the third premise explicit: a diagnosis should earn reward only when the causal path checks out against ground truth.",
            "Transition by saying that once world-model RCA is feasible, the real bottleneck becomes how RCA is defined and evaluated today.",
        )
    )
    sb.animate([[loop], [truth], [bridge]])
    return sb


def _build_slide_6(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[5], reference="Goal-driven RCA survey")

    old_taxonomy = VStack(
        gap=0.16,
        width=5.5,
        children=[
            Badge(text="Old taxonomy", color="negative"),
            RoundedBox(text="Metrics / logs / traces", color="bg_alt", height=0.86),
            TextBlock(
                text="Groups papers by input modality, even when their operational goals differ.",
                font_size="body",
                color="text",
            ),
            TextBlock(
                text="This obscures progress because the same data can support very different RCA objectives.",
                font_size="caption",
                color="text_mid",
            ),
        ],
    )

    goal_taxonomy = VStack(
        gap=0.16,
        width=5.5,
        children=[
            Badge(text="Goal-driven taxonomy", color="positive"),
            Flow(
                labels=["Triage", "Mitigation", "Resolution"],
                colors=["secondary", "accent", "positive"],
                arrow_color="text_light",
            ),
            Callout(
                title="Why it fits practice",
                body="Incident management cares about speed, actionability, interpretability, and granularity, not only data modality.",
                color="positive",
            ),
        ],
    )

    conclusion = TextBlock(
        text="Survey lens: organize RCA by the job it needs to do in incident management, not only by the telemetry it consumes.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.28, children=[HStack(gap=0.40, children=[old_taxonomy, goal_taxonomy]), conclusion]))
    sb.notes(
        _speaker_notes(
            "Use the survey to argue that the classical metrics-logs-traces taxonomy is exactly where the field drifts away from the world-model target.",
            "Transition by showing the deeper systems constraint that shapes all practical RCA methods.",
        )
    )
    sb.animate([[old_taxonomy], [goal_taxonomy], [conclusion]])
    return sb


def _build_slide_7(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[6], reference="Goal-driven RCA survey")

    effectiveness_circle = Circle(text="Effectiveness", color="primary", width=1.85, height=1.85)
    data_circle = Circle(text="Data", color="secondary", width=1.65, height=1.65)
    cost_circle = Circle(text="Cost", color="warning", width=1.65, height=1.65)

    triangle = VStack(
        gap=0.14,
        children=[
            HStack(
                gap=0.20,
                children=[Padding(width=3.0), effectiveness_circle, Padding(width=3.0)],
            ),
            HStack(gap=1.00, children=[data_circle, cost_circle]),
        ],
    )
    detail_row = HStack(
        gap=0.20,
        children=[
            Callout(
                title="Effectiveness",
                body="Better diagnosis quality and broader causal coverage.",
                color="primary",
            ),
            Callout(
                title="Data",
                body="More telemetry, more modalities, deeper observability.",
                color="secondary",
            ),
            Callout(
                title="Cost",
                body="Ingestion, storage, and compute rise with scope.",
                color="warning",
            ),
        ],
    )
    tension = TextBlock(
        text="Better RCA needs richer data, and richer data raises cost.",
        font_size="small",
        color="text_mid",
        italic=True,
    )

    sb.layout(
        VStack(
            gap=0.14,
            children=[triangle, detail_row, tension],
        )
    )
    sb.notes(
        _speaker_notes(
            "Explain the survey's Effectiveness-Data-Cost triangle as the governing trade-off behind practical RCA systems.",
            "Transition by saying that once the trade-off is clear, we can define the ideal north-star formulation explicitly.",
        )
    )
    sb.animate(
        [
            [effectiveness_circle],
            [data_circle, cost_circle, detail_row],
            [tension],
        ]
    )
    return sb


def _build_slide_8(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[7], reference="Goal-driven RCA survey")

    formula = TextBlock(
        text="F: O -> G",
        font_size=30,
        color="primary",
        bold=True,
        align="center",
    )
    framing = Flow(
        labels=["Observation", "Inference", "Propagation graph"],
        colors=["bg_alt", "secondary", "positive"],
        arrow_color="text_light",
    )
    spaces = HStack(
        gap=0.28,
        children=[
            Callout(
                title="O: observation data",
                body="Logs, metrics, traces, events, and supplementary context from the system.",
                color="secondary",
            ),
            Callout(
                title="G: incident propagation graph",
                body="Root causes, triggers, symptoms, and the causal edges that connect them.",
                color="positive",
            ),
        ],
    )
    conclusion = TextBlock(
        text="Ideal RCA is graph-building: the output is not only a root-cause node, but the propagation graph that explains why it is correct.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.28, children=[formula, framing, spaces, conclusion]))
    sb.notes(
        _speaker_notes(
            "Present the survey's formalization F: O -> G and stress that G is an incident propagation graph, not merely a ranked root-cause label.",
            "Transition by using that north star to diagnose why existing benchmarks create a misleading sense of progress.",
        )
    )
    sb.animate([[formula], [framing, spaces], [conclusion]])
    return sb


def _build_slide_9(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[8], reference="FSE RCA benchmark study")

    simple_baseline = VStack(
        gap=0.16,
        width=5.5,
        children=[
            Badge(text="Simple baseline", color="accent"),
            RoundedBox(
                text="Alerts + thresholds\nrule-based ranking",
                color="bg_alt",
                height=1.10,
            ),
            TextBlock(
                text="Can match or beat sophisticated models on several public RCA benchmarks.",
                font_size="body",
                color="text",
            ),
        ],
    )
    sota = VStack(
        gap=0.16,
        width=5.5,
        children=[
            Badge(text="Published SOTA", color="primary"),
            RoundedBox(
                text="Graph models\nmulti-modal reasoning\ncausal inference",
                color="bg_alt",
                height=1.28,
            ),
            TextBlock(
                text="Looks strong on paper, but the benchmark may be testing easy correlations instead of hard causality.",
                font_size="body",
                color="text",
            ),
        ],
    )
    verdict = Callout(
        title="Benchmark crisis",
        body="If a simple baseline rivals SOTA, the benchmark is oversimplifying real RCA difficulty.",
        color="negative",
    )

    sb.layout(VStack(gap=0.28, children=[HStack(gap=0.40, children=[simple_baseline, sota]), verdict]))
    sb.notes(
        _speaker_notes(
            "Use the dataset study's headline result: simple heuristics are surprisingly competitive with SOTA on public benchmarks.",
            "Transition by asking what structural defects make those benchmarks so forgiving.",
        )
    )
    sb.animate([[simple_baseline], [sota], [verdict]])
    return sb


def _build_slide_10(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[9], reference="FSE RCA benchmark study")

    banner = Badge(text="Public benchmarks are oversimplified", color="negative")
    defect_one = Callout(
        title="Fault injection strategies",
        body="Too few faults, too localized, and too easy to detect through direct symptom prominence.",
        color="negative",
    )
    defect_two = Callout(
        title="Call graph structures",
        body="Shallow request paths and limited service coverage shrink the search space.",
        color="warning",
    )
    defect_three = Callout(
        title="Telemetry signal patterns",
        body="Signals are clean and obvious instead of partial, noisy, and operationally messy.",
        color="secondary",
    )
    conclusion = TextBlock(
        text="When fault symptoms are obvious and structure is shallow, correlation can masquerade as causality.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(
        VStack(
            gap=0.26,
            children=[
                banner,
                Grid(cols=3, gap=0.20, children=[defect_one, defect_two, defect_three]),
                conclusion,
            ],
        )
    )
    sb.notes(
        _speaker_notes(
            "Summarize the three defects from the benchmark study: fault injection, dependency structure, and telemetry patterns.",
            "Transition by showing the concrete benchmark pipeline built to fix these weaknesses.",
        )
    )
    sb.animate([[banner, defect_one], [defect_two], [defect_three, conclusion]])
    return sb


def _build_slide_11(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[10], reference="FSE RCA benchmark study")

    pipeline = Flowchart(
        nodes={
            "inject": ("9,152 injections", "accent"),
            "cases": ("1,430 validated cases", "secondary"),
            "faults": ("25 fault types", "positive"),
        },
        edges=[
            ("inject", "cases"),
            ("cases", "faults"),
        ],
        direction="right",
        height=1.05,
    )
    scores = HStack(
        gap=0.36,
        children=[
            stat_card("Average Top@1", "0.21", tone="negative"),
            stat_card("Best model", "0.37", tone="warning"),
            Callout(
                title="What changed",
                body="A harder benchmark preserves complex propagation and exposes a real performance collapse.",
                color="primary",
            ),
        ],
    )
    conclusion = TextBlock(
        text="Our solution is not only more data. It is validated, propagation-aware data that reveals how fragile current RCA systems still are.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.32, children=[pipeline, scores, conclusion]))
    sb.notes(
        _speaker_notes(
            "Close the first section with the benchmark pipeline and the core result numbers: 9,152 injections, 1,430 validated cases, 25 fault types, 0.21 average Top@1, and 0.37 best.",
            "Transition by saying the rest of the deck will explain the process-level supervision and benchmark design behind this result.",
        )
    )
    sb.animate([[pipeline], [scores], [conclusion]])
    return sb


def _build_slide_12(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[11], reference="OpenRCA 2.0 / FORGE")

    outcome = VStack(
        gap=0.18,
        width=5.45,
        children=[
            Badge(text="Outcome correctness", color="secondary"),
            stat_card("Pass@1", "0.76", tone="secondary"),
            Callout(
                title="What it means",
                body="The best model often names the right root cause when judged only by the final answer.",
                color="secondary",
            ),
        ],
    )
    process = VStack(
        gap=0.18,
        width=5.45,
        children=[
            Badge(text="Process correctness", color="warning"),
            stat_card("Path Reachability", "0.63", tone="warning"),
            Callout(
                title="What it means",
                body="A sizable slice of those correct answers still cannot trace a valid path from cause to symptom.",
                color="warning",
            ),
        ],
    )
    gap = Callout(
        title="Process-level reasoning gap",
        body="0.76 Pass@1 versus 0.63 Path Reachability exposes the core failure mode: correct outcome, broken process.",
        color="negative",
    )
    conclusion = TextBlock(
        text="Outcome correctness flatters the agent. Process correctness tells us whether the diagnosis can be trusted.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.28, children=[HStack(gap=0.40, children=[outcome, process]), gap, conclusion]))
    sb.notes(
        _speaker_notes(
            "Use this slide to separate outcome correctness from process correctness with the 0.76 versus 0.63 comparison.",
            "Transition by asking how to turn an ill-posed backward RCA task into something verifiable enough to supervise.",
        )
    )
    sb.animate([[outcome], [process], [gap, conclusion]])
    return sb


def _build_slide_13(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[12], reference="OpenRCA 2.0 / FORGE")

    backward = VStack(
        gap=0.18,
        width=5.35,
        children=[
            Badge(text="Backward diagnosis", color="negative"),
            Flow(
                labels=["Symptoms", "Unknown cause", "Ill-posed search"],
                colors=["bg_alt", "negative", "bg_alt"],
                arrow_color="text_light",
            ),
            Callout(
                title="Why it is hard",
                body="The RCA agent must infer an unseen cause from partial telemetry, confounders, and noisy symptoms.",
                color="negative",
            ),
        ],
    )
    forward = VStack(
        gap=0.18,
        width=5.35,
        children=[
            Badge(text="Forward verification", color="positive"),
            Flow(
                labels=["Known intervention", "Expected signatures", "Verified path"],
                colors=["accent", "secondary", "positive"],
                arrow_color="text_light",
            ),
            Callout(
                title="Why FORGE works",
                body="Knowing the intervention turns diagnosis into a well-posed verification task from cause to effect.",
                color="positive",
            ),
        ],
    )
    asymmetry = TextBlock(
        text="FORGE exploits the asymmetry: backward diagnosis is ill-posed, but forward verification is tractable when the intervention is known.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.28, children=[HStack(gap=0.40, children=[backward, forward]), asymmetry]))
    sb.notes(
        _speaker_notes(
            "Explain the information asymmetry at the heart of FORGE: the agent reasons backward, while annotation can verify forward from the known intervention.",
            "Transition by turning that verification mechanism into a concrete benchmark identity.",
        )
    )
    sb.animate([[backward], [forward], [asymmetry]])
    return sb


def _build_slide_14(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[13], reference="OpenRCA 2.0 / FORGE")

    summary = HStack(
        gap=0.28,
        children=[
            stat_card("Benchmark", "500 instances", tone="primary"),
            stat_card("Labels", "step-wise causal annotations", tone="accent"),
            stat_card("Supervision", "process supervision", tone="positive"),
        ],
    )
    dimensions = Flow(
        labels=["Known intervention", "Verified propagation", "Agent graph", "Process metrics"],
        colors=["accent", "secondary", "positive", "primary"],
        arrow_color="text_light",
    )
    identity = Callout(
        title="Benchmark identity",
        body="OpenRCA 2.0 is not just a harder label dataset. It is a process benchmark with step-wise supervision that scores how the diagnosis is constructed.",
        color="primary",
    )
    takeaway = TextBlock(
        text="OpenRCA 2.0 gives us a benchmark where benchmark quality and training signal finally align.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.28, children=[summary, dimensions, identity, takeaway]))
    sb.notes(
        _speaker_notes(
            "Anchor the benchmark identity in three facts: 500 instances, step-wise causal annotations, and process supervision.",
            "Transition by broadening from RCA diagnosis to the adjacent triage workflow where multi-agent coordination already shows practical value.",
        )
    )
    sb.animate([[summary], [dimensions], [identity, takeaway]])
    return sb


def _build_slide_15(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[14], reference="TRIANGLE paper")

    adjacent_badge = Badge(text="Adjacent evidence", color="accent")
    framework = Flowchart(
        nodes={
            "analyzer": ("Analyzer Agent", "secondary"),
            "decider": ("Decider Agent", "accent"),
            "manager": ("Team Manager Agent", "positive"),
            "assignment": ("Team assignment", "primary"),
        },
        edges=[
            ("analyzer", "decider"),
            ("decider", "manager"),
            ("manager", "decider"),
            ("decider", "assignment"),
        ],
        direction="right",
        height=1.10,
    )
    roles = HStack(
        gap=0.24,
        children=[
            Callout(
                title="Analyzer Agent",
                body="Semantically distills noisy incident text into the core location, symptom, and capability signals.",
                color="secondary",
            ),
            Callout(
                title="Decider Agent",
                body="Ranks candidate teams and orchestrates the negotiation over who should own the incident.",
                color="accent",
            ),
            Callout(
                title="Team Manager Agent",
                body="Contributes team-specific knowledge, retrieves context, and votes during collaboration.",
                color="positive",
            ),
        ],
    )
    outcomes = HStack(
        gap=0.32,
        children=[
            stat_card("Triage accuracy", "97%", tone="positive"),
            stat_card("TTE reduction", "91%", tone="accent"),
            Callout(
                title="Lesson",
                body="Role separation plus negotiation can outperform one monolithic prompt on high-stakes routing.",
                color="primary",
                height=1.34,
            ),
        ],
    )

    sb.layout(VStack(gap=0.24, children=[adjacent_badge, framework, roles, outcomes]))
    sb.notes(
        _speaker_notes(
            "Frame TRIANGLE as adjacent evidence rather than the main RCA contribution: structured role separation already works in a neighboring reliability task.",
            "Transition by saying the next step is to turn that structured reliability reasoning into a reusable diagnostic world model.",
        )
    )
    sb.animate([[adjacent_badge, framework], [roles], [outcomes]])
    return sb


def _build_slide_16(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[15])

    badge = Badge(text="Roadmap part 1", color="accent")
    loop = Flow(
        labels=["Simulation", "Hypothesis", "Verify", "Reward"],
        colors=["accent", "secondary", "positive", "warning"],
        arrow_color="text_light",
    )
    training = Callout(
        title="Train on closed loops",
        body="Start with interventions and telemetry so the agent repeatedly proposes a hypothesis, checks it, and gets feedback on the full loop.",
        color="primary",
    )
    takeaway = TextBlock(
        text="The first milestone is an agent that learns from repeated simulation to verification cycles, not from answer labels alone.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.32, children=[badge, loop, training, takeaway]))
    sb.notes(
        _speaker_notes(
            "Build on the adjacent evidence from TRIANGLE and move into the main roadmap: stage one is training an agent inside a simulation-hypothesis-verify-reward loop.",
            "Transition by asking what remains after we collect many successful traces from that loop.",
        )
    )
    sb.animate([[badge, loop], [training], [takeaway]])
    return sb


def _build_slide_17(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[16])

    sources = VStack(
        gap=0.14,
        width=3.4,
        children=[
            Badge(text="Many traces", color="secondary"),
            RoundedBox(text="Incident trace A", color="bg_alt", height=0.60),
            RoundedBox(text="Incident trace B", color="bg_alt", height=0.60),
            RoundedBox(text="Incident trace C", color="bg_alt", height=0.60),
        ],
    )
    compression = Flow(
        labels=["Many traces", "Compress", "world model"],
        colors=["secondary", "accent", "primary"],
        arrow_color="text_light",
    )
    model = VStack(
        gap=0.16,
        width=4.2,
        children=[
            Badge(text="Reusable core", color="primary"),
            Circle(text="world\nmodel", color="primary", width=2.15, height=2.15),
            TextBlock(
                text="A compact causal representation that preserves propagation structure while dropping repetitive incident detail.",
                font_size="body",
                color="text",
            ),
        ],
    )

    sb.layout(VStack(gap=0.24, children=[compression, HStack(gap=0.55, children=[sources, model])]))
    sb.notes(
        _speaker_notes(
            "Stage two is compression: distill many traces into one reusable world model without losing the causal structure we care about.",
            "Transition by showing that once the world model exists, the same core should branch into multiple operational tasks.",
        )
    )
    sb.animate([[compression], [sources], [model]])
    return sb


def _build_slide_18(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[17])

    framework = Flowchart(
        nodes={
            "model": ("World model", "primary"),
            "localize": ("Root-cause localization", "secondary"),
            "predict": ("Fault prediction", "accent"),
            "repair": ("Repair suggestion", "positive"),
        },
        edges=[
            ("model", "localize"),
            ("model", "predict"),
            ("model", "repair"),
        ],
        direction="right",
        height=1.10,
    )
    bridge = Callout(
        title="One reasoning core, many tasks",
        body="The end state is a universal expert that reuses the same causal world model for localization, forecasting, and actionable support.",
        color="primary",
    )
    conclusion = TextBlock(
        text="The value of the world model is leverage: one internal representation fans out to multiple reliability workflows.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.32, children=[framework, bridge, conclusion]))
    sb.notes(
        _speaker_notes(
            "Stage three is where the roadmap cashes out: one world model should support root-cause localization, fault prediction, and repair suggestion.",
            "Transition by summarizing the research philosophy that connects these stages.",
        )
    )
    sb.animate([[framework], [bridge], [conclusion]])
    return sb


def _build_slide_19(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[18])

    header = Badge(text="Three research capabilities", color="primary")
    pillars = Grid(
        cols=3,
        gap=0.20,
        children=[
            Callout(
                title="Simulate",
                body="Create controllable failures so the agent can learn from interventions and propagation, not only historical anecdotes.",
                color="secondary",
            ),
            Callout(
                title="Compress",
                body="Turn many incident traces into a compact world model that remains inspectable and reusable.",
                color="accent",
            ),
            Callout(
                title="Generalize",
                body="Reuse one reasoning core across triage, diagnosis, prediction, and repair support.",
                color="positive",
            ),
        ],
    )
    philosophy = TextBlock(
        text="My philosophy is to build diagnostic systems that can simulate reality, compress it into structure, and generalize that structure across tasks.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.30, children=[header, pillars, philosophy]))
    sb.notes(
        _speaker_notes(
            "Condense the roadmap into three research capabilities: simulate, compress, and generalize.",
            "Transition by turning those capabilities into the three closing takeaways for the audience.",
        )
    )
    sb.animate([[header], [pillars], [philosophy]])
    return sb


def _build_slide_20(prs: Presentation):
    sb = prs.slide(background="bg_accent")

    hero = VStack(
        gap=0.12,
        children=[
            TextBlock(
                text="Three Takeaways",
                font_size=30,
                color="primary",
                bold=True,
            ),
            TextBlock(
                text="What matters is not only finding the right answer, but building a reasoning system we can trust and reuse.",
                font_size="body",
                color="text_mid",
            ),
        ],
    )
    takeaways = HStack(
        gap=0.45,
        children=[
            _closing_takeaway("1", "Score the path, not just the answer.", tone="accent"),
            _closing_takeaway("2", "Train on verification loops, not only labels.", tone="secondary"),
            _closing_takeaway("3", "Build the model once, then reuse it broadly.", tone="positive"),
        ],
    )
    final_line = TextBlock(
        text="Build the model once. Reuse it across diagnosis, prediction, and repair.",
        font_size=22,
        color="primary",
        bold=True,
        align="center",
    )

    sb.layout(VStack(gap=0.42, children=[hero, takeaways, final_line]))
    sb.notes(
        _speaker_notes(
            "Close with three short takeaways rather than a dense recap, keeping the typography strong and the message memorable.",
            "End on the final line as the thesis statement for the whole talk.",
        )
    )
    sb.animate([[hero], [takeaways], [final_line]])
    return sb


def build_presentation(output_path: Path | None = None, render_preview: bool = False):
    prs = Presentation(theme=make_theme())
    implemented_slide_builders = (
        _build_slide_1,
        _build_slide_2,
        _build_slide_3,
        _build_slide_4,
        _build_slide_5,
        _build_slide_6,
        _build_slide_7,
        _build_slide_8,
        _build_slide_9,
        _build_slide_10,
        _build_slide_11,
        _build_slide_12,
        _build_slide_13,
        _build_slide_14,
        _build_slide_15,
        _build_slide_16,
        _build_slide_17,
        _build_slide_18,
        _build_slide_19,
        _build_slide_20,
    )
    for builder in implemented_slide_builders[:IMPLEMENTED_SLIDE_COUNT]:
        builder(prs)

    save_path = output_path or OUTPUT_FILE
    prs.save(str(save_path))
    if render_preview:
        preview_dir = (
            save_path.parent / "preview"
            if output_path is not None
            else Path(__file__).with_name("preview")
        )
        preview_dir.mkdir(parents=True, exist_ok=True)
        for png_path in preview_dir.glob("slide_*.png"):
            png_path.unlink()
        prs.preview(output_dir=str(preview_dir))
    return prs


if __name__ == "__main__":
    build_presentation()
