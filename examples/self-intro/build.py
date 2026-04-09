from __future__ import annotations

from pathlib import Path

from paperops.slides import (
    Arrow,
    Badge,
    Circle,
    Grid,
    HStack,
    Padding,
)
from paperops.slides import Presentation as BasePresentation
from paperops.slides import (
    RoundedBox,
    SvgImage,
    TextBlock,
    VStack,
    themes,
)


def icon_agent(theme, size=80):
    """Agent icon - AI/robot representation."""
    primary = theme.resolve_color("primary") if theme else "#5C82AD"
    white = "#FFFFFF"
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <rect x="25" y="30" width="50" height="45" rx="8" fill="{primary}"/>
        <line x1="50" y1="30" x2="50" y2="15" stroke="{primary}" stroke-width="3"/>
        <circle cx="50" cy="12" r="4" fill="{primary}"/>
        <rect x="35" y="45" width="12" height="10" rx="2" fill="{white}"/>
        <rect x="53" y="45" width="12" height="10" rx="2" fill="{white}"/>
        <rect x="40" y="62" width="20" height="4" rx="2" fill="{white}"/>
    </svg>"""


def icon_world_model(theme, size=80):
    """World Model icon - Globe with network."""
    primary = theme.resolve_color("primary") if theme else "#5C82AD"
    secondary = theme.resolve_color("secondary") if theme else "#8FAAC8"
    accent = theme.resolve_color("accent") if theme else "#D69A68"
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <circle cx="50" cy="50" r="35" fill="none" stroke="{primary}" stroke-width="3"/>
        <ellipse cx="50" cy="50" rx="35" ry="15" fill="none" stroke="{secondary}" stroke-width="1.5"/>
        <ellipse cx="50" cy="50" rx="35" ry="25" fill="none" stroke="{secondary}" stroke-width="1.5"/>
        <ellipse cx="50" cy="50" rx="15" ry="35" fill="none" stroke="{secondary}" stroke-width="1.5"/>
        <ellipse cx="50" cy="50" rx="25" ry="35" fill="none" stroke="{secondary}" stroke-width="1.5"/>
        <circle cx="50" cy="50" r="8" fill="{accent}"/>
    </svg>"""


def icon_database(theme, size=80):
    """Database icon - Cylinder representation."""
    primary = theme.resolve_color("primary") if theme else "#5C82AD"
    secondary = theme.resolve_color("secondary") if theme else "#8FAAC8"
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <path d="M20 35 L20 70 Q20 80 50 80 Q80 80 80 70 L80 35" fill="{primary}" opacity="0.8"/>
        <ellipse cx="50" cy="35" rx="30" ry="10" fill="{primary}"/>
        <path d="M25 50 Q50 55 75 50" stroke="{secondary}" stroke-width="2" fill="none"/>
        <path d="M25 62 Q50 67 75 62" stroke="{secondary}" stroke-width="2" fill="none"/>
    </svg>"""


def icon_process_flow(theme, size=80):
    """Process Flow icon - Pipeline with nodes."""
    secondary = theme.resolve_color("secondary") if theme else "#8FAAC8"
    accent = theme.resolve_color("accent") if theme else "#D69A68"
    positive = theme.resolve_color("positive") if theme else "#7FA195"
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <line x1="20" y1="50" x2="80" y2="50" stroke="#D8E0EA" stroke-width="4"/>
        <circle cx="20" cy="50" r="12" fill="{secondary}"/>
        <circle cx="50" cy="50" r="12" fill="{accent}"/>
        <circle cx="80" cy="50" r="12" fill="{positive}"/>
        <polygon points="72,45 82,50 72,55" fill="#D8E0EA"/>
    </svg>"""


def icon_analysis(theme, size=80):
    """Analysis icon - Bar chart."""
    primary = theme.resolve_color("primary") if theme else "#5C82AD"
    secondary = theme.resolve_color("secondary") if theme else "#8FAAC8"
    accent = theme.resolve_color("accent") if theme else "#D69A68"
    positive = theme.resolve_color("positive") if theme else "#7FA195"
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <rect x="15" y="60" width="15" height="30" rx="2" fill="{primary}"/>
        <rect x="35" y="45" width="15" height="45" rx="2" fill="{secondary}"/>
        <rect x="55" y="30" width="15" height="60" rx="2" fill="{accent}"/>
        <rect x="75" y="50" width="15" height="40" rx="2" fill="{positive}"/>
    </svg>"""


def make_theme():
    return themes.academic_seminar


def Callout(title: str, body: str, color: str = "primary", width: float | None = None, height: float | None = None):
    return VStack(
        gap=0.08,
        width=width,
        height=height,
        children=[
            TextBlock(text=title, font_size="body", color=color, bold=True),
            TextBlock(text=body, font_size="caption", color="text"),
        ],
    )


def Flow(
    labels: list[str],
    colors: list[str] | None = None,
    arrow_color: str = "primary",
    direction: str = "horizontal",
    width: float | None = None,
    height: float | None = None,
):
    box_colors = list(colors or ["bg_alt"] * len(labels))
    while len(box_colors) < len(labels):
        box_colors.append("bg_alt")
    boxes = [
        RoundedBox(
            text=label,
            color=color,
            border="border" if color in {"bg_alt", "bg", "bg_accent"} else color,
            text_color="white" if color not in {"bg_alt", "bg", "bg_accent"} else "text",
            font_size="caption",
            bold=True,
            height=0.82,
            size_mode_x="fit",
        )
        for label, color in zip(labels, box_colors)
    ]
    children = []
    arrow_direction = "vertical" if direction in {"vertical", "down"} else "horizontal"
    for index, box in enumerate(boxes):
        children.append(box)
        if index < len(boxes) - 1:
            children.append(
                Arrow(
                    from_component=box,
                    to_component=boxes[index + 1],
                    color=arrow_color,
                    direction=arrow_direction,
                )
            )
    container_cls = VStack if arrow_direction == "vertical" else HStack
    return container_cls(gap=0.15, width=width, height=height, children=children)


def Flowchart(
    nodes: dict,
    edges: list,
    direction: str = "right",
    node_widths: dict[str, float] | None = None,
    height: float | None = None,
    width: float | None = None,
):
    node_widths = node_widths or {}
    order = list(nodes.keys())
    outgoing: dict[str, list[str]] = {}
    incoming: dict[str, int] = {}
    for edge in edges:
        src, dst = edge[0], edge[1]
        outgoing.setdefault(src, []).append(dst)
        incoming[dst] = incoming.get(dst, 0) + 1
    is_chain = all(len(targets) <= 1 for targets in outgoing.values()) and all(count <= 1 for count in incoming.values())
    if is_chain and edges:
        starts = [node_id for node_id in order if incoming.get(node_id, 0) == 0]
        if starts:
            ordered = []
            current = starts[0]
            seen = set()
            while current not in seen:
                seen.add(current)
                ordered.append(current)
                next_nodes = outgoing.get(current, [])
                if not next_nodes:
                    break
                current = next_nodes[0]
            ordered.extend(node_id for node_id in order if node_id not in seen)
            order = ordered
        labels = []
        colors = []
        for node_id in order:
            value = nodes[node_id]
            if isinstance(value, tuple):
                label, color = value[0], value[1]
            else:
                label, color = value, "bg_alt"
            labels.append(label)
            colors.append(color)
        return Flow(
            labels=labels,
            colors=colors,
            arrow_color="text_light",
            direction="vertical" if direction in {"down", "vertical"} else "horizontal",
            width=width,
            height=height,
        )

    cards = []
    for node_id, value in nodes.items():
        if isinstance(value, tuple):
            label, color = value[0], value[1]
        else:
            label, color = value, "bg_alt"
        cards.append(
            RoundedBox(
                text=label,
                color=color,
                border="border" if color in {"bg_alt", "bg", "bg_accent"} else color,
                text_color="white" if color not in {"bg_alt", "bg", "bg_accent"} else "text",
                bold=True,
                font_size="caption",
                width=node_widths.get(node_id),
                height=height if height is not None else 0.92,
                size_mode_x="fit" if node_widths.get(node_id) is None else "fixed",
            )
        )
    cols = min(max(len(cards), 1), 3)
    return Grid(cols=cols, gap=0.2, width=width, children=cards)


def stat_card(
    label: str,
    value: str,
    tone: str = "primary",
    *,
    width: float | None = None,
    value_font_size: str | float = "heading",
):
    return VStack(
        gap=0.04,
        width=width,
        children=[
            TextBlock(
                text=label.upper(),
                font_size=9,
                color=tone,
                bold=True,
                line_spacing=1.0,
                height=0.38,
            ),
            TextBlock(
                text=value,
                font_size=value_font_size,
                color="text",
                bold=True,
            ),
        ],
    )


SLIDE_TITLES = [
    "Diagnostic intelligence should be trained around world models",
    "World-model RCA should verify causal loops, not predict labels",
    "Fault injection makes RCA learnable",
    "Simulation turns RCA data into controlled experiments",
    "Verified propagation makes reasoning scoreable",
    "RCA should produce graphs, not isolated answers",
    "Public RCA benchmarks reward shortcuts",
    "Existing benchmarks are shallow, static, and point-based",
    "A harder benchmark exposes real performance collapse",
    "Better RCA is constrained by the effectiveness-data-cost trade-off",
    "Process supervision is the missing training signal",
    "Forward verification makes process supervision tractable",
    "OpenRCA 2.0 aligns benchmark quality with training signal",
    "Structured reasoning already works in production",
    "Closed-loop simulation can generate RCA training data",
    "Many verified traces can be distilled into a reusable world model",
    "World models create leverage across the diagnostic workflow",
    "The opportunity is to train around causal structure",
]
IMPLEMENTED_SLIDE_COUNT = len(SLIDE_TITLES)

OUTPUT_FILE = Path(__file__).with_name("diagnostic_world_model.pptx")


class Presentation(BasePresentation):
    @property
    def slides(self):
        return self._pptx.slides

    def evidence_comparison(
        self,
        title: str,
        left_title: str,
        left_visual,
        right_title: str,
        right_visual,
        conclusion: str,
        reference: str | None = None,
    ):
        sb = self.slide(title=title, reference=reference)
        left = VStack(
            gap=0.12,
            width=5.1,
            children=[
                TextBlock(text=left_title, font_size="heading", color="primary", bold=True),
                left_visual,
            ],
        )
        right = VStack(
            gap=0.12,
            width=5.1,
            children=[
                TextBlock(text=right_title, font_size="heading", color="secondary", bold=True),
                right_visual,
            ],
        )
        sb.layout(
            VStack(
                gap=0.18,
                children=[
                    HStack(gap=0.32, children=[left, right]),
                    TextBlock(text=conclusion, font_size="body", color="text", bold=True),
                ],
            )
        )
        return sb

    def seminar_takeaway(
        self,
        title: str,
        message: str,
        evidence: list[str] | None = None,
        reference: str | None = None,
    ):
        sb = self.slide(title=title, reference=reference)
        children = [
            TextBlock(text=message, font_size="heading", color="primary", bold=True),
        ]
        if evidence:
            children.append(
                VStack(
                    gap=0.12,
                    children=[
                        TextBlock(text=item, font_size="body", color="text")
                        for item in evidence
                    ],
                )
            )
        sb.layout(VStack(gap=0.22, children=children))
        return sb


def _speaker_notes(*segments: str) -> str:
    return " ".join(segment.strip() for segment in segments if segment.strip())


def _dark_stat_card(label: str, value: str, tone: str = "secondary"):
    return VStack(
        gap=0.04,
        children=[
            TextBlock(
                text=label.upper(),
                font_size=9,
                color=tone,
                bold=True,
                line_spacing=1.0,
                height=0.38,
            ),
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
    left_visual = VStack(
        gap=0.16,
        children=[
            TextBlock(text="Black-box RCA", font_size="heading", color="negative", bold=True),
            Flow(
                labels=["Data", "LLM guess", "Label"],
                colors=["bg_alt", "negative", "bg_alt"],
                arrow_color="text_light",
            ),
            Callout(
                title="What the benchmark rewards",
                body="Optimize for a correct answer even if the causal path is hidden or hallucinated.",
                color="negative",
            ),
        ],
    )

    right_visual = VStack(
        gap=0.16,
        children=[
            TextBlock(text="World-model RCA", font_size="heading", color="positive", bold=True),
            Flow(
                labels=["Data", "Verify", "Graph"],
                colors=["bg_alt", "accent", "positive"],
                arrow_color="text_light",
            ),
            Callout(
                title="What we actually want",
                body="Build and verify a causal loop so the answer and the reasoning are both inspectable.",
                color="positive",
            ),
        ],
    )

    sb = prs.evidence_comparison(
        title=SLIDE_TITLES[1],
        left_title="Prediction-oriented framing",
        left_visual=left_visual,
        right_title="Verification-oriented framing",
        right_visual=right_visual,
        conclusion="Black-box RCA predicts an answer; world-model RCA verifies a causal explanation.",
        reference="Goal-driven RCA survey",
    )
    sb.notes(
        _speaker_notes(
            "Contrast today's black-box symptom-to-label flow with a world-model loop that hypothesizes and verifies propagation.",
            "Transition by asking what assumptions must be true for that loop to be feasible.",
        )
    )
    return sb


def _build_slide_3(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[2], reference="OpenRCA 2.0 / FORGE")

    premise = Callout(
        title="Fault detection precedes diagnosis",
        body="Before RCA can begin, we need to know a fault exists. OpenRCA and FORGE show that a bounded set of atomic faults can be systematically injected and observed.",
        color="secondary",
    )
    structure = Flow(
        labels=["Finite faults", "Controlled injection", "Observable effects"],
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
        text="Large incidents are combinations of local faults. The key is bounded atomic faults that can be injected, replayed, and verified.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.24, children=[premise, structure, fault_grid, conclusion]))
    sb.notes(
        _speaker_notes(
            "State the first premise: the intervention space is broad but still bounded enough to model.",
            "OpenRCA and FORGE demonstrate this with 25 fault types across 6 categories.",
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

    old_paradigm = VStack(
        gap=0.16,
        width=5.5,
        children=[
            Badge(text="Point-finding paradigm", color="negative"),
            Flow(
                labels=["Telemetry", "Inference", "Root cause"],
                colors=["bg_alt", "negative", "bg_alt"],
                arrow_color="text_light",
            ),
            TextBlock(
                text="Current methods optimize for identifying the faulty component, ignoring the propagation path.",
                font_size="body",
                color="text",
            ),
            TextBlock(
                text="This is insufficient: engineers need to understand why this component caused the symptoms.",
                font_size="caption",
                color="text_mid",
            ),
        ],
    )

    new_paradigm = VStack(
        gap=0.16,
        width=5.5,
        children=[
            Badge(text="Graph-building paradigm", color="positive"),
            Flow(
                labels=["Telemetry", "Inference", "Causal graph"],
                colors=["bg_alt", "accent", "positive"],
                arrow_color="text_light",
            ),
            Callout(
                title="What changes",
                body="The output is not just a node, but the causal graph explaining how the fault propagated to symptoms.",
                color="positive",
            ),
        ],
    )

    conclusion = TextBlock(
        text="Ideal RCA builds an interpretable causal graph, not just a ranked root-cause label.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(
        VStack(
            gap=0.28,
            children=[
                HStack(gap=0.40, children=[old_paradigm, new_paradigm]),
                conclusion,
            ],
        )
    )
    sb.notes(
        _speaker_notes(
            "Frame the survey's core insight: RCA should output a propagation graph, not just a root-cause node.",
            "This is the north star that exposes why current benchmarks are insufficient.",
            "Transition to the benchmark crisis: if we measure only point accuracy, we miss the real challenge.",
        )
    )
    sb.animate([[old_paradigm], [new_paradigm], [conclusion]])
    return sb


def _build_slide_7(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[9], reference="Goal-driven RCA survey")

    effectiveness_circle = Circle(
        text="Effectiveness", color="primary", width=1.85, height=1.85
    )
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
        text="Better RCA needs richer data, and richer data raises cost. The key is measuring the right thing.",
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
            "The crucial point: we must measure graph-building quality, not just point accuracy, to make this trade-off worthwhile.",
            "Transition to the benchmark crisis: current benchmarks measure the wrong thing.",
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
    sb = prs.evidence_comparison(
        title=SLIDE_TITLES[6],
        left_title="Simple baseline",
        left_visual=simple_baseline,
        right_title="Published SOTA",
        right_visual=sota,
        conclusion="If a simple baseline rivals SOTA, the benchmark is not measuring the hardest part of diagnosis.",
        reference="FSE RCA benchmark study",
    )
    sb.notes(
        _speaker_notes(
            "Use the dataset study's headline result: simple heuristics are surprisingly competitive with SOTA on public benchmarks.",
            "This exposes a fundamental problem: current benchmarks measure point accuracy, not graph-building ability.",
            "Transition by asking what structural defects make those benchmarks so forgiving.",
        )
    )
    return sb


def _build_slide_9(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[7], reference="FSE RCA benchmark study")

    header = TextBlock(
        text="Three structural defects make public benchmarks too forgiving:",
        font_size="body",
        color="text",
    )
    defects = Grid(
        cols=3,
        gap=0.20,
        children=[
            Callout(
                title="Shallow ground truth",
                body="Labels only reach service level. No code-level or propagation-level annotation.",
                color="negative",
            ),
            Callout(
                title="Static workloads",
                body="Fixed request patterns hide concurrency and load-dependent fault propagation.",
                color="negative",
            ),
            Callout(
                title="Point-only metrics",
                body="Top@K accuracy rewards lucky guesses. No evaluation of causal paths.",
                color="negative",
            ),
        ],
    )
    consequence = Callout(
        title="Consequence",
        body="Simple heuristics exploit these shortcuts. Reported SOTA progress is inflated by benchmark design, not real diagnostic capability.",
        color="warning",
    )
    bridge = TextBlock(
        text="A harder benchmark must fix all three: deeper labels, dynamic workloads, and propagation-aware evaluation.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.24, children=[header, defects, consequence, bridge]))
    sb.notes(
        _speaker_notes(
            "Drill into the three structural defects behind the benchmark crisis: shallow labels, static workloads, and point-only metrics.",
            "Transition by previewing what a harder benchmark looks like.",
        )
    )
    sb.animate([[header], [defects], [consequence, bridge]])
    return sb


def _build_slide_10(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[8], reference="FSE RCA benchmark study")

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
            stat_card("Average Top@1", "0.21", tone="negative", width=1.55, value_font_size=18),
            stat_card("Best model", "0.37", tone="warning", width=1.55, value_font_size=18),
            Callout(
                title="What changed",
                body="A harder benchmark preserves propagation and exposes a real performance collapse.",
                color="primary",
                width=7.9,
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


def _build_slide_11(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[10], reference="OpenRCA 2.0 / FORGE")

    outcome = VStack(
        gap=0.12,
        width=4.8,
        children=[
            Badge(text="Outcome correctness", color="secondary"),
            TextBlock(
                text="0.76",
                font_size=36,
                color="secondary",
                bold=True,
                align="center",
            ),
            TextBlock(
                text="Pass@1",
                font_size="caption",
                color="text_mid",
                align="center",
            ),
            TextBlock(
                text="Right answer, but not necessarily right reasoning.",
                font_size="caption",
                color="text",
                align="center",
            ),
        ],
    )
    process = VStack(
        gap=0.12,
        width=4.8,
        children=[
            Badge(text="Process correctness", color="warning"),
            TextBlock(
                text="0.63",
                font_size=36,
                color="warning",
                bold=True,
                align="center",
            ),
            TextBlock(
                text="Path Reachability",
                font_size="caption",
                color="text_mid",
                align="center",
            ),
            TextBlock(
                text="Valid causal path from root cause to symptom.",
                font_size="caption",
                color="text",
                align="center",
            ),
        ],
    )
    gap = Callout(
        title="The gap: 13% of correct answers have broken reasoning",
        body="Outcome correctness flatters the agent. Process correctness tells us whether the diagnosis can be trusted.",
        color="negative",
    )

    sb.layout(
        VStack(
            gap=0.32,
            children=[HStack(gap=0.60, children=[outcome, process]), gap],
        )
    )
    sb.notes(
        _speaker_notes(
            "Use this slide to separate outcome correctness from process correctness with the 0.76 versus 0.63 comparison.",
            "The key insight: 13% of answers are 'lucky guesses' with invalid reasoning paths.",
            "Transition by asking how to turn an ill-posed backward RCA task into something verifiable enough to supervise.",
        )
    )
    sb.animate([[outcome], [process], [gap]])
    return sb


def _build_slide_12(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[11], reference="OpenRCA 2.0 / FORGE")

    backward = VStack(
        gap=0.18,
        width=4.8,
        children=[
            Badge(text="Backward diagnosis", color="negative"),
            Flow(
                labels=["Symptoms", "??? ", "Ill-posed"],
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
        width=4.8,
        children=[
            Badge(text="Forward verification", color="positive"),
            Flow(
                labels=["Known fault", "Propagate", "Verified"],
                colors=["accent", "secondary", "positive"],
                arrow_color="text_light",
            ),
            Callout(
                title="Why FORGE works",
                body="Known intervention makes verification tractable: trace cause to effect.",
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

    sb.layout(
        VStack(
            gap=0.28,
            children=[HStack(gap=0.50, children=[backward, forward]), asymmetry],
        )
    )
    sb.notes(
        _speaker_notes(
            "Explain the information asymmetry at the heart of FORGE: the agent reasons backward, while annotation can verify forward from the known intervention.",
            "Transition by turning that verification mechanism into a concrete benchmark identity.",
        )
    )
    sb.animate([[backward], [forward], [asymmetry]])
    return sb


def _build_slide_13(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[12], reference="OpenRCA 2.0 / FORGE")

    summary = HStack(
        gap=0.24,
        children=[
            stat_card("Benchmark", "500 cases", tone="primary", width=3.1, value_font_size=18),
            stat_card("Labels", "Step-wise annotations", tone="accent", width=3.75, value_font_size=18),
            stat_card("Supervision", "Process supervision", tone="positive", width=3.6, value_font_size=18),
        ],
    )
    dimensions = Flow(
        labels=[
            "Known intervention",
            "Verified propagation",
            "Agent graph",
            "Process metrics",
        ],
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


def _build_slide_14(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[13], reference="TRIANGLE paper / Incident triage")

    context = Callout(
        title="From diagnosis to triage",
        body="RCA diagnosis is one step in the incident lifecycle. TRIANGLE tackles the upstream step—triage—and shows that structured multi-agent reasoning outperforms monolithic approaches in production.",
        color="secondary",
    )
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
        height=1.00,
    )
    stats = HStack(
        gap=0.40,
        width=3.9,
        children=[
            stat_card("Accuracy", "97%", tone="positive"),
            stat_card("TTE drop", "91%", tone="accent"),
        ],
    )
    production = Callout(
        title="Production validated",
        body="Deployed at Microsoft for large-scale production triage.",
        color="primary",
        width=7.4,
    )
    outcomes = HStack(
        gap=0.32,
        children=[stats, production],
    )

    sb.layout(VStack(gap=0.24, children=[context, framework, outcomes]))
    sb.notes(
        _speaker_notes(
            "Position TRIANGLE as independent validation of the same research belief: explicit reasoning structures beat black-box approaches.",
            "The key evidence is production deployment at scale—structured multi-agent triage works in practice.",
            "Transition: if structured reasoning works for triage and RCA evaluation, can we train agents to build reusable world models?",
        )
    )
    sb.animate([[context], [framework], [outcomes]])
    return sb


def _build_slide_15(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[14])

    bridge = Callout(
        title="From evaluation to training",
        body="OpenRCA 2.0 gives us process supervision for evaluation. But where does the training data come from? Manual annotation is too expensive.",
        color="secondary",
    )
    loop = Flow(
        labels=["Simulation", "Hypothesis", "Verify", "Reward"],
        colors=["accent", "secondary", "positive", "warning"],
        arrow_color="text_light",
    )
    training = Callout(
        title="Stage 1: Train on closed loops",
        body="Use fault injection to generate unlimited training cases. The agent proposes hypotheses, verifies against ground truth, and learns from the full loop—not just answer labels.",
        color="primary",
    )

    sb.layout(VStack(gap=0.28, children=[bridge, loop, training]))
    sb.notes(
        _speaker_notes(
            "Bridge from OpenRCA 2.0's evaluation to the training challenge: we need automated training data generation.",
            "The solution is closed-loop training with fault injection simulation—leveraging the same verification mechanism as FORGE.",
            "Transition: after collecting many successful traces, what can we do with them?",
        )
    )
    sb.animate([[bridge], [loop], [training]])
    return sb


def _build_slide_16(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[15])

    compression = Flow(
        labels=["Many traces", "Distill", "World model"],
        colors=["secondary", "accent", "primary"],
        arrow_color="text_light",
    )
    sources = VStack(
        gap=0.14,
        width=3.4,
        children=[
            Badge(text="Training traces", color="secondary"),
            RoundedBox(text="Trace A: CPU → Latency", color="bg_alt", height=0.60),
            RoundedBox(text="Trace B: Memory → OOM", color="bg_alt", height=0.60),
            RoundedBox(text="Trace C: Network → Timeout", color="bg_alt", height=0.60),
        ],
    )
    model = VStack(
        gap=0.16,
        width=4.2,
        children=[
            Badge(text="Stage 2: Reusable world model", color="primary"),
            Circle(text="world\nmodel", color="primary", width=2.15, height=2.15),
            TextBlock(
                text="Compress many traces into a compact causal representation. Preserve propagation structure, drop incident-specific noise.",
                font_size="body",
                color="text",
            ),
        ],
    )

    sb.layout(
        VStack(
            gap=0.24,
            children=[compression, HStack(gap=0.55, children=[sources, model])],
        )
    )
    sb.notes(
        _speaker_notes(
            "Stage two is distillation: compress many training traces into one reusable world model.",
            "The key challenge: preserve causal structure while generalizing across different incidents.",
            "Transition: once we have the world model, how do we maximize its value?",
        )
    )
    sb.animate([[compression], [sources], [model]])
    return sb


def _build_slide_17(prs: Presentation):
    sb = prs.slide(title=SLIDE_TITLES[16])

    leverage = Callout(
        title="Stage 3: Maximize leverage",
        body="Build the world model once, then apply it across the entire incident management lifecycle.",
        color="primary",
    )
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
    conclusion = TextBlock(
        text="The world model's value is reuse: one causal representation supports diagnosis, prediction, and repair across different operational contexts.",
        font_size="body",
        color="text_mid",
        italic=True,
    )

    sb.layout(VStack(gap=0.28, children=[leverage, framework, conclusion]))
    sb.notes(
        _speaker_notes(
            "Stage three is about leverage: the same world model enables multiple downstream tasks.",
            "This is the payoff of the three-stage roadmap: simulate, compress, then generalize.",
            "Transition to the closing takeaways.",
        )
    )
    sb.animate([[leverage], [framework], [conclusion]])
    return sb


def _build_slide_18(prs: Presentation):
    sb = prs.seminar_takeaway(
        title=SLIDE_TITLES[17],
        message="What matters is not only finding the right answer, but building a reasoning system we can trust and reuse.",
        evidence=[
            "Score the reasoning path, not only the final answer.",
            "Train on verification loops instead of answer labels alone.",
            "Distill reusable world models from many verified traces.",
        ],
    )
    sb.notes(
        _speaker_notes(
            "Close with three short takeaways rather than a dense recap, keeping the typography strong and the message memorable.",
            "End on the final line as the thesis statement for the whole talk.",
        )
    )
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
        _build_slide_8,
        _build_slide_9,
        _build_slide_10,
        _build_slide_7,
        _build_slide_11,
        _build_slide_12,
        _build_slide_13,
        _build_slide_14,
        _build_slide_15,
        _build_slide_16,
        _build_slide_17,
        _build_slide_18,
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
