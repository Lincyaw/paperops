"""
Presentation for FSE'26 Paper:
"Rethinking the Evaluation of Microservice RCA with a Fault Propagation-Aware Benchmark"

Target audience: Cross-domain researchers (not necessarily SE/ops experts)
Duration: ~10 minutes
"""

from pathlib import Path

from paperops.slides import (
    Arrow,
    Presentation, themes,
    HStack, VStack, Grid, Padding,
    RoundedBox, Circle,
    TextBlock, BulletList, Table,
    Image, SvgImage,
)

# Create presentation with academic theme
prs = Presentation(theme=themes.academic)
ASSET_DIR = Path(__file__).parent / "assets"


def asset(filename: str) -> str:
    """Resolve local image assets for this deck."""
    return str(ASSET_DIR / filename)


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


def Flowchart(nodes: dict, edges: list, direction: str = "right", node_widths: dict[str, float] | None = None, height: float | None = None):
    node_widths = node_widths or {}
    order = list(nodes.keys())
    boxes = {}
    for node_id in order:
        value = nodes[node_id]
        if isinstance(value, tuple):
            label, color = value[0], value[1]
        else:
            label, color = value, "bg_alt"
        boxes[node_id] = RoundedBox(
            text=label,
            color=color,
            border="border" if color in {"bg_alt", "bg", "bg_accent"} else color,
            text_color="white" if color not in {"bg_alt", "bg", "bg_accent"} else "text",
            font_size="caption",
            bold=True,
            width=node_widths.get(node_id),
            height=height if height is not None else 0.9,
            size_mode_x="fit" if node_widths.get(node_id) is None else "fixed",
        )
    out_edges = {}
    in_degree = {}
    for edge in edges:
        out_edges.setdefault(edge[0], []).append(edge[1])
        in_degree[edge[1]] = in_degree.get(edge[1], 0) + 1
    is_chain = all(len(targets) <= 1 for targets in out_edges.values()) and all(count <= 1 for count in in_degree.values())
    if not is_chain:
        return Grid(cols=min(max(len(order), 1), 3), gap=0.18, children=[boxes[node_id] for node_id in order])
    starts = [node_id for node_id in order if in_degree.get(node_id, 0) == 0]
    if starts:
        ordered = []
        current = starts[0]
        seen = set()
        while current not in seen:
            seen.add(current)
            ordered.append(current)
            next_nodes = out_edges.get(current, [])
            if not next_nodes:
                break
            current = next_nodes[0]
        ordered.extend(node_id for node_id in order if node_id not in seen)
        order = ordered
    children = []
    arrow_direction = "vertical" if direction in {"down", "vertical"} else "horizontal"
    for index, node_id in enumerate(order):
        box = boxes[node_id]
        children.append(box)
        if index < len(order) - 1:
            children.append(
                Arrow(
                    from_component=box,
                    to_component=boxes[order[index + 1]],
                    color="primary",
                    direction=arrow_direction,
                )
            )
    container_cls = VStack if arrow_direction == "vertical" else HStack
    return container_cls(gap=0.15, children=children)


def BarChart(groups: list, y_label: str = "", show_values: bool = True):
    width = 700
    height = 360
    margin_left = 70
    margin_bottom = 60
    chart_height = 220
    chart_top = 30
    colors = {
        "primary": "#8B4513",
        "accent": "#4A6FA5",
        "negative": "#A0522D",
        "warning": "#B8860B",
    }
    flat_values = [item[1] for _group, items in groups for item in items]
    max_value = max(flat_values) if flat_values else 1.0
    group_width = 180
    bar_width = 34
    gap = 14
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFEF8"/>',
        f'<text x="22" y="{chart_top + chart_height / 2}" font-size="16" fill="#5A5A5A" transform="rotate(-90 22 {chart_top + chart_height / 2})">{y_label}</text>',
        f'<line x1="{margin_left}" y1="{chart_top + chart_height}" x2="{width - 20}" y2="{chart_top + chart_height}" stroke="#C8C0B0" stroke-width="2"/>',
    ]
    for group_index, (group_label, items) in enumerate(groups):
        group_x = margin_left + 30 + group_index * group_width
        for item_index, (label, value, tone) in enumerate(items):
            bar_h = 0 if max_value <= 0 else (value / max_value) * chart_height
            x = group_x + item_index * (bar_width + gap)
            y = chart_top + chart_height - bar_h
            fill = colors.get(tone, "#8B4513")
            svg.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" rx="4" fill="{fill}" opacity="0.85"/>')
            if show_values:
                svg.append(f'<text x="{x + bar_width / 2}" y="{y - 8}" font-size="13" text-anchor="middle" fill="#2C2C2C">{value:.2f}</text>')
            svg.append(f'<text x="{x + bar_width / 2}" y="{chart_top + chart_height + 18}" font-size="12" text-anchor="middle" fill="#5A5A5A">{label}</text>')
        center_x = group_x + ((len(items) - 1) * (bar_width + gap) + bar_width) / 2
        svg.append(f'<text x="{center_x}" y="{height - 16}" font-size="14" text-anchor="middle" fill="#2C2C2C">{group_label}</text>')
    svg.append("</svg>")
    return SvgImage(svg="".join(svg), width=6.7, height=3.2)

# ============================================================================
# SVG ICONS (raw SVG strings)
# ============================================================================

def svg_cloud(color="#4A90E2"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <circle cx="25" cy="35" r="18" fill="{color}" opacity="0.8"/>
        <circle cx="45" cy="35" r="15" fill="{color}" opacity="0.8"/>
        <circle cx="20" cy="40" r="12" fill="{color}" opacity="0.8"/>
        <rect x="20" y="35" width="30" height="15" fill="{color}" opacity="0.8"/>
    </svg>'''

def svg_server(color="#7B68EE"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <rect x="15" y="10" width="30" height="40" rx="3" fill="{color}" stroke="#333" stroke-width="1"/>
        <circle cx="22" cy="20" r="3" fill="#4CAF50"/>
        <circle cx="22" cy="30" r="3" fill="#4CAF50"/>
        <circle cx="22" cy="40" r="3" fill="#FFC107"/>
        <line x1="45" y1="20" x2="55" y2="20" stroke="#999" stroke-width="1"/>
        <line x1="45" y1="30" x2="55" y2="30" stroke="#999" stroke-width="1"/>
    </svg>'''

def svg_ai_brain(color="#9C27B0"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <circle cx="30" cy="30" r="22" fill="{color}" opacity="0.3" stroke="{color}" stroke-width="2"/>
        <circle cx="22" cy="25" r="4" fill="{color}"/>
        <circle cx="38" cy="25" r="4" fill="{color}"/>
        <circle cx="30" cy="35" r="4" fill="{color}"/>
        <circle cx="25" cy="38" r="3" fill="{color}"/>
        <circle cx="35" cy="38" r="3" fill="{color}"/>
        <line x1="22" y1="25" x2="30" y2="35" stroke="{color}" stroke-width="1.5"/>
        <line x1="38" y1="25" x2="30" y2="35" stroke="{color}" stroke-width="1.5"/>
        <line x1="30" y1="35" x2="25" y2="38" stroke="{color}" stroke-width="1"/>
        <line x1="30" y1="35" x2="35" y2="38" stroke="{color}" stroke-width="1"/>
    </svg>'''

def svg_warning(color="#FF9800"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <polygon points="30,8 52,45 8,45" fill="{color}" opacity="0.9"/>
        <line x1="30" y1="20" x2="30" y2="32" stroke="white" stroke-width="3"/>
        <circle cx="30" cy="38" r="2.5" fill="white"/>
    </svg>'''

def svg_chart_down(color="#F44336"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <rect x="8" y="8" width="44" height="44" rx="3" fill="#f5f5f5" stroke="#ddd"/>
        <line x1="15" y1="20" x2="25" y2="28" stroke="{color}" stroke-width="2.5"/>
        <line x1="25" y1="28" x2="35" y2="35" stroke="{color}" stroke-width="2.5"/>
        <line x1="35" y1="35" x2="45" y2="42" stroke="{color}" stroke-width="2.5"/>
        <line x1="40" y1="38" x2="45" y2="42" stroke="{color}" stroke-width="2"/>
        <line x1="45" y1="38" x2="45" y2="42" stroke="{color}" stroke-width="2"/>
    </svg>'''

def svg_search(color="#4A90E2"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <circle cx="28" cy="28" r="16" fill="none" stroke="{color}" stroke-width="3"/>
        <line x1="40" y1="40" x2="50" y2="50" stroke="{color}" stroke-width="4"/>
    </svg>'''

def svg_tools(color="#4CAF50"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <line x1="15" y1="45" x2="45" y2="15" stroke="{color}" stroke-width="4"/>
        <circle cx="18" cy="42" r="5" fill="none" stroke="{color}" stroke-width="3"/>
        <line x1="42" y1="12" x2="48" y2="18" stroke="{color}" stroke-width="3"/>
    </svg>'''

def svg_data(color="#7B68EE"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <ellipse cx="30" cy="18" rx="20" ry="8" fill="{color}" stroke="#333" stroke-width="1.5"/>
        <rect x="10" y="18" width="40" height="28" fill="{color}" stroke="#333" stroke-width="1.5"/>
        <ellipse cx="30" cy="46" rx="20" ry="8" fill="{color}" stroke="#333" stroke-width="1.5"/>
        <line x1="10" y1="18" x2="10" y2="46" stroke="#333" stroke-width="1.5"/>
        <line x1="50" y1="18" x2="50" y2="46" stroke="#333" stroke-width="1.5"/>
    </svg>'''

def svg_speed(color="#F44336"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <path d="M 8 35 Q 8 15 30 8 Q 52 15 52 35" fill="none" stroke="#333" stroke-width="2.5"/>
        <line x1="30" y1="35" x2="18" y2="45" stroke="{color}" stroke-width="2.5"/>
        <circle cx="30" cy="35" r="4" fill="#333"/>
    </svg>'''

def svg_eye(color="#4A90E2"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <ellipse cx="30" cy="30" rx="22" ry="14" fill="#f5f5f5" stroke="#333" stroke-width="2"/>
        <circle cx="30" cy="30" r="8" fill="{color}"/>
        <circle cx="30" cy="30" r="3" fill="white"/>
    </svg>'''

def svg_puzzle(color="#9C27B0"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <rect x="15" y="15" width="30" height="30" rx="4" fill="{color}" opacity="0.3" stroke="{color}" stroke-width="2"/>
        <circle cx="30" cy="15" r="5" fill="{color}"/>
        <circle cx="45" cy="30" r="5" fill="{color}"/>
    </svg>'''

def svg_github(color="#333"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <polyline points="20,20 12,30 20,40" fill="none" stroke="{color}" stroke-width="2.5"/>
        <polyline points="40,20 48,30 40,40" fill="none" stroke="{color}" stroke-width="2.5"/>
        <line x1="25" y1="42" x2="35" y2="18" stroke="#666" stroke-width="2"/>
    </svg>'''

def svg_metrics(color="#2196F3"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <rect x="5" y="5" width="50" height="50" fill="none" stroke="#333" stroke-width="2"/>
        <line x1="10" y1="45" x2="20" y2="35" stroke="{color}" stroke-width="2"/>
        <line x1="20" y1="35" x2="30" y2="40" stroke="{color}" stroke-width="2"/>
        <line x1="30" y1="40" x2="40" y2="25" stroke="{color}" stroke-width="2"/>
        <line x1="40" y1="25" x2="50" y2="30" stroke="{color}" stroke-width="2"/>
    </svg>'''

def svg_logs(color="#FF9800"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <rect x="10" y="8" width="40" height="44" fill="none" stroke="#333" stroke-width="2"/>
        <line x1="16" y1="18" x2="44" y2="18" stroke="{color}" stroke-width="2"/>
        <line x1="16" y1="26" x2="38" y2="26" stroke="#666" stroke-width="2"/>
        <line x1="16" y1="34" x2="42" y2="34" stroke="#666" stroke-width="2"/>
        <line x1="16" y1="42" x2="35" y2="42" stroke="#666" stroke-width="2"/>
    </svg>'''

def svg_trace(color="#4CAF50"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
        <circle cx="15" cy="30" r="6" fill="{color}"/>
        <circle cx="30" cy="20" r="6" fill="{color}"/>
        <circle cx="30" cy="40" r="6" fill="{color}"/>
        <circle cx="45" cy="30" r="6" fill="{color}"/>
        <line x1="21" y1="28" x2="24" y2="22" stroke="{color}" stroke-width="2"/>
        <line x1="21" y1="32" x2="24" y2="38" stroke="{color}" stroke-width="2"/>
        <line x1="36" y1="22" x2="39" y2="28" stroke="{color}" stroke-width="2"/>
        <line x1="36" y1="38" x2="39" y2="32" stroke="{color}" stroke-width="2"/>
    </svg>'''

# ============================================================================
# SLIDE 1: TITLE
# ============================================================================

sb = prs.slide(background="#0B1220")

hero = HStack(gap=0.35, children=[
    VStack(gap=0.28, width=6.5, children=[
        RoundedBox(
            text="FSE 2026",
            color="#1F3B63",
            border="#2D5B94",
            text_color="#E5EDF8",
            font_size="caption",
            bold=True,
            width=1.45,
            height=0.42,
        ),
        TextBlock(
            text="Rethinking the Evaluation\nof Microservice RCA",
            font_size=37,
            bold=True,
            color="#F4F7FB",
            height=1.85,
        ),
        TextBlock(
            text="with a Fault Propagation-Aware Benchmark",
            font_size=22,
            bold=False,
            color="#B7C7DD",
            height=0.55,
        ),
        TextBlock(
            text="Aoyang Fang, Songhan Zhang, Yifan Yang, Haotong Wu, Junjielong Xu,\n"
                 "Xuyang Wang, Rui Wang, Manyi Wang, Qisheng Lu, Pinjia He",
            font_size="caption",
            color="#D0DBEA",
            height=0.75,
        ),
        TextBlock(
            text="The Chinese University of Hong Kong, Shenzhen",
            font_size="caption",
            color="#8EA2BE",
        ),
    ]),
    VStack(gap=0.2, width=4.1, children=[
        Image(path=asset("wm-data-center-unc.jpg"), width=4.1, height=4.55),
        TextBlock(text="Fault propagation in large-scale service systems", font_size="small", color="#8EA2BE"),
    ]),
])

sb.layout(Padding(child=hero, all=0.25))
sb.animate([[hero.children[0]], [hero.children[1]]])
sb.notes(
    "大家好，今天我要汇报的工作是重新思考微服务 RCA，也就是根因分析任务的评测方式。\n"
    "这篇工作的核心观点很简单：如果 benchmark 设计得过于简单，那么模型表现再好，也不一定代表它真的具备了处理复杂故障传播的能力。\n"
    "我们提出了一个 fault propagation-aware 的 benchmark，用来更真实地评估 RCA 方法在复杂微服务系统中的表现。\n"
    "接下来我会先讲问题背景，再讲我们发现了什么，最后讲我们如何重新构建评测基准。"
)

# ============================================================================
# SLIDE 2: MOTIVATION
# ============================================================================

sb = prs.slide(title="Motivation: When Systems Fail")

# Real world cases
cases = VStack(gap=0.3, children=[
    TextBlock(text="Real-world incidents:", font_size="heading", bold=True),
    BulletList(items=[
        "2024 CrowdStrike: 8.5M Windows devices crashed",
        "2023 Cloud outages: millions of users affected",
        "Enterprise downtime cost: $23,000 per minute",
    ]),
])

# RCA definition
rca_def = VStack(gap=0.3, children=[
    TextBlock(text="Root Cause Analysis (RCA) Task:", font_size="heading", bold=True),
    TextBlock(text="Input: Telemetry data when system fails", font_size="body"),
    TextBlock(text="Output: Ranked list of suspected root cause services", font_size="body"),
    TextBlock(text="Evaluation: Top@K (is root cause in top K predictions?)", font_size="body"),
])

# Three modalities
modalities = VStack(gap=0.2, children=[
    TextBlock(text="Three Data Modalities:", font_size="heading", bold=True),
    HStack(gap=0.3, children=[
        VStack(gap=0.1, children=[
            SvgImage(svg=svg_metrics(), width=0.5, height=0.5),
            TextBlock(text="Metrics", font_size="caption"),
            TextBlock(text="(CPU, memory, latency)", font_size="small", color="text_mid"),
        ]),
        VStack(gap=0.1, children=[
            SvgImage(svg=svg_logs(), width=0.5, height=0.5),
            TextBlock(text="Logs", font_size="caption"),
            TextBlock(text="(error messages)", font_size="small", color="text_mid"),
        ]),
        VStack(gap=0.1, children=[
            SvgImage(svg=svg_trace(), width=0.5, height=0.5),
            TextBlock(text="Traces", font_size="caption"),
            TextBlock(text="(call chains)", font_size="small", color="text_mid"),
        ]),
    ]),
])

incident_visual = VStack(gap=0.2, width=4.4, children=[
    Image(path=asset("server-room.jpg"), width=4.4, height=2.3),
    TextBlock(
        text="Operational context: distributed cloud services under failure pressure",
        font_size="small",
        color="text_mid",
    ),
    modalities,
])

sb.layout(VStack(gap=0.35, children=[
    Padding(child=HStack(gap=0.4, children=[
        VStack(gap=0.35, width=6.2, children=[cases, rca_def]),
        incident_visual,
    ]), left=0.2, right=0.2),
]))

sb.animate([[cases], [rca_def], [incident_visual]])
sb.notes(
    "先从动机开始。现实里的系统故障代价非常高，比如 CrowdStrike 事件、云服务中断、以及企业级停机成本，都说明 RCA 是一个非常重要的问题。\n"
    "RCA 任务本身可以理解为：系统出问题之后，我们拿到 telemetry 数据，最后输出一个按可疑程度排序的服务列表。\n"
    "评测时通常会看 Top@K，也就是根因服务是否出现在前 K 个预测里。\n"
    "这里的数据通常来自三类模态：metrics、logs 和 traces。后面你会看到，很多方法宣称做了多模态融合，但 benchmark 本身未必足够有挑战。"
)

# ============================================================================
# SLIDE 3: EXISTING APPROACHES
# ============================================================================

sb = prs.slide(title="The Promise: AI for RCA")

# Evolution
evolution = VStack(gap=0.3, children=[
    TextBlock(text="Evolution of RCA Methods:", font_size="heading", bold=True),
    BulletList(items=[
        "Early: Single-modality (metrics only, logs only, or traces only)",
        "Recent: Multi-modal fusion (GNN, Causal Inference, Deep Learning)",
    ]),
])

# Reported performance
performance = VStack(gap=0.3, children=[
    TextBlock(text="Reported Performance:", font_size="heading", bold=True),
    HStack(gap=0.3, children=[
        RoundedBox(text="80-95%", color="positive", text_color="white", bold=True),
        TextBlock(text="Top@1 accuracy", font_size="body"),
    ]),
    HStack(gap=0.3, children=[
        RoundedBox(text="0.85+", color="positive", text_color="white", bold=True),
        TextBlock(text="MRR (Mean Reciprocal Rank)", font_size="body"),
    ]),
])

trend_visual = Image(path=asset("wm-control-room.jpg"), height=0.6)

# The question
question = Callout(
    title="But...",
    body="Different papers use different datasets.\nAre the scores really comparable?",
    color="warning"
)

sb.layout(VStack(gap=0.12, children=[
    Padding(child=evolution, left=0.2, right=0.2),
    Padding(child=trend_visual, left=0.2, right=0.2),
    Padding(child=performance, left=0.2, right=0.2),
    Padding(child=question, left=0.3, right=0.3),
]))

sb.animate([[evolution], [trend_visual], [performance], [question]])
sb.notes(
    "过去 RCA 方法的发展路径大致是从单模态走向多模态，从规则方法走向图模型、因果推断和深度学习。\n"
    "从论文结果看，很多工作都报告了非常高的 Top@1 和 MRR，表面上看这个问题似乎已经被做得很好了。\n"
    "但这里有一个关键问题：这些数字真的可靠吗？\n"
    "因为不同论文通常使用不同数据集、不同故障注入方式、不同系统规模，所以我们怀疑这些高分有可能来自 benchmark 过于简单，而不是真正的能力提升。"
)

# ============================================================================
# SLIDE 4: SIMPLERCA
# ============================================================================

sb = prs.slide(title="Our Question: Are Benchmarks Reliable?")

# Core question
core_q = VStack(gap=0.3, children=[
    Callout(
        title="Core Question",
        body="Is SOTA progress real capability?\nOr an artifact of simplistic benchmarks?",
        color="primary"
    ),
])

# SimpleRCA design
simplerca = VStack(gap=0.3, children=[
    TextBlock(text="SimpleRCA: A Rule-Based Baseline", font_size="heading", bold=True),
    TextBlock(text="Zero machine learning, just three intuitive rules:", font_size="body"),
    BulletList(items=[
        "Metrics: 3-sigma threshold detection",
        "Traces: P95 latency > 3x baseline",
        "Logs: Count ERROR/FAIL keywords",
    ]),
    TextBlock(text="Root cause = service with most alerts", font_size="body", italic=True, color="text_mid"),
])

baseline_visual = VStack(gap=0.18, width=4.6, children=[
    Image(path=asset("wm-load-balancing.png"), width=4.6, height=2.4),
    TextBlock(text="Simple dependency-aware heuristics can already localize many failures", font_size="small", color="text_mid"),
])

sb.layout(VStack(gap=0.38, children=[
    Padding(child=core_q, left=0.3, right=0.3),
    Padding(child=HStack(gap=0.35, children=[
        VStack(gap=0.25, width=6.15, children=[simplerca]),
        baseline_visual,
    ]), left=0.2, right=0.2),
]))

sb.animate([[core_q], [simplerca], [baseline_visual]])
sb.notes(
    "所以我们提出的核心问题是：所谓 SOTA 进展，到底是真实能力，还是 benchmark artifact？\n"
    "为了回答这个问题，我们先设计了一个非常简单的 baseline，叫 SimpleRCA。\n"
    "它完全不用机器学习，只做三件事：在 metrics 里看异常阈值，在 traces 里看延迟偏移，在 logs 里统计错误关键词。\n"
    "最后谁的异常信号最多，就把谁排到前面。\n"
    "如果这样一个简单规则都能和复杂模型打平甚至超过它们，就说明现有 benchmark 很可能没有真正区分简单相关性和复杂推理能力。"
)

# ============================================================================
# SLIDE 5: SHOCKING DISCOVERY
# ============================================================================

sb = prs.slide(title="Shocking Discovery")

# Results table
results_table = Table(
    headers=["Dataset", "SOTA Best", "SimpleRCA"],
    rows=[
        ["RE2-TT", "0.67", "0.80"],
        ["RE3-TT", "0.50", "0.83"],
        ["Nezha-TT", "0.87", "0.93"],
        ["Eadro-TT", "0.99", "0.81"],
    ],
    header_color="primary"
)

# Conclusion
conclusion = Callout(
    title="Conclusion",
    body="Simple rules match or beat complex AI!\nExisting benchmarks cannot distinguish simple from complex methods.",
    color="negative"
)

results_visual = VStack(gap=0.18, width=3.9, children=[
    Image(path=asset("wm-analytics-dashboard.jpg"), width=3.9, height=2.25),
    TextBlock(text="Reported dashboards can hide benchmark bias", font_size="small", color="text_mid"),
])

sb.layout(VStack(gap=0.42, children=[
    Padding(child=HStack(gap=0.38, children=[
        Padding(child=results_table, left=0.0, right=0.0, width=6.85),
        results_visual,
    ]), left=0.3, right=0.3),
    Padding(child=conclusion, left=0.3, right=0.3),
]))

sb.animate([[results_table], [results_visual], [conclusion]])
sb.notes(
    "结果非常直接，也有些出人意料。\n"
    "在多个已有 benchmark 上，SimpleRCA 不仅没有明显落后，反而经常能够追平甚至超过复杂方法。\n"
    "这说明一个严重问题：这些 benchmark 不能有效区分简单启发式和复杂 AI 模型。\n"
    "也就是说，模型高分不一定是因为理解了故障传播，而可能只是抓住了很容易的表面相关性。"
)

# ============================================================================
# SLIDE 6: ROOT CAUSE ANALYSIS
# ============================================================================

sb = prs.slide(title="Why? Analysis of Existing Benchmarks")

# Three limitations
limitations = VStack(gap=0.3, children=[
    TextBlock(text="Three Key Limitations:", font_size="heading", bold=True),
    Grid(cols=3, gap=0.3, children=[
        VStack(gap=0.1, children=[
            SvgImage(svg=svg_data(), width=0.6, height=0.6),
            TextBlock(text="Limited Cases", font_size="body", bold=True),
            TextBlock(text="<200 faults\n(easy to overfit)", font_size="caption", color="text_mid"),
        ]),
        VStack(gap=0.1, children=[
            SvgImage(svg=svg_server(), width=0.6, height=0.6),
            TextBlock(text="Simple Structure", font_size="body", bold=True),
            TextBlock(text="2-3 call depth\n(real: 7+)", font_size="caption", color="text_mid"),
        ]),
        VStack(gap=0.1, children=[
            SvgImage(svg=svg_chart_down(), width=0.6, height=0.6),
            TextBlock(text="Narrow Spectrum", font_size="body", bold=True),
            TextBlock(text="1-2 fault types\n(should be 6+)", font_size="caption", color="text_mid"),
        ]),
    ]),
])

# Fault pattern
pattern = VStack(gap=0.2, children=[
    TextBlock(text="Fault Injection Pattern:", font_size="heading", bold=True),
    TextBlock(text="86% are Type I: symptoms only in injected service", font_size="body"),
    TextBlock(text="Simple correlation works → no need for complex causal reasoning", font_size="body", italic=True, color="text_mid"),
])

complexity_visual = Image(path=asset("network-cables.jpg"), height=0.78)

sb.layout(VStack(gap=0.18, children=[
    Padding(child=limitations, left=0.2, right=0.2),
    Padding(child=complexity_visual, left=0.2, right=0.2),
    Padding(child=pattern, left=0.2, right=0.2),
]))

sb.animate([[limitations.children[0]], [limitations.children[1]], [complexity_visual], [pattern]])
sb.notes(
    "接下来我们分析为什么会这样。\n"
    "现有 benchmark 主要有三类问题：第一，样本量太小，容易过拟合；第二，系统结构过浅，调用深度不够；第三，故障谱过窄，覆盖不了真实复杂场景。\n"
    "更关键的是，很多故障注入模式其实属于一种非常简单的情况：症状主要出现在被注入的那个服务本身。\n"
    "在这种设定下，只要做局部相关性匹配就够了，不需要真正理解跨服务的故障传播。\n"
    "这正是我们认为 benchmark 失真的根本原因。"
)

# ============================================================================
# SLIDE 7: OUR SOLUTION
# ============================================================================

sb = prs.slide(title="Our Solution: A Better Benchmark")

# Key numbers
numbers = Grid(cols=4, gap=0.3, children=[
    VStack(gap=0.1, children=[
        Circle(text="1,430", color="positive", text_color="white", radius=0.35),
        TextBlock(text="validated cases", font_size="caption"),
    ]),
    VStack(gap=0.1, children=[
        Circle(text="25", color="primary", text_color="white", radius=0.35),
        TextBlock(text="fault types", font_size="caption"),
    ]),
    VStack(gap=0.1, children=[
        Circle(text="50", color="secondary", text_color="white", radius=0.35),
        TextBlock(text="microservices", font_size="caption"),
    ]),
    VStack(gap=0.1, children=[
        Circle(text="7", color="accent", text_color="white", radius=0.35),
        TextBlock(text="call depth", font_size="caption"),
    ]),
])

# Innovation
innovation = Callout(
    title="Core Innovation: Impact-Driven Validation",
    body="84.4% of injected faults are 'silent' (no user impact)\nWe filter via SLIs (success rate, latency)\nOnly keep operationally relevant cases",
    color="accent"
)

benchmark_visual = VStack(gap=0.16, children=[
    Image(path=asset("data-center-wide.jpg"), height=1.45),
    TextBlock(text="Larger and noisier environments demand stronger RCA reasoning", font_size="small", color="text_mid"),
])

sb.layout(VStack(gap=0.28, children=[
    Padding(child=benchmark_visual, left=0.2, right=0.2),
    Padding(child=numbers, left=0.2, right=0.2),
    Padding(child=innovation, left=0.2, right=0.2),
]))

sb.animate([[benchmark_visual], [numbers], [innovation]])
sb.notes(
    "所以我们的解决方案不是继续调模型，而是先把 benchmark 做对。\n"
    "这个新 benchmark 在规模和复杂度上都显著提升：更多 case、更多 fault type、更大的服务数和更深的调用链。\n"
    "但更重要的创新其实不是更大，而是 impact-driven validation。\n"
    "我们发现很多注入进去的 fault 实际上对用户没有可观测影响，这种 case 对运维场景并不关键。\n"
    "因此我们用 SLI 去过滤，只保留那些真正会造成用户侧影响的 case，让评测更贴近真实运维。"
)

# ============================================================================
# SLIDE 8: SIX-STAGE FRAMEWORK
# ============================================================================

sb = prs.slide(title="Six-Stage Framework")

pipeline = Flowchart(
    nodes={
        "a": ("System\n(50 services)", "primary"),
        "b": ("Workload\n(36 paths)", "secondary"),
        "c": ("Fault Injection\n(31 types)", "accent"),
        "d": ("Collection\n(multi-modal)", "primary"),
        "e": ("Annotation\n(hierarchical)", "secondary"),
        "f": ("Validation\n(SLI-based)", "positive"),
    },
    edges=[("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")],
    direction="right",
    height=0.95,
    node_widths={key: 1.45 for key in ["a", "b", "c", "d", "e", "f"]},
)

workflow_visual = VStack(gap=0.15, children=[
    Image(path=asset("wm-workflow-summary.png"), height=1.75),
    TextBlock(text="Pipeline design principle: traceable transitions across stages", font_size="small", color="text_mid"),
])

framework_cards = Grid(cols=3, gap=0.2, children=[
    VStack(gap=0.1, children=[
        SvgImage(svg=svg_server(color="#8B4513"), width=0.6, height=0.6),
        TextBlock(text="1. Realistic System Scope", font_size="small", bold=True, color="primary"),
        TextBlock(text="50 services, 36 workloads, and 31 injected fault types prevent shortcut solutions.", font_size="small", color="text_mid"),
    ]),
    VStack(gap=0.1, children=[
        SvgImage(svg=svg_trace(color="#2E6B4F"), width=0.6, height=0.6),
        TextBlock(text="2. Multi-Modal Evidence", font_size="small", bold=True, color="secondary"),
        TextBlock(text="Metrics, logs, and traces are collected together so RCA methods must reconcile conflicting signals.", font_size="small", color="text_mid"),
    ]),
    VStack(gap=0.1, children=[
        SvgImage(svg=svg_warning(color="#4A6FA5"), width=0.6, height=0.6),
        TextBlock(text="3. Impact-Driven Filtering", font_size="small", bold=True, color="accent"),
        TextBlock(text="Only cases with observable SLI impact survive, making the benchmark operationally meaningful.", font_size="small", color="text_mid"),
    ]),
])

framework_takeaway = Callout(
    title="Why the framework matters",
    body="The benchmark is not just larger; it closes the loop from realistic faults to validated user-facing impact.",
    color="primary",
)

sb.layout(VStack(gap=0.16, children=[
    Padding(child=pipeline, left=0.15, right=0.15),
    Padding(child=framework_cards, left=0.25, right=0.25),
    Padding(child=framework_takeaway, left=0.35, right=0.35),
]))

sb.animate([[pipeline], [framework_cards], [framework_takeaway]])
sb.notes(
    "这页展示的是我们构建 benchmark 的整体流程。\n"
    "从左到右，先是系统本身和 workload 设计，然后进行 fault injection，接着收集多模态数据，再做层次化标注，最后进行基于 SLI 的验证。\n"
    "下面三张卡片可以帮助理解这个流程为什么重要。\n"
    "第一，它覆盖了足够真实的系统规模和故障复杂度；第二，它保留了 metrics、logs、traces 之间的联合证据；第三，它不是只看有没有注入 fault，而是看这个 fault 是否真的造成了用户可感知的影响。\n"
    "所以这个 framework 的价值在于，把 realistic fault 和 validated user impact 真正闭环起来了。"
)

# ============================================================================
# SLIDE 9: RE-EVALUATION RESULTS
# ============================================================================

sb = prs.slide(title="Re-evaluation: The Harsh Reality")

# Performance comparison chart
chart = BarChart(
    groups=[
        ("Old Benchmarks", [
            ("SOTA Avg", 0.75, "primary"),
            ("SimpleRCA", 0.85, "accent"),
        ]),
        ("Our Benchmark", [
            ("SOTA Avg", 0.21, "negative"),
            ("Best Model", 0.37, "warning"),
            ("SimpleRCA", 0.28, "accent"),
        ]),
    ],
    y_label="Top@1 Accuracy",
    show_values=True
)

# Key findings
findings = VStack(gap=0.18, children=[
    Callout(
        title="Performance Drop",
        body="Top@1: 0.75 → 0.21\nBest model: only 0.37",
        color="negative"
    ),
    HStack(gap=0.3, children=[
        SvgImage(svg=svg_speed(), width=0.5, height=0.5),
        TextBlock(text="Execution time: seconds → hours", font_size="body"),
    ]),
    TextBlock(
        text="Harder benchmark, slower inference.",
        font_size="small",
        color="text_mid",
    ),
])

reeval_stats = Grid(cols=2, gap=0.22, children=[
    VStack(gap=0.1, children=[
        RoundedBox(text="72%", color="negative", text_color="white", bold=True, width=1.2, height=0.45),
        TextBlock(text="average Top@1 drop", font_size="caption"),
    ]),
    VStack(gap=0.1, children=[
        RoundedBox(text="0.37", color="warning", text_color="white", bold=True, width=1.2, height=0.45),
        TextBlock(text="best model on new benchmark", font_size="caption"),
    ]),
    VStack(gap=0.1, children=[
        RoundedBox(text="hrs", color="primary", text_color="white", bold=True, width=1.2, height=0.45),
        TextBlock(text="execution shifts from seconds", font_size="caption"),
    ]),
    VStack(gap=0.1, children=[
        RoundedBox(text="0.28", color="accent", text_color="white", bold=True, width=1.2, height=0.45),
        TextBlock(text="SimpleRCA still competitive", font_size="caption"),
    ]),
])

sb.layout(HStack(gap=0.32, children=[
    Padding(child=VStack(gap=0.16, width=7.05, children=[
        TextBlock(text="Reported SOTA gains shrink sharply once the benchmark reflects realistic propagation.", font_size="caption", color="text_mid"),
        chart,
    ]), left=0.0, right=0.0, width=7.05),
    VStack(gap=0.14, width=3.75, children=[findings, reeval_stats]),
]))

sb.animate([[chart], [findings], [reeval_stats]])
sb.notes(
    "接下来是最关键的一页：把已有方法重新放到我们的 benchmark 上评估。\n"
    "可以看到，在旧 benchmark 上看起来很强的方法，到了新 benchmark 上性能大幅下降。\n"
    "平均 Top@1 从大约 0.75 掉到 0.21，最好的模型也只有 0.37。\n"
    "而且问题不只是精度下降，执行时间也从秒级上升到小时级，这对实际运维来说是非常关键的。\n"
    "所以这页想传达的结论是：旧 benchmark 很大程度上奖励的是 shortcut correlation，而不是可扩展、可落地的 RCA 能力。"
)

# ============================================================================
# SLIDE 10: THREE FAILURE PATTERNS
# ============================================================================

sb = prs.slide(title="Three Critical Failure Patterns")

patterns = HStack(gap=0.26, children=[
    VStack(gap=0.16, width=3.5, children=[
        HStack(gap=0.18, children=[
            SvgImage(svg=svg_speed(), width=0.65, height=0.65),
            RoundedBox(text="39.8%", color="negative", text_color="white", bold=True, width=1.2, height=0.42),
        ]),
        TextBlock(text="1. Scalability Breakdowns", font_size="body", bold=True, color="negative"),
        TextBlock(text="Longer telemetry windows and larger graphs cause near-linear time growth.", font_size="caption", color="text_mid"),
        TextBlock(text="Example: 8 min data -> 10 min processing", font_size="caption", color="text"),
    ]),
    VStack(gap=0.16, width=3.5, children=[
        HStack(gap=0.18, children=[
            SvgImage(svg=svg_eye(), width=0.65, height=0.65),
            RoundedBox(text="47.4%", color="warning", text_color="white", bold=True, width=1.2, height=0.42),
        ]),
        TextBlock(text="2. Observability Blind Spots", font_size="body", bold=True, color="warning"),
        TextBlock(text="RCA models miss weak or conflicting signals across unmonitored components.", font_size="caption", color="text_mid"),
        TextBlock(text="Typical symptoms: signal loss, hidden hops, contradictory evidence", font_size="caption", color="text"),
    ]),
    VStack(gap=0.16, width=3.5, children=[
        HStack(gap=0.18, children=[
            SvgImage(svg=svg_puzzle(), width=0.65, height=0.65),
            RoundedBox(text="conceptual", color="accent", text_color="white", bold=True, width=1.75, height=0.42),
        ]),
        TextBlock(text="3. Modeling Bottlenecks", font_size="body", bold=True, color="accent"),
        TextBlock(text="Built-in assumptions about locality and causality fail in dense propagation scenarios.", font_size="caption", color="text_mid"),
        TextBlock(text="The failure is conceptual, not only numerical", font_size="caption", color="text"),
    ]),
])

warning_visual = HStack(gap=0.25, children=[
    SvgImage(svg=svg_warning(color="#4A6FA5"), width=0.8, height=0.8),
    TextBlock(
        text="Failure is multi-dimensional: speed, observability, and model assumptions collapse together.",
        font_size="body",
        color="text_mid",
    ),
])

failure_takeaway = Callout(
    title="What these patterns imply",
    body="Better RCA needs scalable inference, coverage-aware observability, and models that tolerate long propagation chains.",
    color="accent",
)

sb.layout(VStack(gap=0.24, children=[
    Padding(child=patterns, all=0.25),
    Padding(child=warning_visual, left=0.6, right=0.6),
    Padding(child=failure_takeaway, left=0.35, right=0.35),
]))

sb.animate([
    [patterns.children[0]],
    [patterns.children[1]],
    [patterns.children[2]],
    [warning_visual],
    [failure_takeaway],
])
sb.notes(
    "我们进一步分析了这些方法失败的原因，主要归纳成三类。\n"
    "第一类是 scalability breakdown，也就是随着时间窗口和图规模增长，推理代价快速上升。\n"
    "第二类是 observability blind spot，模型面对弱信号、缺失监控或者相互矛盾的证据时容易失效。\n"
    "第三类是 modeling bottleneck，也就是方法本身对 locality、causality 等假设在复杂传播场景下不再成立。\n"
    "这三类问题一起说明，未来做 RCA 不能只看 accuracy，还要同时考虑可扩展性、观测覆盖和模型假设的鲁棒性。"
)

# ============================================================================
# SLIDE 11: CONCLUSION
# ============================================================================

sb = prs.slide(title="Conclusion & Contributions")

core_message = Callout(
    title="Core message",
    body="Benchmarks must reject trivial heuristics, reflect propagation depth, and remain operationally meaningful.",
    color="primary",
)

# Contributions
contributions = VStack(gap=0.12, children=[
    TextBlock(text="Contributions:", font_size="heading", bold=True),
    Grid(cols=2, gap=0.14, children=[
        VStack(gap=0.04, children=[
            SvgImage(svg=svg_search(), width=0.5, height=0.5),
            TextBlock(text="Revealed benchmark oversimplification", font_size="caption", bold=True),
            TextBlock(text="Simple baselines can match or beat reported SOTA methods.", font_size="small", color="text_mid"),
        ]),
        VStack(gap=0.04, children=[
            SvgImage(svg=svg_tools(), width=0.5, height=0.5),
            TextBlock(text="Built propagation-aware benchmark", font_size="caption", bold=True),
            TextBlock(text="System size, fault diversity, and call depth scale together.", font_size="small", color="text_mid"),
        ]),
        VStack(gap=0.04, children=[
            SvgImage(svg=svg_warning(), width=0.5, height=0.5),
            TextBlock(text="Identified three failure patterns", font_size="caption", bold=True),
            TextBlock(text="The analysis surfaces scalability, visibility, and modeling bottlenecks.", font_size="small", color="text_mid"),
        ]),
        VStack(gap=0.04, children=[
            SvgImage(svg=svg_github(), width=0.5, height=0.5),
            TextBlock(text="Open-sourced all artifacts", font_size="caption", bold=True),
            TextBlock(text="Code, data, labels, and evaluation scripts support reproducibility.", font_size="small", color="text_mid"),
        ]),
    ]),
])

closing_points = VStack(gap=0.1, width=4.0, children=[
    TextBlock(text="Bottom line", font_size="heading", bold=True, color="primary"),
    HStack(gap=0.14, children=[
        RoundedBox(text="1", color="primary", text_color="white", bold=True, width=0.42, height=0.42),
        TextBlock(text="Reject trivial heuristics before claiming progress.", font_size="caption"),
    ]),
    HStack(gap=0.14, children=[
        RoundedBox(text="2", color="secondary", text_color="white", bold=True, width=0.42, height=0.42),
        TextBlock(text="Realistic RCA evaluation must include propagation depth and user impact.", font_size="small"),
    ]),
    HStack(gap=0.14, children=[
        RoundedBox(text="3", color="accent", text_color="white", bold=True, width=0.42, height=0.42),
        TextBlock(text="Current methods still have substantial headroom on realistic benchmarks.", font_size="small"),
    ]),
])

sb.layout(VStack(gap=0.10, children=[
    Padding(child=core_message, left=0.25, right=0.25),
    Padding(child=HStack(gap=0.24, children=[
        VStack(gap=0.18, width=6.9, children=[contributions]),
        VStack(gap=0.18, width=4.1, children=[closing_points]),
    ]), left=0.04, right=0.04),
]))

sb.animate([[core_message], [contributions], [closing_points]])
sb.notes(
    "最后总结一下。\n"
    "这项工作的核心 message 是：benchmark 本身必须 evolve with models，不能让简单启发式轻易通过，然后再宣称模型有巨大进步。\n"
    "我们的贡献主要有四点：揭示了现有 benchmark 的过度简化；构建了 propagation-aware 的新 benchmark；系统分析了三类 failure pattern；并把代码、数据和评测脚本开源出来。\n"
    "从更大的角度看，这项工作想强调的是，benchmark design 本身就是 RCA 研究中的一等问题，而不只是辅助工作。"
)

# ============================================================================
# SLIDE 12: THANK YOU
# ============================================================================

sb = prs.slide(background="#0A101B")

thanks_layout = HStack(gap=0.4, children=[
    VStack(gap=0.25, width=5.0, children=[
        RoundedBox(
            text="THANK YOU",
            color="#1E3A5F",
            border="#2D5B94",
            text_color="#EAF2FC",
            font_size="caption",
            bold=True,
            width=1.8,
            height=0.42,
        ),
        TextBlock(text="Questions?", font_size=42, bold=True, color="#F2F6FC", height=0.95),
        TextBlock(text="Code, data, and benchmark artifacts are open-sourced.", font_size="body", color="#D5E0EE"),
        TextBlock(text="Contact: Aoyang Fang et al. (CUHK-Shenzhen)", font_size="caption", color="#B1C1D6"),
        TextBlock(text="Image credits in assets/IMAGE_CREDITS.md", font_size="small", color="#8EA2BE"),
    ]),
    VStack(gap=0.18, width=5.4, children=[
        Image(path=asset("wm-data-center-unc.jpg"), width=5.4, height=3.0),
        TextBlock(text="Toward realistic and reliable RCA evaluation", font_size="small", color="#8EA2BE"),
    ]),
])

sb.layout(Padding(child=thanks_layout, all=0.3))
sb.animate([[thanks_layout.children[0]], [thanks_layout.children[1]]])
sb.notes(
    "我的汇报就到这里。\n"
    "如果大家感兴趣，我们很愿意进一步讨论 benchmark 设计、failure pattern 分析，或者这个 benchmark 对未来 RCA 模型研究意味着什么。\n"
    "也欢迎大家关注我们的开源数据和代码。谢谢大家。"
)

# ============================================================================
# SAVE AND VALIDATE
# ============================================================================

output_path = Path(__file__).with_name("rca_benchmark_presentation.pptx")


if __name__ == "__main__":
    prs.save(str(output_path))

    print(f"Presentation saved to: {output_path}")
    print("\nRunning validation...")
    report = prs.review()
    print(f"Total slides: {report['total_slides']}")
    print(f"Issues found: {report['total_issues']}")
    if report['issues']:
        for issue in report['issues']:
            print(f"  - [{issue['type']}] {issue['detail']}")
    else:
        print("No issues found!")
