"""
Presentation for FSE'26 Paper:
"Rethinking the Evaluation of Microservice RCA with a Fault Propagation-Aware Benchmark"

Target audience: Cross-domain researchers (not necessarily SE/ops experts)
Duration: ~10 minutes
"""

from paperops.slides import (
    Presentation, themes,
    HStack, VStack, Grid, Padding,
    RoundedBox, Circle,
    TextBlock, BulletList, Table,
    Callout, Flowchart,
    BarChart,
    SvgImage,
)

# Create presentation with academic theme
prs = Presentation(theme=themes.academic)

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

prs.cover(
    "Rethinking the Evaluation of Microservice RCA\nwith a Fault Propagation-Aware Benchmark",
    subtitle="FSE 2026",
    author="Aoyang Fang, Songhan Zhang, Yifan Yang, Haotong Wu, Junjielong Xu,\n"
            "Xuyang Wang, Rui Wang, Manyi Wang, Qisheng Lu, Pinjia He\n"
            "The Chinese University of Hong Kong, Shenzhen"
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

sb.layout(VStack(gap=0.4, children=[
    Padding(child=cases, left=0.2, right=0.2),
    Padding(child=rca_def, left=0.2, right=0.2),
    Padding(child=modalities, left=0.2, right=0.2),
]))

sb.animate([[cases], [rca_def], [modalities]])
sb.notes("Start with real incidents to show importance. Then define RCA task and three data modalities.")

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

# The question
question = Callout(
    title="But...",
    body="Different papers use different datasets\nAre these numbers reliable?",
    color="warning"
)

sb.layout(VStack(gap=0.4, children=[
    Padding(child=evolution, left=0.2, right=0.2),
    Padding(child=performance, left=0.2, right=0.2),
    Padding(child=question, left=0.3, right=0.3),
]))

sb.animate([[evolution], [performance], [question]])
sb.notes("Show the progression of methods and reported high performance. Then raise the question about reliability.")

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

sb.layout(VStack(gap=0.4, children=[
    Padding(child=core_q, left=0.3, right=0.3),
    Padding(child=simplerca, left=0.2, right=0.2),
]))

sb.animate([[core_q], [simplerca]])
sb.notes("Introduce the core question and SimpleRCA design.")

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

sb.layout(VStack(gap=0.5, children=[
    Padding(child=results_table, left=0.8, right=0.8),
    Padding(child=conclusion, left=0.3, right=0.3),
]))

sb.animate([[results_table], [conclusion]])
sb.notes("Present the comparison results. Highlight that SimpleRCA matches or beats SOTA.")

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

sb.layout(VStack(gap=0.4, children=[
    Padding(child=limitations, left=0.2, right=0.2),
    Padding(child=pattern, left=0.2, right=0.2),
]))

sb.animate([[limitations.children[0]], [limitations.children[1]], [pattern]])
sb.notes("Explain why existing benchmarks are problematic.")

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

sb.layout(VStack(gap=0.5, children=[
    Padding(child=numbers, left=0.3, right=0.3),
    Padding(child=innovation, left=0.3, right=0.3),
]))

sb.animate([[numbers], [innovation]])
sb.notes("Present the new benchmark statistics and core innovation.")

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
    direction="right"
)

sb.layout(Padding(child=pipeline, all=0.3))

sb.animate([[pipeline]])
sb.notes("Show the six-stage pipeline.")

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
findings = VStack(gap=0.3, children=[
    Callout(
        title="Performance Drop",
        body="Top@1: 0.75 → 0.21\nBest model: only 0.37",
        color="negative"
    ),
    HStack(gap=0.3, children=[
        SvgImage(svg=svg_speed(), width=0.5, height=0.5),
        TextBlock(text="Execution time: seconds → hours", font_size="body"),
    ]),
])

sb.layout(HStack(gap=0.4, children=[
    Padding(child=chart, left=0.2),
    Padding(child=findings, right=0.2),
]))

sb.animate([[chart], [findings]])
sb.notes("Show the dramatic performance drop on new benchmark.")

# ============================================================================
# SLIDE 10: THREE FAILURE PATTERNS
# ============================================================================

sb = prs.slide(title="Three Critical Failure Patterns")

patterns = VStack(gap=0.4, children=[
    HStack(gap=0.3, children=[
        VStack(gap=0.2, children=[
            SvgImage(svg=svg_speed(), width=0.7, height=0.7),
            TextBlock(text="1. Scalability (39.8%)", font_size="body", bold=True, color="negative"),
            TextBlock(text="Linear time growth\n8min data → 10min process", font_size="caption", color="text_mid"),
        ]),
        VStack(gap=0.2, children=[
            SvgImage(svg=svg_eye(), width=0.7, height=0.7),
            TextBlock(text="2. Blind Spots (47.4%)", font_size="body", bold=True, color="warning"),
            TextBlock(text="Signal loss, unmonitored\ncomponents, contradictions", font_size="caption", color="text_mid"),
        ]),
        VStack(gap=0.2, children=[
            SvgImage(svg=svg_puzzle(), width=0.7, height=0.7),
            TextBlock(text="3. Modeling Bottlenecks", font_size="body", bold=True, color="accent"),
            TextBlock(text="Core assumptions fail\nunder complex scenarios", font_size="caption", color="text_mid"),
        ]),
    ]),
])

sb.layout(Padding(child=patterns, all=0.3))

sb.animate([
    [patterns.children[0].children[0]],
    [patterns.children[0].children[1]],
    [patterns.children[0].children[2]],
])
sb.notes("Explain the three failure patterns found in analysis.")

# ============================================================================
# SLIDE 11: CONCLUSION
# ============================================================================

sb = prs.slide(title="Conclusion & Contributions")

# Key messages
messages = VStack(gap=0.3, children=[
    Callout(
        title="Key Messages",
        body="Benchmarks must evolve with models\nSimple baselines should be first gate\nNeed realistic, challenging evaluation",
        color="primary"
    ),
])

# Contributions
contributions = VStack(gap=0.3, children=[
    TextBlock(text="Contributions:", font_size="heading", bold=True),
    Grid(cols=2, gap=0.3, children=[
        VStack(gap=0.1, children=[
            SvgImage(svg=svg_search(), width=0.5, height=0.5),
            TextBlock(text="Revealed benchmark oversimplification", font_size="caption"),
        ]),
        VStack(gap=0.1, children=[
            SvgImage(svg=svg_tools(), width=0.5, height=0.5),
            TextBlock(text="Built propagation-aware benchmark", font_size="caption"),
        ]),
        VStack(gap=0.1, children=[
            SvgImage(svg=svg_warning(), width=0.5, height=0.5),
            TextBlock(text="Identified three failure patterns", font_size="caption"),
        ]),
        VStack(gap=0.1, children=[
            SvgImage(svg=svg_github(), width=0.5, height=0.5),
            TextBlock(text="Open-sourced all artifacts", font_size="caption"),
        ]),
    ]),
])

sb.layout(VStack(gap=0.4, children=[
    Padding(child=messages, left=0.3, right=0.3),
    Padding(child=contributions, left=0.3, right=0.3),
]))

sb.animate([[messages], [contributions]])
sb.notes("Summarize key messages and contributions.")

# ============================================================================
# SLIDE 12: THANK YOU
# ============================================================================

prs.end("Thank You", subtitle="Questions?")

# ============================================================================
# SAVE AND VALIDATE
# ============================================================================

output_path = "/Users/lincyaw/workspace/paperops/examples/fse26/rca_benchmark_presentation.pptx"
prs.save(output_path)

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
