from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paperops.slides.dsl import Deck, Grid, Heading, KPI, Slide, Subtitle, Text, Title

OUTPUT_FILE = Path(__file__).with_name("fse26_ir_first.pptx")


def build_deck() -> Deck:
    deck = Deck(
        theme="minimal",
        sheet="academic",
        meta={
            "title": "FSE 2026 benchmark story",
            "author": "paperops examples",
            "lang": "en",
        },
        styles={
            ".hero": {"bg": "bg_accent", "padding": "xl", "radius": "md"},
            ".fact": {"bg": "bg_alt", "padding": "md", "radius": "sm", "border": "border"},
        },
    )
    deck += Slide(class_="cover")[
        Title("Rethinking microservice RCA evaluation with propagation-aware tasks"),
        Subtitle("FSE'26 example deck migrated to the IR-first API"),
        Text("The benchmark should test whether a model can explain an incident, not merely classify it.", class_="hero"),
    ]
    deck += Slide()[
        Heading("Why current RCA benchmarks mislead us"),
        Grid(style={"cols": "1fr 1fr 1fr", "gap": "md"})[
            Text("Root-cause labels collapse multi-hop incidents into one token.", class_="fact"),
            Text("Most datasets do not preserve the temporal path from intervention to symptom.", class_="fact"),
            Text("Leaderboard gains can come from exploiting superficial dataset priors.", class_="fact"),
        ],
    ]
    deck += Slide()[
        Heading("Benchmark recipe"),
        Grid(style={"cols": "1.4fr 1fr", "gap": "lg"})[
            Text(
                "Collect realistic workloads, inject controlled faults, record metrics/logs/traces, and annotate the forward propagation chain."
                " Each case therefore supports both answer checking and reasoning-path checking.",
                class_="hero",
            ),
            Grid(style={"cols": "1fr", "gap": "sm"})[
                KPI(label="Fault families", value="6", delta="db, queue, network...", trend="neutral"),
                KPI(label="Cases", value="9,152", delta="validated", trend="positive"),
                KPI(label="Signals", value="3", delta="multi-modal", trend="positive"),
            ],
        ],
    ]
    deck += Slide()[
        Heading("What a better evaluation scores"),
        Grid(style={"cols": "1fr 1fr", "gap": "lg"})[
            Text("Accuracy: does the system find the triggering service or fault?", class_="fact"),
            Text("Faithfulness: do the cited observations and propagation path match the intervention?", class_="fact"),
            Text("Coverage: does the explanation account for metrics, logs, and traces together?", class_="fact"),
            Text("Actionability: would an operator trust the recommendation enough to act on it?", class_="fact"),
        ],
    ]
    return deck


def build_presentation(*, output_path: Path | None = None) -> Path:
    destination = output_path or OUTPUT_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    return build_deck().render(destination)


def main() -> None:
    build_presentation()


if __name__ == "__main__":
    main()
