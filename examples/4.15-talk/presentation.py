from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paperops.slides.dsl import Deck, Grid, Heading, KPI, Slide, Subtitle, Text, Title

OUTPUT_FILE = Path(__file__).with_name("talk_4_15_ir_first.pptx")


SLIDE_TITLES = [
    "World-model RCA needs benchmark tasks that preserve causal structure",
    "Three gaps make current RCA evaluation too easy to game",
    "A propagation-aware benchmark turns diagnosis into a reasoning task",
    "The benchmark should score evidence quality, not just final labels",
    "IR-first authoring lets one story ship in multiple visual sheets",
]


BASE_STYLES = {
    ".hero-card": {"bg": "primary", "color": "bg", "padding": "lg", "radius": "md"},
    ".hero-card .text": {"color": "bg"},
    ".problem-card": {"bg": "bg_alt", "padding": "md", "radius": "md", "border": "border"},
    ".pull": {"font": "heading", "color": "accent"},
    ".rail": {"bg": "bg_accent", "padding": "md", "radius": "sm"},
}


def build_deck(*, sheet: str = "seminar") -> Deck:
    deck = Deck(
        theme="minimal",
        sheet=sheet,
        meta={
            "title": "HKU talk — world-model RCA benchmark",
            "author": "paperops examples",
            "lang": "en",
        },
        styles=BASE_STYLES,
    )

    deck += Slide(class_="cover")[
        Title(SLIDE_TITLES[0]),
        Subtitle("HKU seminar cutover example built with the IR-first SlideCraft DSL"),
        Text(
            "Claim: trustworthy RCA systems must reason over interventions, propagation, and evidence chains instead of shortcut labels.",
            class_="pull",
        ),
    ]

    deck += Slide()[
        Heading(SLIDE_TITLES[1]),
        Grid(style={"cols": "1fr 1fr 1fr", "gap": "md"})[
            Text(
                "Shallow labels: benchmarks reward picking a service name without checking whether the explanation is causally valid.",
                class_="problem-card",
            ),
            Text(
                "Weak environments: incidents are sampled from narrow templates, so models can memorize artifacts instead of learning mechanisms.",
                class_="problem-card",
            ),
            Text(
                "Missing interventions: the evaluation rarely asks whether the claimed fault would reproduce the observed downstream evidence.",
                class_="problem-card",
            ),
        ],
    ]

    deck += Slide()[
        Heading(SLIDE_TITLES[2]),
        Grid(style={"cols": "1.5fr 1fr", "gap": "lg"})[
            Text(
                "Pipeline: define workload -> inject fault -> collect metrics/logs/traces -> annotate propagation -> package benchmark cases."
                " The important move is that annotation records the forward causal path, not just the final root-cause label.",
                class_="hero-card",
            ),
            Grid(style={"cols": "1fr", "gap": "sm"})[
                KPI(label="Cases", value="9,152", delta="fault paths", trend="neutral"),
                KPI(label="Signals", value="3", delta="metrics + logs + traces", trend="positive"),
                KPI(label="Goal", value="Trust", delta="faithful reasoning", trend="positive"),
            ],
        ],
    ]

    deck += Slide()[
        Heading(SLIDE_TITLES[3]),
        Grid(style={"cols": "1fr 1fr", "gap": "lg"})[
            Text(
                "Scoring axes: outcome accuracy, propagation fidelity, evidence coverage, and intervention consistency.",
                class_="rail",
            ),
            Text(
                "A model only earns full credit when the answer, cited evidence, and predicted forward path agree with the injected fault.",
                class_="rail",
            ),
        ],
    ]

    deck += Slide()[
        Heading(SLIDE_TITLES[4]),
        Grid(style={"cols": "1fr 1fr 1fr", "gap": "md"})[
            Text("Same IR -> `seminar`: balanced density for research talks.", class_="problem-card"),
            Text("Same IR -> `keynote`: higher contrast and larger emphasis blocks.", class_="problem-card"),
            Text("Same IR -> `whitepaper`: report-friendly rhythm with denser reading lanes.", class_="problem-card"),
        ],
    ]
    return deck


def build_presentation(*, output_path: Path | None = None, sheet: str = "seminar") -> Path:
    destination = output_path or OUTPUT_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    return build_deck(sheet=sheet).render(destination)


def main() -> None:
    build_presentation()


if __name__ == "__main__":
    main()
