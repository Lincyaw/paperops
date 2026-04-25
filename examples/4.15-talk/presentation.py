from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paperops.slides.dsl import (
    Card,
    Deck,
    Grid,
    Heading,
    KPI,
    Slide,
    Subtitle,
    Title,
)

OUTPUT_FILE = Path(__file__).with_name("talk_4_15_ir_first.pptx")


SLIDE_TITLES = [
    "World-model RCA needs benchmark tasks that preserve causal structure",
    "Three gaps make current RCA evaluation too easy to game",
    "A propagation-aware benchmark turns diagnosis into a reasoning task",
    "The benchmark should score evidence quality, not just final labels",
    "IR-first authoring lets one story ship in multiple visual sheets",
]


BASE_STYLES = {
    ".hero-slide": {"bg": "#0B1724", "gap": "md"},
    ".cover-title": {"font": 29, "font-weight": "bold", "color": "#F8FAFC"},
    ".cover-subtitle": {"font": 17, "color": "#B7C7D8"},
    ".dark-card": {
        "bg": "#14233A",
        "color": "#F8FAFC",
        "border": "#37506B",
        "padding": "lg",
        "radius": "md",
        "font": 17,
        "line-height": 1.22,
    },
    ".dark-kpi": {"bg": "#2563EB", "color": "#F8FAFC", "border": "#60A5FA"},
    ".problem-card": {
        "bg": "#F8FBFE",
        "color": "#142033",
        "padding": "md",
        "radius": "md",
        "border": "#BFDBFE",
        "font": 15,
        "line-height": 1.18,
        "height": 1.25,
    },
    ".callout": {
        "bg": "#EAF2FF",
        "color": "#12345A",
        "border": "#3182CE",
        "font": 14,
        "height": 0.9,
    },
    ".chain": {
        "bg": "#122033",
        "color": "#F8FAFC",
        "border": "#60A5FA",
        "font": 17,
        "height": 2.1,
    },
    ".rail": {
        "bg": "#EDF7F2",
        "color": "#153B2B",
        "padding": "md",
        "radius": "sm",
        "border": "#38A169",
    },
    ".score-card": {
        "bg": "#FFF7ED",
        "color": "#3B2410",
        "border": "#F59E0B",
        "font": 15,
        "height": 0.9,
    },
    ".variant-card": {
        "bg": "#F7FAFC",
        "padding": "md",
        "radius": "md",
        "border": "#CBD5E1",
        "font": 14,
        "height": 0.8,
    },
}


def build_deck(*, sheet: str = "seminar") -> Deck:
    deck = Deck(
        theme="minimal",
        sheet=sheet,
        meta={
            "title": "HKU talk - world-model RCA benchmark",
            "author": "paperops examples",
            "lang": "en",
        },
        styles=BASE_STYLES,
    )

    deck += Slide(class_="cover hero-slide")[
        Title(SLIDE_TITLES[0], class_="cover-title"),
        Subtitle(
            "HKU seminar cutover example built with the IR-first SlideCraft DSL",
            class_="cover-subtitle",
        ),
        Grid(style={"cols": "1.35fr 0.8fr", "gap": "lg"})[
            Card(
                "Claim: trustworthy RCA systems must reason over interventions, propagation paths, and evidence chains instead of shortcut labels.",
                class_="dark-card",
            ),
            Grid(style={"cols": "1fr", "gap": "sm"})[
                KPI(
                    label="Cases",
                    value="9,152",
                    delta="validated",
                    trend="positive",
                    class_="dark-kpi",
                ),
                KPI(
                    label="Signals",
                    value="3",
                    delta="metrics / logs / traces",
                    trend="neutral",
                    class_="dark-kpi",
                ),
                KPI(
                    label="Target",
                    value="Trust",
                    delta="faithful reasoning",
                    trend="positive",
                    class_="dark-kpi",
                ),
            ],
        ],
    ]

    deck += Slide(style={"justify": "center", "gap": "md"})[
        Heading(SLIDE_TITLES[1]),
        Card(
            "Why labels are not enough: a benchmark can look strong while rewarding artifact matching instead of causal diagnosis.",
            class_="callout",
        ),
        Grid(style={"cols": "1fr 1fr 1fr", "gap": "md"})[
            Card(
                "Shallow labels: models can name a service without proving the explanation is causally valid.",
                class_="problem-card",
            ),
            Card(
                "Weak environments: narrow templates let models memorize incident artifacts rather than mechanisms.",
                class_="problem-card",
            ),
            Card(
                "Missing interventions: scores rarely ask whether the claimed fault reproduces downstream evidence.",
                class_="problem-card",
            ),
        ],
    ]

    deck += Slide(style={"justify": "center", "gap": "lg"})[
        Heading(SLIDE_TITLES[2]),
        Grid(style={"cols": "1.35fr 0.85fr", "gap": "lg"})[
            Card(
                "Workload -> fault injection -> metrics/logs/traces -> propagation labels -> validated benchmark cases. The key move is recording the forward causal path, not just the final root-cause label.",
                class_="chain",
            ),
            Grid(style={"cols": "1fr", "gap": "sm"})[
                KPI(label="Fault paths", value="6+", delta="families", trend="neutral"),
                KPI(
                    label="Evidence",
                    value="3-way",
                    delta="cross-modal",
                    trend="positive",
                ),
                KPI(label="Check", value="path", delta="not token", trend="positive"),
            ],
        ],
    ]

    deck += Slide(style={"justify": "center", "gap": "lg"})[
        Heading(SLIDE_TITLES[3]),
        Grid(style={"cols": "1fr 1fr", "gap": "md"})[
            Card(
                "Outcome accuracy: identify the triggering fault or service.",
                class_="score-card",
            ),
            Card(
                "Propagation fidelity: match the annotated causal path.",
                class_="score-card",
            ),
            Card(
                "Evidence coverage: cite metrics, logs, and traces together.",
                class_="score-card",
            ),
            Card(
                "Intervention consistency: predict what changes if the fault is removed.",
                class_="score-card",
            ),
        ],
    ]

    deck += Slide(style={"justify": "center", "gap": "lg"})[
        Heading(SLIDE_TITLES[4]),
        Grid(style={"cols": "1fr 1fr 1fr", "gap": "md"})[
            Card(
                "`seminar`: balanced density for research talks.", class_="variant-card"
            ),
            Card(
                "`keynote`: larger emphasis blocks and sharper contrast.",
                class_="variant-card",
            ),
            Card(
                "`whitepaper`: denser reading lanes for report-like decks.",
                class_="variant-card",
            ),
        ],
        Card(
            "Same IR, different sheets: authors keep semantic structure stable while visual rhythm changes at render time.",
            class_="rail",
        ),
    ]
    return deck


def build_presentation(
    *, output_path: Path | None = None, sheet: str = "seminar"
) -> Path:
    destination = output_path or OUTPUT_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    return build_deck(sheet=sheet).render(destination)


def main() -> None:
    build_presentation()


if __name__ == "__main__":
    main()
