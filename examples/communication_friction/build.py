from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paperops.slides.dsl import Deck, Grid, Heading, Slide, Subtitle, Text, Title

OUTPUT_FILE = Path(__file__).with_name("communication_friction_ir_first.pptx")


def build_deck() -> Deck:
    deck = Deck(
        theme="minimal",
        sheet="pitch",
        meta={"title": "减少沟通摩擦", "author": "paperops examples", "lang": "zh-CN"},
        styles={
            ".layer": {"bg": "bg_alt", "padding": "md", "radius": "md", "border": "border"},
            ".accent-strip": {"bg": "accent", "color": "bg", "padding": "md", "radius": "sm"},
        },
    )
    deck += Slide(class_="cover")[
        Title("减少沟通摩擦：信息的三层结构"),
        Subtitle("从想法到表达，中间要经过可协作、可推理、可呈现的多次压缩"),
        Text("好的表达不是把内容变少，而是把每一层都压缩成对下游最友好的接口。", class_="accent-strip"),
    ]
    deck += Slide()[
        Heading("三层各自负责什么"),
        Grid(style={"cols": "1fr 1fr 1fr", "gap": "md"})[
            Text("探索层：保留问题空间、备选路径、失败尝试。", class_="layer"),
            Text("协作层：把状态、约束、接口和决策变成他人可接手的对象。", class_="layer"),
            Text("呈现层：把受众真正需要的信号压缩成一条清晰叙事。", class_="layer"),
        ],
    ]
    deck += Slide()[
        Heading("摩擦从哪里来"),
        Grid(style={"cols": "1fr 1fr", "gap": "lg"})[
            Text("探索层内容直接扔给受众 -> 细节太多，看不见主线。", class_="layer"),
            Text("呈现层结论直接扔给协作者 -> 信息太薄，没法继续推进。", class_="layer"),
            Text("协作层没有稳定接口 -> 每次交接都要重新解释上下文。", class_="layer"),
            Text("解决办法：每次切层都明确保留什么、删除什么、用什么结构表达。", class_="layer"),
        ],
    ]
    deck += Slide()[
        Heading("给 LLM 的作者化提示"),
        Text(
            "先确定受众与目标，再选择 MDX / JSON / Python 前端；一页只保留一个核心论点，并把样式写进 sheet 或 styles，而不是塞进单个组件。",
            class_="accent-strip",
        ),
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
