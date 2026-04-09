"""减少沟通摩擦：信息的三层结构 — PPT 生成脚本.

Visual style: 「晨雾蓝」— icon-led, whitespace-rich, analogous blue-cyan-mint.
"""

import os

from paperops.slides import (
    Arrow,
    Badge,
    BulletList,
    Callout,
    Flow,
    Flowchart,
    Grid,
    HStack,
    Line,
    Padding,
    Presentation,
    RoundedBox,
    SvgCanvas,
    SvgImage,
    Table,
    TextBlock,
    VStack,
    themes,
)

# ============================================================
# 主题：晨雾蓝 + 仿宋
# ============================================================
theme = themes.professional.override(
    font_family="FangSong",
    colors={
        "primary": "#2E5A88",  # 呈现层 — 深海蓝
        "secondary": "#6BC4A6",  # 探索层 — 薄荷绿
        "accent": "#3D8B8B",  # 协作层 — 青石色
        "positive": "#4A9B80",  # 正面
        "negative": "#E8805C",  # 珊瑚橙 — 摩擦/警示
        "highlight": "#5B7FA5",  # 柔蓝
        "warning": "#D4955A",  # 暖橙
        "text": "#2C3E50",  # 深灰
        "text_mid": "#7B8FA3",  # 辅助灰
        "text_light": "#B0BEC5",  # 淡灰
        "bg": "#FAFBFC",  # 暖白
        "bg_alt": "#F0F4F8",  # 卡片底
        "bg_accent": "#E8EFF5",  # 强调底
        "border": "#E2E8F0",  # 分隔线
    },
)
prs = Presentation(theme=theme)


# ============================================================
# SVG 图标工厂 — 统一到晨雾蓝色板
# ============================================================

# 颜色常量（用于 SVG 内部，避免重复硬编码）
C_BLUE = "#2E5A88"
C_CYAN = "#3D8B8B"
C_MINT = "#6BC4A6"
C_CORAL = "#E8805C"
C_DARK = "#2C3E50"
C_MID = "#7B8FA3"
C_LIGHT_BG = "#F0F4F8"


def icon_flask():
    """探索层 — 实验烧瓶."""
    s = SvgCanvas(120, 120, theme=theme)
    s.polygon(
        [(38, 40), (82, 40), (98, 100), (22, 100)],
        fill=C_MINT,
        stroke="none",
        opacity=0.8,
    )
    s.rect(48, 20, 24, 22, fill=C_MINT, stroke="none", rx=3)
    s.rect(52, 20, 16, 4, fill="white", stroke="none", rx=2, opacity=0.6)
    s.line(30, 70, 90, 70, color="white", width=1, dashed=True)
    s.circle(50, 80, 5, fill="white", text_color="white", opacity=0.5)
    s.circle(68, 75, 3, fill="white", text_color="white", opacity=0.4)
    s.circle(55, 88, 4, fill="white", text_color="white", opacity=0.45)
    s.rect(22, 96, 76, 6, fill=C_MINT, stroke="none", rx=3, opacity=0.85)
    return s


def icon_bubbles():
    """协作层 — 对话气泡."""
    s = SvgCanvas(120, 120, theme=theme)
    s.rounded_rect(
        35,
        15,
        70,
        50,
        fill=C_CYAN,
        stroke="none",
        rx=18,
        opacity=0.4,
        text="",
        font_size=1,
    )
    s.polygon([(85, 60), (95, 75), (75, 58)], fill=C_CYAN, stroke="none", opacity=0.4)
    s.rounded_rect(
        15,
        40,
        70,
        50,
        fill=C_CYAN,
        stroke="none",
        rx=18,
        opacity=0.8,
        text="",
        font_size=1,
    )
    s.polygon([(35, 85), (25, 100), (45, 83)], fill=C_CYAN, stroke="none", opacity=0.8)
    s.circle(35, 63, 4, fill="white", text_color="white", opacity=0.7)
    s.circle(50, 63, 4, fill="white", text_color="white", opacity=0.7)
    s.circle(65, 63, 4, fill="white", text_color="white", opacity=0.7)
    return s


def icon_stage():
    """呈现层 — 聚光灯讲台."""
    s = SvgCanvas(120, 120, theme=theme)
    s.polygon(
        [(50, 10), (70, 10), (95, 75), (25, 75)],
        fill=C_BLUE,
        stroke="none",
        opacity=0.15,
    )
    s.rounded_rect(
        46,
        5,
        28,
        16,
        fill=C_BLUE,
        stroke="none",
        rx=4,
        text="",
        font_size=1,
        opacity=0.85,
    )
    s.polygon(
        [(20, 78), (100, 78), (105, 90), (15, 90)],
        fill=C_BLUE,
        stroke="none",
        opacity=0.8,
    )
    s.rect(35, 90, 50, 20, fill=C_BLUE, stroke="none", rx=2, opacity=0.65)
    s.line(40, 83, 80, 83, color="white", width=1.5)
    return s


def icon_lightning():
    """摩擦 — 闪电."""
    s = SvgCanvas(120, 120, theme=theme)
    s.polygon(
        [(65, 8), (35, 55), (55, 55), (42, 112), (90, 50), (65, 50)],
        fill=C_CORAL,
        stroke="none",
        opacity=0.85,
    )
    s.polygon(
        [(63, 22), (48, 52), (58, 52), (50, 90), (78, 50), (63, 50)],
        fill="white",
        stroke="none",
        opacity=0.2,
    )
    return s


def icon_shuttle():
    """层间穿梭 — 双向环形箭头."""
    s = SvgCanvas(120, 120, theme=theme)
    s.path("M 30,50 A 30,30 0 0,1 90,50", stroke=C_BLUE, fill="none", stroke_width=5)
    s.polygon([(90, 42), (100, 50), (90, 58)], fill=C_BLUE, stroke="none")
    s.path("M 90,70 A 30,30 0 0,1 30,70", stroke=C_MINT, fill="none", stroke_width=5)
    s.polygon([(30, 62), (20, 70), (30, 78)], fill=C_MINT, stroke="none")
    return s


def icon_robot():
    """AI — 机器人头."""
    s = SvgCanvas(120, 120, theme=theme)
    s.line(60, 8, 60, 28, color=C_MID, width=3)
    s.circle(60, 8, 5, fill=C_MID, text_color=C_MID)
    s.rounded_rect(
        25,
        28,
        70,
        60,
        fill=C_MID,
        stroke="none",
        rx=14,
        text="",
        font_size=1,
        opacity=0.8,
    )
    s.circle(45, 52, 8, fill="white", text_color="white")
    s.circle(75, 52, 8, fill="white", text_color="white")
    s.circle(45, 52, 4, fill=C_BLUE, text_color=C_BLUE)
    s.circle(75, 52, 4, fill=C_BLUE, text_color=C_BLUE)
    s.rounded_rect(
        42,
        70,
        36,
        8,
        fill="white",
        stroke="none",
        rx=3,
        text="",
        font_size=1,
        opacity=0.8,
    )
    s.line(51, 70, 51, 78, color=C_MID, width=1)
    s.line(60, 70, 60, 78, color=C_MID, width=1)
    s.line(69, 70, 69, 78, color=C_MID, width=1)
    s.rect(15, 45, 10, 20, fill=C_MID, stroke="none", rx=3, opacity=0.6)
    s.rect(95, 45, 10, 20, fill=C_MID, stroke="none", rx=3, opacity=0.6)
    return s


def icon_human():
    """人 — 简笔人形."""
    s = SvgCanvas(120, 120, theme=theme)
    s.circle(60, 25, 16, fill=C_BLUE, text_color=C_BLUE, opacity=0.8)
    s.rounded_rect(
        38,
        46,
        44,
        45,
        fill=C_BLUE,
        stroke="none",
        rx=10,
        text="",
        font_size=1,
        opacity=0.8,
    )
    s.polygon([(50, 46), (60, 60), (70, 46)], fill="white", stroke="none", opacity=0.3)
    s.rounded_rect(
        18,
        50,
        20,
        10,
        fill=C_BLUE,
        stroke="none",
        rx=5,
        text="",
        font_size=1,
        opacity=0.65,
    )
    s.rounded_rect(
        82,
        50,
        20,
        10,
        fill=C_BLUE,
        stroke="none",
        rx=5,
        text="",
        font_size=1,
        opacity=0.65,
    )
    return s


def icon_dao():
    """术与道 — 太极简化."""
    s = SvgCanvas(120, 120, theme=theme)
    s.circle(60, 60, 45, fill=C_BLUE, text_color=C_BLUE, opacity=0.1)
    s.path("M 60,15 A 45,45 0 0,1 60,105", stroke=C_BLUE, fill="none", stroke_width=3)
    s.path("M 60,105 A 45,45 0 0,1 60,15", stroke=C_MINT, fill="none", stroke_width=3)
    s.path(
        "M 60,15 A 22.5,22.5 0 0,1 60,60 A 22.5,22.5 0 0,0 60,105",
        stroke=C_MID,
        fill="none",
        stroke_width=2,
    )
    s.circle(60, 37, 6, fill=C_MINT, text_color=C_MINT)
    s.circle(60, 83, 6, fill=C_BLUE, text_color=C_BLUE)
    return s


# 小圆点标记（替代 Badge 的轻量方案）
def dot_label(text, color):
    """小圆点 + 文字标签，替代大面积 Badge."""
    return HStack(
        gap=0.08,
        children=[
            RoundedBox(text="", color=color, width=0.18, height=0.18, border=color),
            TextBlock(text=text, font_size="caption", bold=True, color="text"),
        ],
    )


# ============================================================
# Slide 1: 封面
# ============================================================
prs.cover(
    title="减少沟通摩擦",
    subtitle="信息的三层结构与研究元能力",
)

# ============================================================
# Slide 2: 四个常见场景 — 图标+文字，无方框
# ============================================================
sb = prs.slide(title="这些场景是否似曾相识？")

ico_lightning = SvgImage(svg=icon_lightning(), width=1.8, height=1.8)

scenes = VStack(
    gap=0.25,
    children=[
        TextBlock(
            text="论文读起来逻辑清晰，但实际过程完全不同",
            font_size="body",
            color="text",
        ),
        TextBlock(text="初学者与高年级对话总是费劲", font_size="body", color="text"),
        TextBlock(
            text='提问只说"不work"，得不到有效回复', font_size="body", color="text"
        ),
        TextBlock(
            text="组会讲了20分钟细节，听众不知道重点", font_size="body", color="text"
        ),
    ],
)

conclusion = Callout(
    title="共同根源",
    body="不是能力问题，不是态度问题\n而是信息没有匹配接收方的 context 容量",
    color="negative",
)

sb.layout(
    VStack(
        gap=0.3,
        children=[
            HStack(gap=0.5, children=[scenes, ico_lightning]),
            conclusion,
        ],
    )
)
sb.animate(
    [
        [scenes],
        [ico_lightning],
        [conclusion],
    ]
)

# ============================================================
# Slide 3: 三层模型 — 大图标主导
# ============================================================
sb = prs.slide(title="模型：信息的三层结构")

ico_f = SvgImage(svg=icon_flask(), width=1.5, height=1.5)
ico_b = SvgImage(svg=icon_bubbles(), width=1.5, height=1.5)
ico_s = SvgImage(svg=icon_stage(), width=1.5, height=1.5)

col_explore = VStack(
    gap=0.1,
    children=[
        ico_f,
        TextBlock(text="探索层", font_size="heading", bold=True, color="secondary"),
        TextBlock(
            text="Notebook / 个人笔记\n完整的实验过程与推理链",
            font_size="caption",
            color="text_mid",
        ),
    ],
)
col_collab = VStack(
    gap=0.1,
    children=[
        ico_b,
        TextBlock(text="协作层", font_size="heading", bold=True, color="accent"),
        TextBlock(
            text="组会讨论 / 合作者消息\n支撑决策的关键信息",
            font_size="caption",
            color="text_mid",
        ),
    ],
)
col_present = VStack(
    gap=0.1,
    children=[
        ico_s,
        TextBlock(text="呈现层", font_size="heading", bold=True, color="primary"),
        TextBlock(
            text="论文 / Slides / Poster\n面向外部读者的narrative",
            font_size="caption",
            color="text_mid",
        ),
    ],
)

a1 = Arrow(from_component=col_explore, to_component=col_collab, color="border")
a2 = Arrow(from_component=col_collab, to_component=col_present, color="border")

sb.layout(
    HStack(
        gap=0.12,
        children=[
            col_explore,
            a1,
            col_collab,
            a2,
            col_present,
        ],
    )
)
sb.animate(
    [
        [col_explore],
        [a1, col_collab],
        [a2, col_present],
    ]
)

# ============================================================
# Slide 4: 每层特征
# ============================================================
sb = prs.slide(title="每层的特征")
tbl = Table(
    headers=["", "呈现层", "协作层", "探索层"],
    rows=[
        ["受众", "审稿人、读者", "合作者、导师", "自己"],
        ["注意力", "低", "中等", "高（但会遗忘）"],
        ["信息密度", "极高压缩", "聚焦决策点", "保留推理链"],
        ["载体", "论文、slides", "讨论记录、邮件", "notebook、笔记"],
    ],
    header_color="primary",
)
sb.layout(tbl)

# ============================================================
# Slide 5: APR 例子 — 图标标题 + 裸排文字 + 细线分隔
# ============================================================
sb = prs.slide(title="例子：基于LLM的自动程序修复")

ico_f2 = SvgImage(svg=icon_flask(), width=0.6, height=0.6)
ico_b2 = SvgImage(svg=icon_bubbles(), width=0.6, height=0.6)
ico_s2 = SvgImage(svg=icon_stage(), width=0.6, height=0.6)

e_col = VStack(
    gap=0.15,
    children=[
        dot_label("探索层", "secondary"),
        BulletList(
            items=[
                "尝试3种prompt策略",
                "few-shot示例选择影响大",
                "调tokenizer截断花两天",
                "CoT对复杂bug引入噪声",
            ],
            font_size="caption",
        ),
    ],
)

line1 = Line(from_component=e_col, to_component=e_col, color="border", dashed=True)

c_col = VStack(
    gap=0.15,
    children=[
        dot_label("协作层", "accent"),
        BulletList(
            items=[
                "few-shot最优，关键在示例选择",
                "CoT对复杂bug帮助有限",
                "下一步：自动选择相关示例",
            ],
            font_size="caption",
        ),
    ],
)

p_col = VStack(
    gap=0.15,
    children=[
        dot_label("呈现层", "primary"),
        BulletList(
            items=[
                "提出检索式few-shot APR框架",
                "两个benchmark超过SOTA",
            ],
            font_size="caption",
        ),
    ],
)

arrow1 = Arrow(from_component=e_col, to_component=c_col, color="text_light")
arrow2 = Arrow(from_component=c_col, to_component=p_col, color="text_light")

sb.layout(HStack(gap=0.12, children=[e_col, arrow1, c_col, arrow2, p_col]))
sb.animate(
    [
        [e_col],
        [arrow1, c_col],
        [arrow2, p_col],
    ]
)

# ============================================================
# Slide 6: 用模型解释现象
# ============================================================
sb = prs.slide(title="摩擦力 = 层级错配")
tbl = Table(
    headers=["现象", "层级错配分析"],
    rows=[
        ["论文 ≠ 真实过程", "呈现层是对探索层的极度压缩"],
        ["初学者对话困难", "探索层不扎实 → 协作层无料"],
        ["提问缺乏context", "只给呈现层描述，未暴露协作层信息"],
        ["组会抓不住重点", "探索层内容搬到呈现层场合"],
    ],
    header_color="negative",
)
point = Callout(
    title="本质",
    body="信息的层级与接收方的期待不匹配",
    color="negative",
)
sb.layout(VStack(gap=0.3, children=[tbl, point]))
sb.animate([[tbl], [point]])

# ============================================================
# Slide 7: 过渡
# ============================================================
prs.transition(
    text="元能力：在三层之间自由切换",
    sub_text="正向构建 × 逆向解构",
)

# ============================================================
# Slide 8: 正向与逆向 — 穿梭图标
# ============================================================
sb = prs.slide(title="两个方向的层间切换")

ico_shuttle_lg = SvgImage(svg=icon_shuttle(), width=1.7, height=1.7)

fwd_label = TextBlock(
    text="正向构建：从探索到呈现", font_size="heading", bold=True, color="text"
)
fwd = Flow(
    labels=["探索层", "协作层", "呈现层"],
    colors=["secondary", "accent", "primary"],
    arrow_color="text_light",
    direction="horizontal",
)

rev_label = TextBlock(
    text="逆向解构：从呈现到探索", font_size="heading", bold=True, color="text"
)
rev = Flow(
    labels=["呈现层", "协作层", "探索层"],
    colors=["primary", "accent", "secondary"],
    arrow_color="text_light",
    direction="horizontal",
)

flows = VStack(gap=0.22, width=9.2, children=[fwd_label, fwd, rev_label, rev])

sb.layout(HStack(gap=0.28, align="center", children=[flows, ico_shuttle_lg]))
sb.animate(
    [
        [fwd_label, fwd],
        [rev_label, rev],
        [ico_shuttle_lg],
    ]
)

# ============================================================
# Slide 9: 两次压缩 — Callout 替代 RoundedBox
# ============================================================
sb = prs.slide(title="正向构建：两次压缩")

compress1 = Callout(
    title="探索 → 协作",
    body="提炼值得讨论的发现\n和需要判断的决策点",
    color="secondary",
)
compress2 = Callout(
    title="协作 → 呈现",
    body="构建对外部读者\n自洽的叙事线",
    color="primary",
)

arrow = Arrow(from_component=compress1, to_component=compress2, color="text_light")

note = TextBlock(
    text="不需要列举所有尝试，但需要说明判断依据",
    font_size="body",
    italic=True,
    color="text_mid",
)

sb.layout(
    VStack(
        gap=0.3,
        children=[
            HStack(gap=0.3, children=[compress1, arrow, compress2]),
            note,
        ],
    )
)
sb.animate(
    [
        [compress1],
        [arrow, compress2],
        [note],
    ]
)

# ============================================================
# Slide 10: 逆向解构
# ============================================================
sb = prs.slide(title="逆向解构：透过论文还原过程")

example_text = TextBlock(
    text='论文："覆盖5种语言，1000个问题，评估8个LLM"',
    font_size="body",
    italic=True,
    color="text_mid",
)

questions = VStack(
    gap=0.15,
    children=[
        TextBlock(
            text="为什么是这5种语言？可能其他语言数据收集困难",
            font_size="caption",
            color="text",
        ),
        TextBlock(
            text="1000个问题如何筛选？初始收集规模可能远大于此",
            font_size="caption",
            color="text",
        ),
        TextBlock(
            text="8个模型的选择标准？可能受限于API访问或算力",
            font_size="caption",
            color="text",
        ),
    ],
)

benefit = Callout(
    title="收益",
    body="更准确评估贡献 · 更好判断边界 · 更高效改进",
    color="primary",
)

sb.layout(VStack(gap=0.3, children=[example_text, questions, benefit]))
sb.animate(
    [
        [example_text],
        [questions],
        [benefit],
    ]
)

# ============================================================
# Slide 11: 初学者 vs 熟练研究者
# ============================================================
prs.comparison(
    title="熟练度的差异",
    left=(
        "初学者",
        [
            "探索层：零散、无记录",
            "协作层：全倒或太少",
            "呈现层：流水账或空洞",
            "读论文：只看到呈现层",
        ],
    ),
    right=(
        "熟练研究者",
        [
            "探索层：结构化记录",
            "协作层：精准高效",
            "呈现层：清晰有力",
            "读论文：读出弦外之音",
        ],
    ),
)

# ============================================================
# Slide 12: 过渡
# ============================================================
prs.transition(
    text="实践建议",
    sub_text="从探索层到呈现层",
)

# ============================================================
# Slide 13: 维护探索层 — 大图标 + 文字
# ============================================================
sb = prs.slide(title="维护探索层")

ico_f3 = SvgImage(svg=icon_flask(), width=2.0, height=2.0)

principle = Callout(
    title="核心原则",
    body='以"两周后的自己能看懂"为标准记录',
    color="secondary",
)

template = BulletList(
    items=[
        "做了什么：Defects4J + BM25检索few-shot",
        "结果：plausible patch 18%→27%",
        "分析：相似示例帮编译不帮语义",
        "下一步：尝试AST结构相似性",
    ],
    font_size="caption",
)

key_point = TextBlock(
    text="关键：记录推理链，不只是记录操作",
    font_size="body",
    bold=True,
    color="accent",
)

left_content = VStack(gap=0.2, children=[principle, template, key_point])

sb.layout(HStack(gap=0.4, children=[left_content, ico_f3]))
sb.animate(
    [
        [principle, ico_f3],
        [template],
        [key_point],
    ]
)

# ============================================================
# Slide 14: 协作层粒度 — 图标 + 对比
# ============================================================
sb = prs.slide(title="在协作层找到合适的粒度")

ico_b3 = SvgImage(svg=icon_bubbles(), width=1.8, height=1.8)

checklist = BulletList(
    items=[
        "需要对方帮助判断什么？",
        "对方需要哪些信息才能判断？",
        "哪些是自己应处理的执行细节？",
    ],
    font_size="body",
)

bad = Callout(
    title="信息过载",
    body="试了BM25、TF-IDF、CodeBERT\ntop-k试了1、3、5、10……",
    color="negative",
)
good = Callout(
    title="聚焦决策",
    body="三种策略对比，BM25和CodeBERT接近\n倾向BM25，是否值得混合方案？",
    color="accent",
)

sb.layout(
    VStack(
        gap=0.25,
        children=[
            HStack(gap=0.4, children=[checklist, ico_b3]),
            HStack(gap=0.3, children=[bad, good]),
        ],
    )
)
sb.animate(
    [
        [checklist, ico_b3],
        [bad, good],
    ]
)

# ============================================================
# Slide 15: 构建呈现层 + 逆向解构
# ============================================================
sb = prs.slide(title="呈现层构建 与 逆向解构练习")

left_col = VStack(
    gap=0.15,
    children=[
        dot_label("构建呈现层", "primary"),
        BulletList(
            items=[
                "先结论后过程",
                "只保留支撑结论的证据",
                "时间线重构是正常的",
            ],
            font_size="body",
        ),
    ],
)

right_col = VStack(
    gap=0.15,
    children=[
        dot_label("逆向解构", "secondary"),
        BulletList(
            items=[
                "作者呈现了什么？",
                "为什么是这个叙事？",
                "哪些实验没写进论文？",
            ],
            font_size="body",
        ),
    ],
)

sb.layout(HStack(gap=0.5, children=[left_col, right_col]))
sb.animate([[left_col], [right_col]])

# ============================================================
# Slide 16: AI时代 — 图标对比主导
# ============================================================
sb = prs.slide(title="AI时代下的三层模型")

ico_robot_lg = SvgImage(svg=icon_robot(), width=1.3, height=1.3)
ico_human_lg = SvgImage(svg=icon_human(), width=1.3, height=1.3)

ai_side = VStack(
    gap=0.15,
    children=[
        ico_robot_lg,
        TextBlock(text="AI可辅助", font_size="heading", bold=True, color="text"),
        TextBlock(
            text="探索层的执行细节\n呈现层的格式范式",
            font_size="caption",
            color="text_mid",
        ),
    ],
)

human_side = VStack(
    gap=0.15,
    children=[
        ico_human_lg,
        TextBlock(text="人不可替代", font_size="heading", bold=True, color="text"),
        TextBlock(
            text="协作层的信息凝练\n层间穿梭的判断力",
            font_size="caption",
            color="text_mid",
        ),
    ],
)

reason = Callout(
    title="为什么协作层不能交给AI？",
    body="从探索到协作的压缩，本质上是在验证\n自己是否真正理解了所做的事情",
    color="negative",
)

sb.layout(
    VStack(
        gap=0.3,
        children=[
            HStack(gap=0.8, children=[ai_side, human_side]),
            reason,
        ],
    )
)
sb.animate(
    [
        [ai_side, human_side],
        [reason],
    ]
)

# ============================================================
# Slide 17: 核心能力 — 穿梭图标主导
# ============================================================
sb = prs.slide(title="人的核心能力")

ico_shuttle2 = SvgImage(svg=icon_shuttle(), width=2.8, height=2.8)

desc = VStack(
    gap=0.25,
    children=[
        Callout(
            title="层间穿梭",
            body="正向：从探索中提炼结构化理解\n逆向：从呈现中还原决策过程",
            color="primary",
        ),
        TextBlock(
            text="工具会不断迭代\n但对信息结构的理解与掌控不会过时",
            font_size="body",
            italic=True,
            color="text_mid",
        ),
    ],
)

sb.layout(HStack(gap=0.5, children=[desc, ico_shuttle2]))
sb.animate([[ico_shuttle2], [desc]])

# ============================================================
# Slide 18: 术与道 — 太极图标居中
# ============================================================
sb = prs.slide(title="术与道")

ico_taiji = SvgImage(svg=icon_dao(), width=2.0, height=2.0)

dao_col = VStack(
    gap=0.1,
    children=[
        dot_label("道", "primary"),
        BulletList(
            items=[
                "理解信息的三层结构",
                "掌握层间穿梭的能力",
                "判断何时在哪层操作",
            ],
            font_size="body",
        ),
    ],
)

shu_col = VStack(
    gap=0.1,
    children=[
        dot_label("术", "text_mid"),
        BulletList(
            items=[
                "实验记录模板与工具",
                "组会汇报准备流程",
                "论文写作结构范式",
                "AI工具使用技巧",
            ],
            font_size="body",
        ),
    ],
)

callout = Callout(
    title="关键区分",
    body='术是"怎么做"，道是"为什么这么做"\n术随工具更新，道不会过时',
    color="accent",
)

sb.layout(
    VStack(
        gap=0.25,
        children=[
            HStack(gap=0.3, children=[dao_col, ico_taiji, shu_col]),
            callout,
        ],
    )
)
sb.animate(
    [
        [dao_col, ico_taiji, shu_col],
        [callout],
    ]
)

# ============================================================
# Slide 19: 总结
# ============================================================
sb = prs.slide(title="小结")

summary = Flowchart(
    nodes={
        "root": ("沟通摩擦的根源\n层级与受众不匹配", "primary"),
        "understand": ("理解三层结构", "accent"),
        "practice": ("练习层间切换", "secondary"),
        "distinguish": ("区分术与道", "text_mid"),
    },
    edges=[
        ("root", "understand"),
        ("root", "practice"),
        ("root", "distinguish"),
    ],
    direction="down",
)

sb.layout(summary)

# ============================================================
# Slide 20: 结尾
# ============================================================
prs.end(
    title="谢谢",
    subtitle="层间穿梭是值得刻意练习的研究元能力",
)

# ============================================================
# 保存 & 预览
# ============================================================
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "communication_friction.pptx")
prs.save(output_path)
print(f"Saved to {output_path}")

report = prs.review()
print(f"Slides: {report['total_slides']}, Issues: {report['total_issues']}")
for iss in report["issues"]:
    print(f"  [{iss['type']}] {iss['detail']}")

preview_dir = os.path.join(output_dir, "preview")
os.makedirs(preview_dir, exist_ok=True)
paths = prs.preview(output_dir=preview_dir)
print(f"Preview: {len(paths)} slides rendered")
for p in paths:
    print(f"  {p}")
