"""减少沟通摩擦：信息的三层结构 — PPT 生成脚本."""

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

# 主题：professional + 仿宋字体
theme = themes.professional.override(font_family="FangSong")
prs = Presentation(theme=theme)


# ============================================================
# SVG 图标工厂
# ============================================================


def icon_flask(th):
    """探索层图标 — 实验烧瓶，内有气泡."""
    s = SvgCanvas(120, 120, theme=th)
    # 瓶身（梯形）
    s.polygon(
        [(38, 40), (82, 40), (98, 100), (22, 100)],
        fill="secondary",
        stroke="none",
        opacity=0.85,
    )
    # 瓶口
    s.rect(48, 20, 24, 22, fill="secondary", stroke="none", rx=3)
    # 瓶口高光
    s.rect(52, 20, 16, 4, fill="white", stroke="none", rx=2, opacity=0.5)
    # 液面分界线
    s.line(30, 70, 90, 70, color="white", width=1, dashed=True)
    # 气泡
    s.circle(50, 80, 5, fill="white", text_color="white", opacity=0.5)
    s.circle(68, 75, 3, fill="white", text_color="white", opacity=0.4)
    s.circle(55, 88, 4, fill="white", text_color="white", opacity=0.45)
    # 底部圆角
    s.rect(22, 96, 76, 6, fill="secondary", stroke="none", rx=3, opacity=0.9)
    return s


def icon_bubbles(th):
    """协作层图标 — 两个重叠的对话气泡."""
    s = SvgCanvas(120, 120, theme=th)
    # 后方气泡（略大，偏右上）
    s.rounded_rect(
        35,
        15,
        70,
        50,
        fill="accent",
        stroke="none",
        rx=18,
        opacity=0.5,
        text="",
        font_size=1,
    )
    # 后方气泡尾巴
    s.polygon([(85, 60), (95, 75), (75, 58)], fill="accent", stroke="none", opacity=0.5)
    # 前方气泡（偏左下）
    s.rounded_rect(
        15,
        40,
        70,
        50,
        fill="accent",
        stroke="none",
        rx=18,
        opacity=0.85,
        text="",
        font_size=1,
    )
    # 前方气泡尾巴
    s.polygon(
        [(35, 85), (25, 100), (45, 83)], fill="accent", stroke="none", opacity=0.85
    )
    # 气泡内的省略号（模拟对话）
    s.circle(35, 63, 4, fill="white", text_color="white", opacity=0.7)
    s.circle(50, 63, 4, fill="white", text_color="white", opacity=0.7)
    s.circle(65, 63, 4, fill="white", text_color="white", opacity=0.7)
    return s


def icon_stage(th):
    """呈现层图标 — 聚光灯+讲台."""
    s = SvgCanvas(120, 120, theme=th)
    # 聚光灯光束（梯形，从上到下扩散）
    s.polygon(
        [(50, 10), (70, 10), (95, 75), (25, 75)],
        fill="primary",
        stroke="none",
        opacity=0.18,
    )
    # 聚光灯头
    s.rounded_rect(
        46,
        5,
        28,
        16,
        fill="primary",
        stroke="none",
        rx=4,
        text="",
        font_size=1,
        opacity=0.9,
    )
    # 讲台面
    s.polygon(
        [(20, 78), (100, 78), (105, 90), (15, 90)],
        fill="primary",
        stroke="none",
        opacity=0.85,
    )
    # 讲台腿
    s.rect(35, 90, 50, 20, fill="primary", stroke="none", rx=2, opacity=0.7)
    # 讲台高光线
    s.line(40, 83, 80, 83, color="white", width=1.5)
    return s


def icon_lightning(th):
    """摩擦图标 — 闪电."""
    s = SvgCanvas(120, 120, theme=th)
    s.polygon(
        [(65, 8), (35, 55), (55, 55), (42, 112), (90, 50), (65, 50)],
        fill="negative",
        stroke="none",
        opacity=0.9,
    )
    # 内部高光
    s.polygon(
        [(63, 22), (48, 52), (58, 52), (50, 90), (78, 50), (63, 50)],
        fill="white",
        stroke="none",
        opacity=0.2,
    )
    return s


def icon_shuttle(th):
    """层间穿梭图标 — 双向环形箭头."""
    s = SvgCanvas(120, 120, theme=th)
    # 上弧 (顺时针箭头，从左到右)
    s.path(
        "M 30,50 A 30,30 0 0,1 90,50",
        stroke="primary",
        fill="none",
        stroke_width=5,
    )
    # 上弧箭头头
    s.polygon([(90, 42), (100, 50), (90, 58)], fill="primary", stroke="none")
    # 下弧 (顺时针箭头，从右到左)
    s.path(
        "M 90,70 A 30,30 0 0,1 30,70",
        stroke="secondary",
        fill="none",
        stroke_width=5,
    )
    # 下弧箭头头
    s.polygon([(30, 62), (20, 70), (30, 78)], fill="secondary", stroke="none")
    return s


def icon_robot(th):
    """AI图标 — 机器人头."""
    s = SvgCanvas(120, 120, theme=th)
    # 天线
    s.line(60, 8, 60, 28, color="text_mid", width=3)
    s.circle(60, 8, 5, fill="text_mid", text_color="text_mid")
    # 头部
    s.rounded_rect(
        25,
        28,
        70,
        60,
        fill="text_mid",
        stroke="none",
        rx=14,
        text="",
        font_size=1,
        opacity=0.85,
    )
    # 眼睛
    s.circle(45, 52, 8, fill="white", text_color="white")
    s.circle(75, 52, 8, fill="white", text_color="white")
    # 瞳孔
    s.circle(45, 52, 4, fill="primary", text_color="primary")
    s.circle(75, 52, 4, fill="primary", text_color="primary")
    # 嘴巴
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
    # 嘴巴格子线
    s.line(51, 70, 51, 78, color="text_mid", width=1)
    s.line(60, 70, 60, 78, color="text_mid", width=1)
    s.line(69, 70, 69, 78, color="text_mid", width=1)
    # 耳朵
    s.rect(15, 45, 10, 20, fill="text_mid", stroke="none", rx=3, opacity=0.7)
    s.rect(95, 45, 10, 20, fill="text_mid", stroke="none", rx=3, opacity=0.7)
    return s


def icon_human(th):
    """人图标 — 简笔人形."""
    s = SvgCanvas(120, 120, theme=th)
    # 头
    s.circle(60, 25, 16, fill="primary", text_color="primary", opacity=0.85)
    # 身体
    s.rounded_rect(
        38,
        46,
        44,
        45,
        fill="primary",
        stroke="none",
        rx=10,
        text="",
        font_size=1,
        opacity=0.85,
    )
    # 领口 V 形
    s.polygon([(50, 46), (60, 60), (70, 46)], fill="white", stroke="none", opacity=0.3)
    # 手臂
    s.rounded_rect(
        18,
        50,
        20,
        10,
        fill="primary",
        stroke="none",
        rx=5,
        text="",
        font_size=1,
        opacity=0.7,
    )
    s.rounded_rect(
        82,
        50,
        20,
        10,
        fill="primary",
        stroke="none",
        rx=5,
        text="",
        font_size=1,
        opacity=0.7,
    )
    return s


def icon_dao(th):
    """道图标 — 太极/阴阳简化."""
    s = SvgCanvas(120, 120, theme=th)
    # 外圈
    s.circle(60, 60, 45, fill="primary", text_color="primary", opacity=0.15)
    s.circle(60, 60, 45, fill="none", text_color="none")
    s.path(
        "M 60,15 A 45,45 0 0,1 60,105", stroke="primary", fill="none", stroke_width=3
    )
    s.path(
        "M 60,105 A 45,45 0 0,1 60,15", stroke="secondary", fill="none", stroke_width=3
    )
    # 内部 S 曲线
    s.path(
        "M 60,15 A 22.5,22.5 0 0,1 60,60 A 22.5,22.5 0 0,0 60,105",
        stroke="text_mid",
        fill="none",
        stroke_width=2,
    )
    # 阴阳点
    s.circle(60, 37, 6, fill="secondary", text_color="secondary")
    s.circle(60, 83, 6, fill="primary", text_color="primary")
    return s


# ============================================================
# Slide 1: 封面
# ============================================================
prs.cover(
    title="减少沟通摩擦",
    subtitle="信息的三层结构与研究元能力",
)

# ============================================================
# Slide 2: 四个常见现象
# ============================================================
sb = prs.slide(title="这些场景是否似曾相识？")

lightning = SvgImage(svg=icon_lightning(theme), width=1.0, height=1.0)

s1 = RoundedBox(
    text="论文读起来逻辑清晰\n但实际过程完全不同",
    color="bg_alt",
    text_color="text",
    font_size="caption",
)
s2 = RoundedBox(
    text="初学者与高年级\n对话总是费劲",
    color="bg_alt",
    text_color="text",
    font_size="caption",
)
s3 = RoundedBox(
    text='提问只说"不work"\n得不到有效回复',
    color="bg_alt",
    text_color="text",
    font_size="caption",
)
s4 = RoundedBox(
    text="组会讲了20分钟细节\n听众不知道重点",
    color="bg_alt",
    text_color="text",
    font_size="caption",
)
conclusion = Callout(
    title="共同根源",
    body="不是能力问题，不是态度问题\n而是信息没有匹配接收方的 context 容量",
    color="primary",
)
sb.layout(
    VStack(
        gap=0.25,
        children=[
            HStack(
                gap=0.2,
                children=[
                    Grid(cols=2, gap=0.2, children=[s1, s2, s3, s4]),
                    lightning,
                ],
            ),
            conclusion,
        ],
    )
)
sb.animate(
    [
        [s1, s2, s3, s4],
        [lightning],
        [conclusion],
    ]
)

# ============================================================
# Slide 3: 三层模型总览 — 带图标
# ============================================================
sb = prs.slide(title="模型：信息的三层结构")

ico_flask = SvgImage(svg=icon_flask(theme), width=1.2, height=1.2)
ico_bubbles = SvgImage(svg=icon_bubbles(theme), width=1.2, height=1.2)
ico_stage = SvgImage(svg=icon_stage(theme), width=1.2, height=1.2)

col_explore = VStack(
    gap=0.1,
    children=[
        ico_flask,
        RoundedBox(
            text="探索层\nNotebook / 个人笔记",
            color="secondary",
            text_color="white",
            font_size="caption",
        ),
    ],
)
col_collab = VStack(
    gap=0.1,
    children=[
        ico_bubbles,
        RoundedBox(
            text="协作层\n组会 / 合作者讨论",
            color="accent",
            text_color="white",
            font_size="caption",
        ),
    ],
)
col_present = VStack(
    gap=0.1,
    children=[
        ico_stage,
        RoundedBox(
            text="呈现层\n论文 / Slides / Poster",
            color="primary",
            text_color="white",
            font_size="caption",
        ),
    ],
)

a1 = Arrow(from_component=col_explore, to_component=col_collab, color="text_light")
a2 = Arrow(from_component=col_collab, to_component=col_present, color="text_light")

desc = TextBlock(
    text="每层有不同的受众、注意力容量、信息密度要求",
    font_size="body",
    color="text_mid",
    align="center",
)

sb.layout(
    VStack(
        gap=0.3,
        children=[
            HStack(gap=0.15, children=[col_explore, a1, col_collab, a2, col_present]),
            desc,
        ],
    )
)
sb.animate(
    [
        [col_explore],
        [a1, col_collab],
        [a2, col_present],
        [desc],
    ]
)

# ============================================================
# Slide 4: 每层特征对比
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
# Slide 5: 具体例子 — LLM APR（带图标）
# ============================================================
sb = prs.slide(title="例子：基于LLM的自动程序修复")

ico_f2 = SvgImage(svg=icon_flask(theme), width=0.7, height=0.7)
ico_b2 = SvgImage(svg=icon_bubbles(theme), width=0.7, height=0.7)
ico_s2 = SvgImage(svg=icon_stage(theme), width=0.7, height=0.7)

e_col = VStack(
    gap=0.1,
    children=[
        HStack(gap=0.1, children=[ico_f2, Badge(text="探索层", color="secondary")]),
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

c_col = VStack(
    gap=0.1,
    children=[
        HStack(gap=0.1, children=[ico_b2, Badge(text="协作层", color="accent")]),
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
    gap=0.1,
    children=[
        HStack(gap=0.1, children=[ico_s2, Badge(text="呈现层", color="primary")]),
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
# Slide 7: 过渡 — 元能力
# ============================================================
prs.transition(
    text="元能力：在三层之间自由切换",
    sub_text="正向构建 × 逆向解构",
)

# ============================================================
# Slide 8: 正向与逆向 — 带穿梭图标
# ============================================================
sb = prs.slide(title="两个方向的层间切换")

ico_shuttle = SvgImage(svg=icon_shuttle(theme), width=1.5, height=1.5)

fwd_label = TextBlock(
    text="正向构建：从探索到呈现",
    font_size="heading",
    bold=True,
    color="primary",
)
fwd = Flow(
    labels=["探索层", "协作层", "呈现层"],
    colors=["secondary", "accent", "primary"],
    arrow_color="text_mid",
    direction="horizontal",
)

rev_label = TextBlock(
    text="逆向解构：从呈现到探索",
    font_size="heading",
    bold=True,
    color="secondary",
)
rev = Flow(
    labels=["呈现层", "协作层", "探索层"],
    colors=["primary", "accent", "secondary"],
    arrow_color="text_mid",
    direction="horizontal",
)

flows = VStack(gap=0.25, children=[fwd_label, fwd, rev_label, rev])

sb.layout(HStack(gap=0.3, children=[flows, ico_shuttle]))
sb.animate(
    [
        [fwd_label, fwd],
        [rev_label, rev],
        [ico_shuttle],
    ]
)

# ============================================================
# Slide 9: 正向构建 — 两次压缩
# ============================================================
sb = prs.slide(title="正向构建：两次压缩")

compress1 = VStack(
    gap=0.2,
    children=[
        RoundedBox(
            text="探索 → 协作", color="secondary", text_color="white", font_size="body"
        ),
        TextBlock(
            text="提炼值得讨论的发现\n和需要判断的决策点",
            font_size="caption",
            color="text_mid",
        ),
    ],
)
compress2 = VStack(
    gap=0.2,
    children=[
        RoundedBox(
            text="协作 → 呈现", color="primary", text_color="white", font_size="body"
        ),
        TextBlock(
            text="构建对外部读者\n自洽的叙事线", font_size="caption", color="text_mid"
        ),
    ],
)

arrow = Arrow(from_component=compress1, to_component=compress2, color="text_light")

note = Callout(
    title="关键",
    body="不需要列举所有尝试，但需要说明判断依据",
    color="accent",
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

q1 = RoundedBox(
    text="为什么是这5种语言？", color="bg_alt", text_color="text", font_size="caption"
)
q2 = RoundedBox(
    text="1000题如何筛选？", color="bg_alt", text_color="text", font_size="caption"
)
q3 = RoundedBox(
    text="8个模型选择标准？", color="bg_alt", text_color="text", font_size="caption"
)

benefit = Callout(
    title="收益",
    body="更准确评估贡献 · 更好判断边界 · 更高效改进",
    color="primary",
)

sb.layout(
    VStack(
        gap=0.25,
        children=[
            example_text,
            HStack(gap=0.3, children=[q1, q2, q3]),
            benefit,
        ],
    )
)
sb.animate(
    [
        [example_text],
        [q1, q2, q3],
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
# Slide 12: 过渡 — 实践
# ============================================================
prs.transition(
    text="实践建议",
    sub_text="从探索层到呈现层",
)

# ============================================================
# Slide 13: 维护探索层 — 带烧瓶图标
# ============================================================
sb = prs.slide(title="维护探索层")

ico_f3 = SvgImage(svg=icon_flask(theme), width=1.5, height=1.5)

principle = Callout(
    title="核心原则",
    body='以"两周后的自己能看懂"为标准记录',
    color="primary",
)

template = VStack(
    gap=0.1,
    children=[
        Badge(text="实验记录模板", color="text_mid"),
        BulletList(
            items=[
                "做了什么：Defects4J + BM25检索few-shot",
                "结果：plausible patch 18%→27%",
                "分析：相似示例帮编译不帮语义",
                "下一步：尝试AST结构相似性",
            ],
            font_size="caption",
        ),
    ],
)

key_point = TextBlock(
    text="关键：记录推理链，不只是记录操作",
    font_size="body",
    bold=True,
    color="accent",
)

left_content = VStack(gap=0.25, children=[principle, template, key_point])

sb.layout(HStack(gap=0.3, children=[left_content, ico_f3]))
sb.animate(
    [
        [principle],
        [template, ico_f3],
        [key_point],
    ]
)

# ============================================================
# Slide 14: 协作层粒度 — 带对话气泡图标
# ============================================================
sb = prs.slide(title="在协作层找到合适的粒度")

ico_b3 = SvgImage(svg=icon_bubbles(theme), width=1.3, height=1.3)

checklist = BulletList(
    items=[
        "需要对方帮助判断什么？",
        "对方需要哪些信息才能判断？",
        "哪些是自己应处理的执行细节？",
    ],
    font_size="body",
)

bad = VStack(
    gap=0.1,
    children=[
        Badge(text="信息过载", color="negative"),
        BulletList(
            items=[
                "试了BM25、TF-IDF、CodeBERT",
                "top-k试了1、3、5、10",
                "CodeBERT遇到OOM……",
            ],
            font_size="caption",
        ),
    ],
)
good = VStack(
    gap=0.1,
    children=[
        Badge(text="聚焦决策", color="positive"),
        BulletList(
            items=[
                "三种策略对比，BM25和CodeBERT接近",
                "计算开销差异大，倾向BM25",
                "是否值得混合方案？",
            ],
            font_size="caption",
        ),
    ],
)

sb.layout(
    VStack(
        gap=0.25,
        children=[
            HStack(gap=0.3, children=[checklist, ico_b3]),
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
# Slide 15: 构建呈现层 + 逆向解构练习
# ============================================================
sb = prs.slide(title="呈现层构建 与 逆向解构练习")

left_col = VStack(
    gap=0.2,
    children=[
        Badge(text="构建呈现层", color="primary"),
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
    gap=0.2,
    children=[
        Badge(text="逆向解构", color="secondary"),
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
# Slide 16: AI时代 — 带机器人和人形图标
# ============================================================
sb = prs.slide(title="AI时代下的三层模型")

ico_robot = SvgImage(svg=icon_robot(theme), width=1.0, height=1.0)
ico_person = SvgImage(svg=icon_human(theme), width=1.0, height=1.0)

ai_explore = VStack(
    gap=0.1,
    children=[
        ico_robot,
        RoundedBox(
            text="探索层 — AI可辅助\n执行、代码、记录",
            color="secondary",
            text_color="white",
            font_size="caption",
        ),
    ],
)
ai_collab = VStack(
    gap=0.1,
    children=[
        ico_person,
        RoundedBox(
            text="协作层 — 不宜AI代劳\n凝练过程即核心价值",
            color="negative",
            text_color="white",
            font_size="caption",
        ),
    ],
)
ai_present = VStack(
    gap=0.1,
    children=[
        ico_robot,
        RoundedBox(
            text="呈现层 — AI可辅助\n范式固定、执行可外包",
            color="primary",
            text_color="white",
            font_size="caption",
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
        gap=0.25,
        children=[
            HStack(gap=0.2, children=[ai_explore, ai_collab, ai_present]),
            reason,
        ],
    )
)
sb.animate(
    [
        [ai_explore, ai_collab, ai_present],
        [reason],
    ]
)

# ============================================================
# Slide 17: 人的核心能力 — 带穿梭图标
# ============================================================
sb = prs.slide(title="人的核心能力")

ico_shuttle2 = SvgImage(svg=icon_shuttle(theme), width=2.5, height=2.5)

left_desc = VStack(
    gap=0.3,
    children=[
        Callout(
            title="层间穿梭",
            body="正向：从探索中提炼结构化理解\n" "逆向：从呈现中还原决策过程",
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

sb.layout(HStack(gap=0.4, children=[left_desc, ico_shuttle2]))
sb.animate([[ico_shuttle2], [left_desc]])

# ============================================================
# Slide 18: 术与道 — 带太极图标
# ============================================================
sb = prs.slide(title="术与道")

ico_taiji = SvgImage(svg=icon_dao(theme), width=1.8, height=1.8)

dao_col = VStack(
    gap=0.15,
    children=[
        Badge(text="道", color="primary"),
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
    gap=0.15,
    children=[
        Badge(text="术", color="text_mid"),
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
