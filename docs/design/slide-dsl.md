# paperops.slides DSL 设计文档

| 字段 | 值 |
|---|---|
| 文档版本 | 0.1 (draft) |
| 创建日期 | 2026-04-23 |
| 状态 | Design — awaiting review |
| 作用范围 | `paperops.slides` 的内容/样式分层改造与 LLM-facing DSL |

---

## 1. 背景与动机

### 1.1 现状问题

`paperops.slides` 已提供 Flex/Grid/Absolute/Layer/Padding 布局原语、Theme 语义色 + 命名字号、基础组件库（Box/RoundedBox/Badge/Arrow/Text/BulletList/Table/Image）和动画模块。骨架完备，但在实际使用（HKU talk 等）中暴露四个痛点：

1. **风格单一**：同一 Theme 下生成的 deck 视觉高度同质化，很难做出"学术 seminar / keynote / pitch / whitepaper"这种风格差异
2. **对齐弱**：元素间缺乏统一基线，视觉散乱
3. **文字超出**：无自动缩字号/分页机制，排版容易溢出
4. **形状单一**：原子组件覆盖度低，缺卡片/指标/引文/时间节点/流程步等**语义组件**

根因诊断：**缺 CSS 式的"样式抽象层"**。目前所有样式都 inline 写在组件构造参数里，等价于 1996 年的 HTML——有 `<div>` 和 `<span>` 但没有 CSS。换 theme 不够，需要换**stylesheet**；换结构不够，需要换**semantic components**。

### 1.2 核心论断

借鉴 HTML / CSS / MDX 的成熟分层，**但不引入它们的运行时**：

- **Content（HTML-like）**：只写语义结构与类名，不写具体样式
- **Style（CSS-like）**：选择器 + 层叠 + 设计 token，支持外置样式表
- **Renderer**：规则式布局引擎（Flex/Grid/Stack），输出原生可编辑 PPTX

借的是**词汇和分层模型**，不是 CSS/HTML 规范实现。

### 1.3 项目范围（v1.0）

**重构策略**：项目处于 pre-1.0 阶段，允许 breaking change。本次改造**不保留任何向后兼容**——现有 Python builder API、组件构造参数、样式传递方式均可自由重塑。所有 `examples/` 下的样例将作为最后一步统一改写到新 API。这样换来的好处是：
- API surface 可以从零设计为 IR-first，不背历史包袱
- 命名可以统一（CSS 对齐的词汇）
- 模块切分可以按新架构清晰重划，不用做适配层

纳入：
- 完整 JSON IR schema（canonical intermediate representation）
- Theme token 扩展（colors / fonts / spacing / radius / shadow / duration / density）
- StyleSheet + 选择器 + cascade 求解
- Semantic components library（Card / Metric / Callout / Quote / KeyPoint / Stepper / Timeline / Figure / Caption / Pullquote / Divider / Prose / Subtitle / Note）
- 四种 authoring 前端：JSON IR / MDX / Markdown / Python builder
- Inline HTML 白名单（仅影响 text run 的标签）
- Autofit 三策略（shrink / reflow / clip / error）
- Animation 作为 style property + groups + stagger
- 预置 sheets（`academic / seminar / keynote / whitepaper / pitch / minimal`）
- JSON Schema 严格验证 + 结构化错误

不纳入：
- 真·HTML/CSS 兼容（见 1.4）
- Cassowary / 线性约束求解
- 自研文本测量引擎（用 PPT 原生 autofit）
- 动态交互（hover/focus/click 状态之外的行为）
- 媒体查询 `@media`（用 deck variant 替代）

### 1.4 非目标（明确拒绝）

| 非目标 | 理由 | 替代 |
|---|---|---|
| 支持 `<div>` / `<section>` / 块级 HTML | PPT 无对应渲染语义；映射必然漂移 | 用 MDX 组件（`<Grid>` / `<Stack>` / `<Card>`） |
| 支持 `display:flex` / `position:absolute` / `float` / `z-index` | 布局由**组件类型**选择，不由 CSS 属性切换 | `<Flex>` / `<Absolute>` / `<Layer>` 组件 |
| `@media` / `:hover` / `transition` | 无对应运行时 | deck variant + `animate` style key |
| Cassowary 约束求解 | CSS 自己都不用；规则式引擎足够 | Flex/Grid 的 grow/shrink/basis |
| PIL / Pillow 文本测量 | 与 PPT 渲染差异 10%+，且不支持 CJK 断行 | `MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE` |
| HTML / CSS 语法兼容 | 期望管理失败成本远大于收益 | MDX 风格语法 + 借词不借义 |

---

## 2. 整体架构

### 2.1 三层模型

```
┌──────────────────────────────────────────────────────────┐
│  Authoring Layer (LLM / Human)                           │
│  ┌────────┐  ┌──────┐  ┌─────────┐  ┌──────────────┐     │
│  │ JSON IR│  │ MDX  │  │ Markdown│  │ Python builder│    │
│  └────┬───┘  └──┬───┘  └────┬────┘  └──────┬───────┘     │
└───────┼─────────┼────────────┼───────────────┼───────────┘
        │         │            │               │
        ▼         ▼            ▼               ▼
┌──────────────────────────────────────────────────────────┐
│  Normalization                                           │
│  All DSLs → canonical JSON IR tree                       │
└───────────────────────┬──────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────┐
│  Style Resolution                                        │
│  Theme tokens + StyleSheet selectors + cascade           │
│  → ComputedStyle per node                                │
└───────────────────────┬──────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────┐
│  Layout Engine (existing paperops.slides.layout)         │
│  Flex / Grid / Stack / Absolute / Layer → Regions        │
└───────────────────────┬──────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────┐
│  PPTX Codegen (existing + animation extension)           │
│  Regions + ComputedStyle → python-pptx shapes            │
└──────────────────────────────────────────────────────────┘
```

### 2.2 五阶段数据流

| 阶段 | 输入 | 处理 | 输出 |
|---|---|---|---|
| **Parse** | JSON/MDX/MD/Python | Schema 验证 + 宏展开 + inline HTML 过滤 | Canonical IR tree |
| **Resolve Style** | IR tree + Theme + StyleSheet | 选择器匹配 + specificity + cascade + inherit | IR tree with `ComputedStyle` on each node |
| **Layout** | Styled IR tree | 现有 engine.compute_layout | Regions per node |
| **Autofit** | Regions + styled nodes | 溢出检测 → shrink/reflow/clip | 修正后的 Regions |
| **Codegen** | Final tree | python-pptx shape creation + OOXML animation | `.pptx` file |

### 2.3 模块分工（新增 vs 现有）

| 模块 | 状态 | 职责 |
|---|---|---|
| `core/theme.py` | **重写** | 重新设计为 token 容器（colors/fonts/spacing/radius/shadow/duration/density） |
| `core/tokens.py` | **新增** | token 解析器：`"md"` → 具体数值；统一所有值解析入口 |
| `style/stylesheet.py` | **新增** | StyleSheet 数据结构 |
| `style/selector.py` | **新增** | CSS 子集选择器解析与匹配 |
| `style/cascade.py` | **新增** | 层叠 + specificity + inherit 求解 |
| `style/computed.py` | **新增** | `ComputedStyle` 数据类 |
| `style/sheets/` | **新增** | 预置 sheets（`academic.py` / `keynote.py` / ...） |
| `ir/schema.py` | **新增** | IR JSON Schema 定义 |
| `ir/node.py` | **新增** | IR node 数据类 |
| `ir/defines.py` | **新增** | `defines` + `$use` 宏展开 |
| `ir/validator.py` | **新增** | 结构化验证 + LLM 可读错误 |
| `dsl/json_loader.py` | **新增** | JSON → IR |
| `dsl/mdx_parser.py` | **新增** | MDX → IR |
| `dsl/markdown_parser.py` | **新增** | Markdown + inline HTML → IR |
| `dsl/inline_html.py` | **新增** | inline HTML 白名单过滤 |
| `dsl/python_builder.py` | **重写** | 从零设计 IR-first 的 Python API；不沿用旧 `Slide()[...]` 签名 |
| `components/` | **重写** | 旧原子组件重构为 IR 注册形式；新增 12 个语义组件 |
| `components/registry.py` | **新增** | 组件注册表（props schema） |
| `layout/engine.py` | **重构** | 接 IR node + ComputedStyle 作为输入（原算法保留） |
| `layout/autofit.py` | **新增** | autofit 四策略；取代现有 `auto_size.py` |
| `layout/baseline.py` | **新增** | 基线网格对齐 |
| `animation.py` | **重构** | 从 style property 收集 AnimationSpec；原 OOXML 写入逻辑保留 |
| `build.py` | **重写** | 五阶段 pipeline 入口 |
| `auto_size.py` | **删除** | 功能并入 `layout/autofit.py` |

---

## 3. Content 层 —— JSON IR

### 3.1 Node schema（六字段）

每个节点严格遵循以下结构：

```json
{
  "type":     "<component name>",    // 必需，注册过的组件名
  "class":    "foo bar baz",         // 可选，空格分隔字符串
  "id":       "<unique id>",         // 可选，deck 内唯一
  "style":    { ... },               // 可选，inline 样式对象
  "props":    { ... },               // 可选，组件私有数据
  "children": [ ... ]                // 可选，子节点数组 | 字符串 | 混合
}
```

**设计决策**：

| 字段 | 决策 | 理由 |
|---|---|---|
| `type` | 用 `type` 而非 `tag` | LLM 在 JSX / tool-call 语料中训练充分 |
| `class` | 空格分隔字符串 | 完全对齐 HTML `className` |
| `id` | 可选，deck 内唯一 | 对应 CSS `#id` 选择器 |
| `style` | 对象，key 用 kebab-case | 对齐 CSS 属性名直觉 |
| `props` | 和 `style` 分开 | 排版样式与组件数据（chart data、table rows）解耦 |
| `children` | 允许字符串混入 | 对齐 JSX：`["Hello ", {"type":"em","text":"world"}]` |

### 3.2 完整文档结构

```json
{
  "$schema": "paperops-slide-1.0",
  "meta": {
    "title":   "Q1 Product Review",
    "author":  "Product Team",
    "date":    "2026-04-01",
    "lang":    "zh-CN",
    "aspect":  "16:9"
  },

  "theme":  "minimal",
  "sheet":  "keynote",

  "styles": {
    ".kpi":                 { "bg": "bg_accent", "radius": "md", "padding": "md" },
    ".kpi .value":          { "font": "title", "color": "primary" },
    ".kpi .delta.positive": { "color": "positive" },
    ".section-intro":       { "padding": "xl", "align": "center" }
  },

  "defines": {
    "KPI": {
      "type": "card", "class": "kpi",
      "children": [
        { "type": "text", "class": "label", "text": "$label" },
        { "type": "text", "class": "value", "text": "$value" },
        { "type": "text", "class": "delta $trend", "text": "$delta" }
      ]
    }
  },

  "slides": [
    {
      "type": "slide", "class": "cover",
      "children": [
        { "type": "title",    "text": "Q1 Product Review" },
        { "type": "subtitle", "text": "Data-driven, iterative" }
      ]
    },
    {
      "type": "slide", "class": "content",
      "children": [
        { "type": "heading", "text": "Key Metrics" },
        {
          "type": "grid",
          "style": { "cols": "1fr 1fr 1fr", "gap": "lg" },
          "children": [
            { "$use": "KPI", "label": "DAU",       "value": "125k", "delta": "+56%", "trend": "positive" },
            { "$use": "KPI", "label": "Retention", "value": "42%",  "delta": "+8pt", "trend": "positive" },
            { "$use": "KPI", "label": "Churn",     "value": "3.2%", "delta": "-1pt", "trend": "positive" }
          ]
        }
      ]
    }
  ]
}
```

**顶层字段**：

| 字段 | 必需 | 说明 |
|---|---|---|
| `$schema` | 是 | schema 版本标识 |
| `meta` | 否 | deck 元数据，写入 PPTX core properties |
| `theme` | 否 | 引用注册 Theme，默认 `minimal` |
| `sheet` | 否 | 引用注册 StyleSheet，默认 `default` |
| `styles` | 否 | deck 局部 stylesheet，**叠加**在 `sheet` 之上 |
| `defines` | 否 | 宏组件定义，仅本 deck 可见 |
| `slides` | 是 | 幻灯片节点数组，每个 `type` 应为 `slide` 或 slide 家族组件 |

### 3.3 children 的字符串混入

```json
{ "type": "p", "children": [
  "我们在 ",
  { "type": "strong", "text": "Q1" },
  " 达成了 ",
  { "type": "em", "text": "125k" },
  " DAU。"
]}
```

规则：
- `children` 中的裸字符串 = `{"type": "text", "text": <string>}`
- `children` 省略时 = 无子节点
- `children` 为单个字符串 = 等价于 `{"text": <string>}`
- 某些组件（如 `text`、`title`）额外支持 `text` 字段作为捷径：`{"type":"title","text":"Hello"}`

### 3.4 defines + $use 宏系统

**定义**：`defines` 是一个命名表，每项是一个带 `$variable` 占位符的 IR 子树。

**引用**：`{"$use": "<name>", "<var>": "<value>", ...}` 展开为 defines 中的子树，`$variable` 被替换。

**规则**：

- 展开在 Parse 阶段**最早**完成，展开后树和手写 IR 完全等价
- 变量支持字符串、数字、数组、对象；对象会递归合并到目标位置
- 变量未提供 → 占位符字符串原样保留（便于调试）→ 严格模式下 error
- `$use` 支持嵌套：被展开的子树里可以再引用其它 `$use`
- `defines` 不支持递归（环检测）
- 变量名 `$CHILDREN` 特殊：接受 `$use` 节点的 children 数组作为插入点

**$CHILDREN 示例**：

```json
"defines": {
  "Callout": {
    "type": "box", "class": "callout",
    "children": [
      { "type": "badge", "text": "$kind" },
      { "type": "prose", "children": "$CHILDREN" }
    ]
  }
},
"slides": [{
  "type": "slide",
  "children": [
    {
      "$use": "Callout",
      "kind": "Insight",
      "children": ["瓶颈从**算力**转移到**环境多样性**。"]
    }
  ]
}]
```

### 3.5 注释与元数据

JSON 本身不支持注释。约定：以 `_` 开头的 key 在 Parse 阶段被忽略，可用作注释。

```json
{
  "type": "slide",
  "_comment": "这页讲清楚 RL scaling 的瓶颈转移",
  "children": [...]
}
```

对机器可读的元数据（讲稿时长、叙事节拍、TTS 标注），使用 `<Note>` 组件而非 `_` 字段，因为它会参与渲染（speaker notes 区）。

### 3.6 最小 IR 示例

```json
{
  "$schema": "paperops-slide-1.0",
  "slides": [
    { "type": "slide", "children": [
      { "type": "title", "text": "Hello World" }
    ]}
  ]
}
```

---

## 4. Style 层 —— Theme + StyleSheet + Cascade

### 4.1 Theme token（扩展后完整定义）

```python
@dataclass
class Theme:
    name: str

    # 已有
    colors:      dict[str, str]         # semantic color → hex
    fonts:       dict[str, float]       # semantic size → pt
    font_family: str
    font_mono:   str

    # 新增 tokens
    spacing:     dict[str, float]       # xs/sm/md/lg/xl/2xl → inches
    radius:      dict[str, float]       # none/sm/md/lg/full → inches
    shadow:      dict[str, ShadowSpec]  # none/sm/md/lg
    duration:    dict[str, float]       # instant/fast/base/slow → seconds
    density:     str                    # compact | normal | airy
    baseline:    float = 0.1            # baseline grid step in inches
```

**Token 值约定**：

| Token | 类型 | 单位 | 示例值（executive theme） |
|---|---|---|---|
| `spacing` | float | inches | xs=0.08, sm=0.16, md=0.32, lg=0.64, xl=1.0, 2xl=1.6 |
| `radius` | float | inches | none=0, sm=0.05, md=0.12, lg=0.24, full=99 |
| `shadow` | struct | — | sm/md/lg，每个含 `(dx, dy, blur, color, opacity)` |
| `duration` | float | seconds | instant=0, fast=0.2, base=0.4, slow=0.8 |
| `density` | enum | — | compact（×0.75）/ normal（×1）/ airy（×1.25）—— 作为 spacing 全局乘子 |
| `baseline` | float | inches | 默认 0.1"，所有 y 坐标 snap 到最近网格线 |

**Token 引用**：所有 style 值可以直接写 token 名（`"md"`）、数值（`0.3`）、或带单位字符串（`"0.3in"` / `"12pt"`）。

### 4.2 StyleSheet 与选择器

StyleSheet 是 `{selector: style_object}` 的有序字典：

```python
StyleSheet({
    "slide":              {"padding": "xl"},
    ".cover":             {"padding": "2xl", "align": "center"},
    ".cover title":       {"font": "title", "color": "primary", "align": "center"},
    ".card":              {"bg": "bg_alt", "radius": "md", "padding": "md"},
    ".card.kpi":          {"bg": "bg_accent", "border-left": ["accent", "md"]},
    ".card .value":       {"font": "title", "color": "primary"},
    ".card .delta.positive": {"color": "positive"},
    "#hero":              {"padding": "2xl"},
    ".intro > *":         {"animate": "fade-up", "stagger": "fast"},
    ".card:first":        {"margin-top": 0}
})
```

**支持的选择器语法（CSS 子集）**：

| 类型 | 语法 | 支持 |
|---|---|---|
| 类型选择器 | `title`, `card`, `grid` | ✅ |
| 类选择器 | `.kpi`, `.primary` | ✅ |
| ID 选择器 | `#hero` | ✅ |
| 多类组合 | `.card.kpi.primary` | ✅ |
| 类型+类 | `card.kpi` | ✅ |
| 后代选择器（空格） | `.card .value` | ✅ |
| 子选择器（`>`） | `.intro > *` | ✅ |
| 通配符 `*` | `.intro > *` | ✅ |
| 伪类 | `:first`, `:last`, `:only`, `:nth(N)` | ✅ |
| 属性选择器 | `[class=foo]` | ❌（v1 不支持） |
| 相邻兄弟 `+` / 通用兄弟 `~` | | ❌（v1 不支持） |
| 伪元素 `::before` | | ❌（PPT 无对应） |
| `:hover/:focus/:active` | | ❌（PPT 无交互） |

**伪类语义**：

- `:first` — 在父 container 的 children 中是第一个同类型节点
- `:last` — 同理，最后一个
- `:only` — 是唯一的同类型节点
- `:nth(N)` — 第 N 个（1-based）

### 4.3 Cascade 与 specificity

求解顺序（后者覆盖前者）：

1. **Theme defaults**：组件的内置默认样式（来自组件定义）
2. **Registered StyleSheet**：`sheet` 字段引用的预置 sheet
3. **Deck-local `styles`**：deck IR 中的 `styles` 字段
4. **Inline `style`**：节点上的 `style` 对象

**Specificity 计算**（复用 CSS 规则）：

| 权重 | 来源 |
|---|---|
| 1000 | 每个 `#id` |
| 100 | 每个 `.class` 或 `:pseudo` |
| 10 | 每个 `tag` |
| 0 | `*` |

同一规则在同一来源内，**后定义者优先**（对应 CSS 的 source order）。不同来源的优先级：**inline > deck-local > sheet > theme default**，与 specificity 正交（不是 CSS 严格行为，但对 PPT 场景更清晰）。

### 4.4 继承规则

默认**继承**的样式 key（从父节点流到子节点）：

- `color`, `font`, `font-family`, `font-weight`, `font-style`
- `line-height`, `text-align`
- `lang`

默认**不继承**的 key：

- `bg`, `border`, `padding`, `margin`, `radius`, `shadow`
- `width`, `height`, `cols`, `rows`, `gap`
- `animate`, `delay`, `duration`

节点上显式指定的 key 终止继承（即被自身值覆盖）。子节点可通过 `inherit` 值显式继承：`{"color": "inherit"}`。

### 4.5 Style key 白名单

完整支持的 key 列表见附录 A。大类：

| 分类 | Keys |
|---|---|
| 颜色 | `color`, `bg`, `border`, `border-left`, `border-top`, `border-right`, `border-bottom` |
| 字体 | `font`, `font-family`, `font-weight`, `font-style`, `line-height`, `letter-spacing` |
| 盒子 | `padding`, `padding-x/y/l/r/t/b`, `margin`, `margin-*`, `radius`, `shadow`, `opacity` |
| 尺寸 | `width`, `height`, `min-width`, `max-width`, `min-height`, `max-height`, `aspect-ratio` |
| 布局 | `gap`, `row-gap`, `column-gap`, `cols`, `rows`, `justify`, `align`, `align-items`, `align-self` |
| Flex | `grow`, `shrink`, `basis`, `wrap` |
| 文本 | `text-align`, `text-transform`, `overflow`, `max-lines` |
| 动画 | `animate`, `delay`, `duration`, `stagger`, `animate-trigger`, `animate-group` |

### 4.6 Style key 黑名单

明确拒绝并给出替代建议（错误消息模板见 §11.3）：

| 拒绝的 key | 理由 | 替代 |
|---|---|---|
| `display` | 布局由组件决定 | 用 `<Flex>` / `<Grid>` / `<Stack>` 组件 |
| `position` | 绝对定位由组件决定 | 用 `<Absolute>` 组件 |
| `float`, `clear` | PPT 无文档流 | 用 `<Grid>` |
| `z-index` | 叠放由组件决定 | 用 `<Layer>` 组件 |
| `transform` | PPT 不支持 CSS 变换 | 用 `<Rotate>` / `<Scale>` 组件 或 `animate: "rotate-in"` |
| `transition` | 时序模型不同 | 用 `animate` + `duration` |
| `@media` | 无视口概念 | 用 deck variant |
| `:hover/:focus/:active` | PPT 无交互状态 | — |
| `grid-template-areas` | v1 不支持 | 用 `cols` / `rows` + `<GridItem>` 定位 |

### 4.7 值的类型与 token 解析

| 值类型 | 示例 | 解析规则 |
|---|---|---|
| Token 名 | `"md"`, `"primary"` | 在对应 token 表中查找（spacing/colors/fonts/radius/…） |
| Hex 颜色 | `"#3B6B9D"` | 直接使用 |
| 数值 | `0.3` | 按该 key 的默认单位解释（spacing=inches, font=pt） |
| 带单位字符串 | `"0.3in"`, `"12pt"`, `"2em"` | 显式单位，`em` 相对当前 font size |
| 元组（复合值） | `["accent", "md"]` | 参照该 key 的复合语义（见 A.3） |
| `"inherit"` | — | 从父节点继承 |
| `"auto"` | — | 由组件决定 |
| `"none"` / `0` / `false` | — | 关闭该属性 |

---

## 5. Component 系统

### 5.1 Atomic vs Semantic

| 层 | 定位 | 例子 |
|---|---|---|
| **Atomic** | 渲染原语，无默认样式语义 | `Box`, `Text`, `Line`, `Circle`, `Image`, `Chart`, `Table` |
| **Semantic** | 带默认样式与结构的"成品块" | `Card`, `KPI`, `Callout`, `Quote`, `KeyPoint`, `Stepper`, `Timeline`, `Figure` |
| **Layout** | 容器 | `Slide`, `Flex`, `Grid`, `Stack` (=VStack), `HStack`, `Absolute`, `Layer`, `Padding` |
| **Text Run** | 仅在 text children 中出现 | `strong`, `em`, `u`, `s`, `code`, `sub`, `sup`, `a`, `br` |

### 5.2 内置原子组件（保留现有 + 补齐）

| 组件 | 用途 | 主要 props |
|---|---|---|
| `text` | 单段文本 | `text` |
| `heading` | 标题（层级由 class 决定） | `text`, `level` |
| `title` | slide 主标题 | `text` |
| `subtitle` | 副标题 | `text` |
| `box` | 矩形 | — |
| `circle` | 圆 | — |
| `line` | 线 | `from`, `to` |
| `arrow` | 箭头 | `from`, `to`, `head` |
| `image` | 位图 | `src`, `fit` |
| `svg` | SVG | `src` or `body` |
| `icon` | 图标（新增） | `name`, `size` |
| `chart` | 原生图表 | `chart_type`, `data` |
| `table` | 表格 | `rows`, `headers` |
| `spacer` | 占位 | `size` |
| `divider` | 分割线 | `orientation` |

### 5.3 内置语义组件（新增，v1 共 12 个）

| 组件 | 语义 | 默认结构 |
|---|---|---|
| `card` | 通用卡片 | Padding + Box + children |
| `kpi` | 关键指标 | label + value + delta |
| `callout` | 提示/强调块 | badge + prose |
| `quote` | 引用 | large quote mark + body + author |
| `pullquote` | 突出引文 | 放大的 quote，跨列 |
| `keypoint` | 要点卡（带编号） | number + title + body |
| `stepper` | 步骤流 | N 个 step（含 number + label + connector） |
| `timeline` | 时间线 | N 个节点（含 date + title + desc） |
| `figure` | 图 + 标题 + 说明 | image/chart + caption + source |
| `caption` | 图说明文字 | text |
| `prose` | 散文段（允许 Markdown 流内容） | children |
| `note` | 讲者备注（不渲染，进 PPTX speaker notes） | children |

每个语义组件定义：
- 默认 class（如 `kpi` 自带 `class="kpi"`）
- 默认子结构（如 `kpi` 默认生成 label/value/delta 三子节点）
- 接受的 props 与 children 模式

### 5.4 组件注册机制

```python
@register_component("kpi")
class KPI(Component):
    props_schema = {
        "label":  {"type": "str", "required": True},
        "value":  {"type": "str", "required": True},
        "delta":  {"type": "str", "required": False},
        "trend":  {"type": "enum", "values": ["positive", "negative", "neutral"], "default": "neutral"},
    }
    default_classes = ["kpi"]
    inherit_text = True

    def expand(self, props, children, style):
        # 从 props 构造子节点树，返回 IR
        return {
            "type": "card",
            "class": "kpi",
            "children": [
                {"type": "text", "class": "label", "text": props["label"]},
                {"type": "text", "class": "value", "text": props["value"]},
                *([{"type": "text", "class": f"delta {props['trend']}", "text": props["delta"]}]
                   if props.get("delta") else [])
            ]
        }
```

**注册约束**：
- `props_schema` 必须完整声明所有可接受的 props；未知 props → error
- `default_classes` 合并到节点的 `class`（在 class 前插入）
- `expand` 返回合法 IR 子树，允许引用其它组件
- 展开在 Parse 之后、Style resolve 之前进行

### 5.5 组件 props 合并规则

节点最终的 class 是 `default_classes + user_class`，去重保序：

```
default_classes = ["kpi"]
user class       = "primary compact"
final class      = "kpi primary compact"
```

`style` 对象按照 §4.3 cascade 规则合并。`props` 严格按 schema 校验，不参与样式 cascade。

---

## 6. Authoring DSLs

### 6.1 四种前端

| 前端 | 场景 | 编译目标 |
|---|---|---|
| **JSON IR** | 最精确；结构密集（图表/网格） | 自身就是 IR |
| **MDX** | 语义组件 + 散文混合（推荐默认） | Parse → IR |
| **Markdown + inline HTML** | 纯散文 deck、快速草稿 | Parse → IR |
| **Python builder** | 代码集成、动态生成 | 构造 IR 对象 |

所有前端编译到**同一** IR，后续管线完全一致。

### 6.2 MDX 语法规范

文件扩展：`.slide.mdx` 或 `.deck.md`（Markdown + 组件标签）。

**Frontmatter**（YAML）：

```yaml
---
theme: minimal
sheet: keynote
meta:
  title: Q1 Review
  author: Product Team
---
```

**组件标签规则**：

- **大写开头** = 组件（对应 IR 的 `type`，小写化后查找）：`<Grid>`, `<KPI>`, `<Callout>`
- **小写开头 + 白名单** = inline HTML（见 §6.4）：`<strong>`, `<em>`, `<span>`
- **小写开头 + 非白名单** = error

**属性映射**：

| MDX 写法 | IR 等价 |
|---|---|
| `<KPI label="DAU" value="125k" />` | `{"type":"kpi","props":{"label":"DAU","value":"125k"}}` |
| `<Card class="kpi primary">...</Card>` | `{"type":"card","class":"kpi primary","children":[...]}` |
| `<Card id="hero">` | `{"type":"card","id":"hero",...}` |
| `<Grid style={{cols:"1fr 1fr",gap:"lg"}}>` | `{"type":"grid","style":{"cols":"1fr 1fr","gap":"lg"}}` |
| `<Grid cols="1fr 1fr" gap="lg">` | 同上（style key 在 attribute 里 = 等价于 style 对象中的 key） |

**children 处理**：

- 子节点内的 Markdown 按 CommonMark 解析
- Markdown 段落 → `{"type":"prose","children":[...]}` 或 `{"type":"p"}`（视上下文）
- 空行分隔段落
- 缩进 2 空格可嵌套组件

**完整示例**：

```mdx
---
theme: minimal
sheet: keynote
---

# 为什么环境才是瓶颈 {.section-intro}

<Subtitle>RL scaling 被忽视的一边</Subtitle>

---

## 瓶颈不在模型

<Grid cols="2fr 1fr" gap="lg">
  <Prose>
    过去三年，社区把算力砸在**更大的模型**和**更长的上下文**上。
    但学习信号来自*环境*——而环境的规模化并没有发生。
  </Prose>
  <Callout kind="Insight">
    瓶颈已经从**算力**转移到**环境多样性**。
  </Callout>
</Grid>

<Note time="90s">叙事节拍 1：重新定义问题</Note>
```

### 6.3 Markdown 扩展语法

对纯 Markdown 用户（不写组件标签），支持 pandoc/kramdown 风格的扩展：

**ATX 标题带属性**：

```markdown
# Title {.cover}
## Heading {#hero .primary}
```

→ `{"type":"title","class":"cover","text":"Title"}`

**Fenced Div（自定义块）**：

```markdown
::: callout.insight
瓶颈已经从**算力**转移到**环境多样性**。
:::
```

→ `{"type":"callout","class":"insight","children":[{"type":"prose","children":[...]}]}`

**Fenced Div with attributes**：

```markdown
::: {.card .kpi #hero}
**DAU** — 125k _+56%_
:::
```

**嵌套**（用更多冒号）：

```markdown
:::: grid {cols="1fr 1fr" gap="lg"}
::: card.kpi
**DAU** — 125k
:::
::: card.kpi
**Retention** — 42%
:::
::::
```

**段落属性**：

```markdown
> Subtitle here.
{.subtitle}
```

**分隔 slide**：

```markdown
---
```

顶层 `---`（不在 frontmatter 内）分隔 slide。

### 6.4 Inline HTML 白名单

**只接受**影响 text run 属性的标签，任何块级 HTML → error。

| 标签 | 映射到 | 允许的属性 |
|---|---|---|
| `<b>`, `<strong>` | `run.bold = true` | — |
| `<i>`, `<em>` | `run.italic = true` | — |
| `<u>` | `run.underline = true` | — |
| `<s>`, `<del>` | `run.strike = true` | — |
| `<sub>` | `run.vertAlign = sub` | — |
| `<sup>` | `run.vertAlign = sup` | — |
| `<code>` | `run.font = theme.font_mono` | — |
| `<br>` | 段内换行 | — |
| `<a>` | `run.hyperlink` | `href`（必需） |
| `<span>` | text run with style | `style`（仅 `color`/`background-color`/`font-weight`/`font-style`/`text-decoration`） |
| `<mark>` | `run.highlight = true` | `class`（可选） |

**style 属性值**：只接受 hex (`#RRGGBB`)、`rgb(...)`、theme token（`primary`）。其它值 → error。

**未列出的标签**（`<div>`, `<p>`, `<section>`, `<h1>`…）→ error with suggestion：

```
Unsupported inline HTML tag: <div>
  at slides[2].children[1]
  Block-level HTML is not supported. Use a fenced div:
    ::: {.your-class}
    ...
    :::
  or an MDX component:
    <YourComponent>...</YourComponent>
```

### 6.5 Python builder（IR-first 新 API）

从零重新设计的 Python 前端，直接构造 IR node。核心原则：
- 每个组件类对应一个注册的 `type`
- 构造参数 = IR 字段（`class_`, `id_`, `style`, `props`, `children`）
- 不再有"inline 大量样式参数"的旧风格；样式统一走 `style=` 或 stylesheet

```python
from paperops.slides import Deck, Slide
from paperops.slides.components import Title, Subtitle, Heading, Grid, KPI

deck = Deck(theme="minimal", sheet="keynote")

deck += Slide(class_="cover")[
    Title("Q1 Review"),
    Subtitle("Data-driven, iterative"),
]

deck += Slide(class_="content")[
    Heading("Key Metrics"),
    Grid(style={"cols": "1fr 1fr 1fr", "gap": "lg"})[
        KPI(label="DAU",       value="125k", delta="+56%", trend="positive"),
        KPI(label="Retention", value="42%",  delta="+8pt", trend="positive"),
        KPI(label="Churn",     value="3.2%", delta="-1pt", trend="positive"),
    ],
]

deck.render("out.pptx")
```

底层：`Deck` / `Slide` / `Title` / `Grid` / `KPI` 等都是 `Node` 子类，构造时立即产出 IR 子树，`deck.render()` 走与 JSON/MDX 完全相同的渲染管线。

### 6.6 前端选择建议

| 任务特征 | 推荐前端 |
|---|---|
| LLM 生成的讲稿 deck（散文为主） | MDX |
| LLM 生成的数据 deck（图表+网格） | JSON IR |
| LLM 混合 deck（最常见） | MDX |
| 模板市场 / 设计师维护 | MDX + 自定义 stylesheet |
| 程序化生成（从数据库拉数据） | Python builder |
| 单元测试 | JSON IR |

---

## 7. Layout 原语

### 7.1 现有原语（保留）

`paperops.slides.layout` 已提供完整的 Flex/Grid 实现，本次不改动核心算法。

| 原语 | IR `type` | 作用 |
|---|---|---|
| `Flex` (row/column) | `flex`, `hstack`, `vstack` | 主轴+交叉轴线性布局 |
| `Grid` | `grid` | 多轨二维布局，`cols`/`rows` 支持 `fr`/`auto`/`fixed` |
| `Absolute` | `absolute` | 子元素按 `left`/`top`/`width`/`height` 绝对定位 |
| `Layer` | `layer` | Z 轴叠放 |
| `Padding` | `padding` | 内边距包装 |
| `Stack` | `stack` | `vstack` 的别名 |

**约定**：用**组件类型**切换布局模式（`<Flex>` vs `<Grid>`），不用 CSS `display` 属性。

### 7.2 对齐与基线网格

**组件级对齐**（通过 style）：

- `justify`: `start / center / end / space-between / space-around / space-evenly`
- `align` / `align-items`: `start / center / end / stretch / baseline`
- `align-self`: 覆盖父容器的 `align-items`

**基线网格**（新增）：

Theme 中定义 `baseline: 0.1`（英寸）。开启方式：

```json
{"type": "slide", "style": {"baseline-snap": true}, ...}
```

或在 stylesheet 中：

```python
{"slide": {"baseline-snap": True}}
```

开启后，所有节点的 `y` 坐标在 Layout 阶段末尾 snap 到最近的 baseline 网格线。这是治"对齐差"的关键。

**兄弟对齐**（新增）：

```json
{"style": {"align-to": "sibling.title"}}
```

把当前节点的 `y` 或 `x` 对齐到兄弟节点（通过 class 或 id 引用）的对应边。

### 7.3 尺寸与 grow/shrink/basis

沿用 flexbox 语义（现已实现）：

- `width`/`height`: 明确尺寸（token 或数值）
- `min-width`/`max-width`/`min-height`/`max-height`: 约束
- `grow`: 剩余空间分配权重
- `shrink`: 不足时压缩权重
- `basis`: 主轴初始尺寸
- Grid 轨道：`"1fr 2fr auto 1.5in"`

### 7.4 组件驱动布局（不用 CSS display）

**设计决策**：不接受 `style: {display: "grid"}`，而是强制用 `<Grid>` 组件。

理由：

- 组件封装了**校验**（`<Grid>` 要求 `cols` 或 `rows` 之一）
- 组件封装了**默认样式**（`<Grid>` 默认 `gap: md`）
- LLM 看到 `<Grid>` 比 `display: grid` 意图更清晰
- 避免"我写了 `display: flex` 为什么没生效"这类歧义（某些属性在 PPT 模型里 meaningless）

---

## 8. Animation 系统

### 8.1 动画作为 style property

```json
{"type": "card", "class": "kpi", "style": {
  "animate": "fade-up",
  "animate-trigger": "on-click",
  "animate-group": "intro",
  "delay": "fast",
  "duration": "base"
}}
```

或在 stylesheet 中：

```python
{
  ".intro > *": {"animate": "fade-up", "stagger": "fast"},
  ".reveal":    {"animate": "zoom-in", "duration": "slow"}
}
```

### 8.2 预置动画库

| 动画 | 类型 | 默认参数 |
|---|---|---|
| `fade-in` / `fade-up` / `fade-down` | enter | duration=base |
| `slide-in-left` / `slide-in-right` / `slide-in-up` / `slide-in-down` | enter | duration=base |
| `zoom-in` / `zoom-out` | enter | duration=base |
| `scale-in` | enter | duration=fast |
| `fly-in` | enter | duration=base |
| `emphasis-pulse` / `emphasis-color` | emphasis | duration=fast |
| `fade-out` / `zoom-out-exit` | exit | duration=fast |

### 8.3 Groups 与 stagger

**Group**：同一 `animate-group` 的元素同时触发；group 按 slide 内出现顺序依次触发（或由 `animate-trigger` 显式指定）。

**Stagger**：在 group 内给成员加延迟 —— `stagger: "fast"` → 每个成员延迟 `duration.fast` 累加。

**Trigger**：

- `on-load`（默认，slide 出现即播）
- `on-click`（点击推进）
- `after-previous`（上一个动画完成后）
- `with-previous`（与上一个同时）

### 8.4 PPTX 时序映射

动画在 Codegen 阶段转为 OOXML `<p:timing>` 节点。现有 `animation.py` 已有直接写 OOXML 的能力，扩展方向：

- 把 style key (`animate`, `delay`, `duration`, `stagger`, `animate-group`, `animate-trigger`) 收集成 AnimationSpec
- 按 group 聚合，生成对应的 timing tree
- 支持 `<Note>` 组件把讲者提示写入 speaker notes

---

## 9. Autofit / Overflow

### 9.1 四种策略

每个有文字内容的组件接受 `overflow` style key：

| 值 | 行为 |
|---|---|
| `shrink` | PPT 原生 `MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE`（默认） |
| `reflow` | 溢出部分拆分到下一个同布局 slide |
| `clip` | 截断（PPT `MSO_AUTO_SIZE.NONE` + 裁剪） |
| `error` | 溢出即报错（strict 模式推荐） |

### 9.2 默认值与配置

- 组件级默认：`title`/`heading` 默认 `shrink`；`prose` 默认 `reflow`；`code` 默认 `clip`
- Sheet 级可覆盖：`".card .value": {"overflow": "shrink"}`
- Inline 级可覆盖：优先级最高

### 9.3 reflow 分页规则

`overflow: reflow` 的组件超出时：

1. 二分查找最大不溢出的 children 数量（以段落/列表项为最小切分单元）
2. 把剩余 children 复制到新 slide，沿用同一 `type` 和 class
3. 新 slide 的 `title` 加后缀 `(cont.)`（或 `(续)`，按 `meta.lang` 决定）
4. 新 slide 继承原 slide 的其它结构，但非 reflow 容器只在第一页渲染

边界情况：单个不可切分节点本身溢出 → 退化为 `shrink`，并产生 warning。

### 9.4 不自研文本测量

所有 "这段文字在这个框里放不放得下" 的判断，优先走 PPT 原生机制：

- `shrink` 策略完全交给 PPT autofit，不预测量
- `reflow` 策略用保守估计（按字符数 × 字号的粗略上限）先切，出错再 shrink fallback
- `clip` / `error` 策略在 Codegen 后读回 PPT shape 实际尺寸（via `shape.height`/`shape.width`）做判断

---

## 10. Variants 与预置 Sheets

### 10.1 Deck variant

同一份 content IR + 不同 `theme` + 不同 `sheet` = 完全不同视觉风格的 deck。

```python
deck.render(ir, theme="academic",  sheet="seminar",    out="talk_seminar.pptx")
deck.render(ir, theme="minimal",   sheet="keynote",    out="talk_keynote.pptx")
deck.render(ir, theme="executive", sheet="whitepaper", out="talk_report.pptx")
```

这是解决"风格单一"的核心抓手。

### 10.2 预置 sheets（v1 交付）

| Sheet | 定位 | 特征 |
|---|---|---|
| `default` | 兜底 | 保守间距、单栏为主 |
| `minimal` | 极简风 | 大留白、细分隔、黑白灰为主 |
| `academic` | 学术 slides | 严谨、文字密度高、引文/方程式友好 |
| `seminar` | 学术讲座 | 中等密度、强调数据可视化 |
| `keynote` | 产品发布风 | 大字号、高对比、少元素/页 |
| `whitepaper` | 报告风 | 多栏、数据表格、章节编号 |
| `pitch` | 融资/路演 | 强叙事节奏、金句突出、动效丰富 |

每个 sheet 是一个 Python 文件，导出 `StyleSheet` 实例。

### 10.3 Sheet 扩展指南

用户定义自己的 sheet：

```python
from paperops.slides.style import StyleSheet, register_sheet

register_sheet("my_brand", StyleSheet({
    ".cover": {"bg": "primary", "color": "bg", "padding": "2xl"},
    ".cover title": {"font": 48, "font-weight": "bold"},
    ...
}))
```

之后 IR 里可以 `{"sheet": "my_brand"}`。

---

## 11. Validation & Error Handling

### 11.1 JSON Schema

IR 的 schema 文件：`paperops/slides/ir/schema.json`（JSON Schema draft 2020-12）。

覆盖：

- 顶层字段必填性与类型
- 每个组件 `type` 的 props schema
- style key 的合法值（枚举或 pattern）
- class 的格式（`^[a-z][a-z0-9-]*( [a-z][a-z0-9-]*)*$`）
- id 唯一性
- `$use` 引用必须在 `defines` 里

**两级校验**：

1. **Structural**（Schema 级）：字段存在、类型正确
2. **Semantic**（Parser 级）：组件存在、class 引用的规则存在（warning）、变量绑定完整

### 11.2 严格模式

`options.strict = true` 时以下情况 error（否则 warning）：

- 未知 class（没有任何 stylesheet 规则匹配）
- 未绑定的 `$variable`
- 图片文件不存在
- 文本溢出且 `overflow: "error"`
- 非白名单 inline HTML

### 11.3 结构化错误格式

所有错误以统一结构输出（既用于 CLI 也用于 LLM 反馈）：

```json
{
  "status": "error",
  "errors": [
    {
      "code": "UNKNOWN_STYLE_KEY",
      "message": "Unknown style key: 'display'",
      "path": "slides[2].children[0].style.display",
      "hint": "Layout is selected by component type, not CSS display.",
      "suggestion": ["<Flex>", "<Grid>", "<Stack>", "<Absolute>"]
    },
    {
      "code": "UNRESOLVED_MACRO_VAR",
      "message": "Macro variable '$trend' not provided",
      "path": "slides[2].children[1]",
      "macro": "KPI",
      "provided": ["label", "value", "delta"]
    }
  ],
  "warnings": [
    {
      "code": "UNMATCHED_CLASS",
      "message": "Class 'highlight-pro' matches no stylesheet rule",
      "path": "slides[3].children[0]"
    }
  ]
}
```

LLM 拿到这个格式能**自愈**：它能准确定位哪个路径出错、期望什么、可以用什么替代。

### 11.4 错误码清单（节选）

| Code | 含义 |
|---|---|
| `INVALID_SCHEMA` | 顶层 schema 错 |
| `UNKNOWN_TYPE` | 组件 `type` 未注册 |
| `MISSING_REQUIRED_PROP` | 必填 prop 缺失 |
| `UNKNOWN_STYLE_KEY` | style key 不在白名单 |
| `INVALID_STYLE_VALUE` | style 值无法解析 |
| `UNKNOWN_TOKEN` | token 名未定义 |
| `UNRESOLVED_MACRO_VAR` | `$variable` 未绑定 |
| `UNDEFINED_MACRO` | `$use` 引用不存在 |
| `CIRCULAR_MACRO` | 宏互相引用 |
| `DUPLICATE_ID` | id 冲突 |
| `UNSUPPORTED_INLINE_HTML` | inline HTML 不在白名单 |
| `OVERFLOW_UNRECOVERABLE` | 溢出且无法降级 |

---

## 12. 渲染管线

### 12.1 五阶段 Pipeline

```
          ┌────────────┐
input ────▶│  1. Parse  │────▶ canonical IR tree (with macros expanded)
          └────────────┘
                 │
                 ▼
          ┌────────────┐
          │  2. Style  │────▶ styled IR tree (ComputedStyle on each node)
          └────────────┘
                 │
                 ▼
          ┌────────────┐
          │  3. Layout │────▶ IR tree + Regions per node
          └────────────┘
                 │
                 ▼
          ┌────────────┐
          │  4. Autofit│────▶ possibly re-laid-out tree (reflow may create slides)
          └────────────┘
                 │
                 ▼
          ┌────────────┐
          │  5. Codegen│────▶ .pptx file
          └────────────┘
```

### 12.2 各阶段契约

| 阶段 | 输入契约 | 输出契约 | 失败处理 |
|---|---|---|---|
| Parse | 任一 DSL 字符串 | canonical IR（Schema 通过） | 结构错即 error，不继续 |
| Style | canonical IR | styled IR（每节点有 ComputedStyle） | 未知 key 依 strict 决定 |
| Layout | styled IR | IR + Regions | overflow 只标记，不处理 |
| Autofit | IR + Regions | 可能新增 slide + 修正 Regions | 按 `overflow` 策略降级 |
| Codegen | 最终 IR + Regions | pptx 字节流 | shape 创建失败 → error |

### 12.3 性能目标

| 指标 | 目标 | 说明 |
|---|---|---|
| 20 页 deck 总耗时 | < 3s | 从 IR 到 pptx |
| Style resolve | < 200ms / deck | 选择器匹配 + cascade |
| Layout | < 500ms / deck | 现有引擎已达标 |
| Codegen | < 1.5s / deck | python-pptx I/O 为主 |
| 大 deck（100 页） | < 15s | 线性缩放 |

### 12.4 可观测性

Pipeline 每阶段输出结构化日志：

```json
{
  "stage": "style",
  "slide": 3,
  "node": "slides[3].children[1]",
  "matched_rules": [".card", ".card.kpi"],
  "computed_style_keys": ["bg", "radius", "padding", "color"],
  "took_ms": 0.42
}
```

开启方式：`options.trace = true`。Debug 和 LLM 反馈两用。

---

## 13. 实施路线

### 13.1 重构原则

项目允许 breaking change，因此实施策略：

- **一次性切换，不做兼容层**：旧 Python API (`Slide()[...]`、`Box(x=, y=, ...)` 等）在 Phase 2 完成时直接退役
- **分支式开发**：在 `refactor/slide-dsl` 分支上完成 Phase 0-4，期间 `main` 不受影响，examples 用旧代码
- **一次性迁移**：Phase 5 结束后，examples 和 skill 文档在同一 PR 里全部切换到新 API，合入 main
- **主线推进**：切换后旧模块文件直接删除，代码库只保留新架构

### 13.2 依赖拓扑

```
Phase 0: Tokens & IR Foundation
  ├─ Theme 重写（colors/fonts/spacing/radius/shadow/duration/density）
  ├─ core/tokens.py 统一值解析入口
  ├─ IR node 数据类
  ├─ JSON Schema 定义
  └─ Component registry（props schema + default classes）
        ↓
Phase 1: Style Resolution
  ├─ style/stylesheet.py
  ├─ style/selector.py（CSS 子集解析 + 匹配）
  ├─ style/cascade.py（specificity + inherit + 层叠求解）
  └─ style/computed.py（ComputedStyle 数据类）
        ↓
Phase 2: Rendering Pipeline（核心切换点）
  ├─ build.py 重写为五阶段 pipeline 入口
  ├─ layout/engine.py 适配 IR node + ComputedStyle
  ├─ Codegen：IR + Regions → python-pptx
  ├─ dsl/json_loader.py
  ├─ dsl/python_builder.py 从零重写（IR-first）
  ├─ ir/defines.py（宏展开）
  └─ 删除：build.py 旧代码、Python API 旧层、auto_size.py
        ↓
Phase 3: Components & Sheets
  ├─ 原子组件按 IR registry 形式重写
  ├─ 12 个语义组件（Card/KPI/Callout/Quote/...）
  ├─ 6 个预置 sheets（minimal/academic/seminar/keynote/whitepaper/pitch）
  └─ 组件与 sheet 文档
        ↓
Phase 4: Authoring DSLs
  ├─ dsl/markdown_parser.py（Markdown + fenced div + 属性语法）
  ├─ dsl/inline_html.py（白名单过滤）
  ├─ dsl/mdx_parser.py（大写组件 + Markdown 混合）
  └─ 四前端联调，输出同 IR
        ↓
Phase 5: Polish & Production Readiness
  ├─ layout/autofit.py（四策略：shrink/reflow/clip/error）
  ├─ layout/baseline.py（基线网格 + align-to-sibling）
  ├─ animation.py 重构（style property → AnimationSpec → OOXML）
  └─ ir/validator.py（结构化错误 + LLM 可读格式）
        ↓
Phase 6: Cutover & Docs
  ├─ examples/ 全部改写到新 API
  ├─ skill 文档更新（slidecraft / slide-review）
  ├─ README + quickstart
  └─ 合并 refactor/slide-dsl → main
```

### 13.3 每阶段交付物

| Phase | 主要交付 | 主要验收方式 |
|---|---|---|
| 0 | Token 系统 + IR 数据结构 + 注册表 | 单元测试：token 解析、registry 查找、IR (de)serialization |
| 1 | StyleSheet + cascade 求解 | 单元测试：给定 IR + sheet，输出 ComputedStyle 快照 |
| 2 | 端到端 JSON → pptx 最小可行管线 | Smoke：最小 IR 能生成可打开的 pptx |
| 3 | 语义组件 + 预置 sheets 可用 | 视觉：同内容 × 6 sheets 产出 6 种视觉风格 |
| 4 | MDX / Markdown / JSON / Python 四前端一致 | 同一逻辑 deck 四前端输出相同 IR |
| 5 | 溢出自动处理；动画可用；错误可读 | 压力测试 + 错误格式契约测试 |
| 6 | examples 全绿；skill 文档同步；main 上线 | `make verify` 全通过 + 人工过 examples |

---

## 14. 外部依赖与风险

### 14.1 Python-pptx 能力边界

- 动画：python-pptx 不支持；现有项目已通过直接写 OOXML 绕过，延续该方案
- 原生图表样式：python-pptx Chart API 不完整；复杂样式需直写 OOXML
- SmartArt：不支持；用自绘 `<Stepper>` / `<Timeline>` 组件替代
- 3D 效果：v1 不支持

### 14.2 字体风险

- CJK 字体在非 Windows 机器上缺失会导致 autofit 行为偏差
- 解决：Theme 声明 `font_family` + `font_fallback`，未找到首选字体时 fallback
- LibreOffice headless 渲染预览作为 CI 验证手段

### 14.3 schema 演进

- IR schema 版本化：`$schema: "paperops-slide-1.0"`
- 未来破坏性变更走新 major 版本；解析器同时支持多版本
- `defines` 和 stylesheet 不进入 schema 版本，可自由扩展

### 14.4 LLM 对 DSL 的熟悉度

- MDX / JSX / CSS 子集都在主流 LLM 预训练数据中充分覆盖
- 需提供一个**紧凑的 system prompt**，说明：
  - 可用组件清单 + props schema
  - 支持的 style key + 黑名单 + 替代
  - inline HTML 白名单
  - 三档 DSL 选择指南
- 这个 prompt 放在 `.claude/skills/slidecraft/SKILL.md`（已有，需更新）

---

## 15. 附录

### A.1 完整 Style Key 参考

| Key | 值类型 | 继承 | 示例 |
|---|---|---|---|
| `color` | color | ✅ | `"primary"`, `"#333"` |
| `bg` | color | ❌ | `"bg_alt"` |
| `border` | `[color, width]` 或 `[color, width, style]` | ❌ | `["border", "sm"]`, `["border", "sm", "solid"]` |
| `border-left/top/right/bottom` | 同上 | ❌ | `["accent", "md"]` |
| `radius` | radius token 或 数值 | ❌ | `"md"`, `0.12` |
| `shadow` | shadow token | ❌ | `"sm"`, `"none"` |
| `opacity` | 0-1 | ❌ | `0.8` |
| `padding` | spacing token 或 数值 或 `[v,h]` 或 `[t,r,b,l]` | ❌ | `"md"`, `["sm","md"]` |
| `padding-x/y/l/r/t/b` | spacing | ❌ | `"md"` |
| `margin` | 同 padding | ❌ | — |
| `margin-*` | 同上 | ❌ | — |
| `gap` | spacing | ❌ | `"lg"` |
| `row-gap` / `column-gap` | spacing | ❌ | `"md"` |
| `font` | font token 或 pt | ✅ | `"title"`, `24` |
| `font-family` | string | ✅ | `"Inter"` |
| `font-weight` | `normal / bold / <100-900>` | ✅ | `"bold"` |
| `font-style` | `normal / italic` | ✅ | `"italic"` |
| `line-height` | number（倍数）或 pt | ✅ | `1.4` |
| `letter-spacing` | pt 或 em | ✅ | `"0.02em"` |
| `text-align` | `start / center / end / justify` | ✅ | `"center"` |
| `text-transform` | `none / upper / lower / capitalize` | ✅ | `"upper"` |
| `width` / `height` | length 或 `"auto"` | ❌ | `4.0`, `"auto"` |
| `min-width` / `max-width` / `min-height` / `max-height` | length | ❌ | — |
| `aspect-ratio` | `w/h` 或 `number` | ❌ | `"16/9"`, `1.777` |
| `cols` / `rows` | track list | ❌ | `"1fr 2fr auto"` |
| `justify` | flex/grid 关键词 | ❌ | `"center"` |
| `align` / `align-items` | 同上 | ❌ | `"baseline"` |
| `align-self` | 同上 | ❌ | — |
| `align-to` | sibling ref | ❌ | `"sibling.title:bottom"` |
| `grow` / `shrink` | number | ❌ | `1` |
| `basis` | length | ❌ | `2.0` |
| `wrap` | `true/false` 或 `"wrap-reverse"` | ❌ | `true` |
| `overflow` | `shrink / reflow / clip / error` | ❌ | `"shrink"` |
| `max-lines` | integer | ❌ | `3` |
| `baseline-snap` | bool | ✅ | `true` |
| `animate` | animation name | ❌ | `"fade-up"` |
| `animate-trigger` | `on-load / on-click / after-previous / with-previous` | ❌ | `"on-click"` |
| `animate-group` | string | ❌ | `"intro"` |
| `delay` | duration token 或 秒 | ❌ | `"fast"`, `0.3` |
| `duration` | 同上 | ❌ | `"base"` |
| `stagger` | duration token 或 秒 | ❌ | `"fast"` |

### A.2 CSS 黑名单 + 替代建议

| 拒绝的 key | 替代 |
|---|---|
| `display` | 用 `<Flex>` / `<Grid>` / `<Stack>` / `<Absolute>` / `<Layer>` 组件 |
| `position` | 用 `<Absolute>` 组件 |
| `float`, `clear` | 用 `<Grid>` |
| `z-index` | 用 `<Layer>` 组件 |
| `transform` | 用 `<Rotate>` / `<Scale>` 组件 或 `animate` 预设 |
| `transition` | 用 `animate` + `duration` |
| `filter` | 用 `opacity` 或 预渲染图片 |
| `flex-direction` | `<Flex direction="row|column">` |
| `flex-wrap` | style `wrap` |
| `justify-content` | style `justify` |
| `align-content` | v1 不支持 |
| `grid-template-areas` | 用 `<GridItem>` 的 row/col span |
| `grid-auto-flow` | v1 不支持 |
| `object-fit` | `image.fit` prop |
| `overflow` (CSS 语义) | 我们的 `overflow` 是 autofit 策略，不是 scroll/hidden |
| `visibility` | 条件生成，不渲染该节点 |
| `cursor` | PPT 无对应 |
| `@media` | deck variant |
| `:hover / :focus / :active` | PPT 无交互态 |

### A.3 复合值规范

| Key | 元组形式 |
|---|---|
| `border` | `[color, width]` 或 `[color, width, style]`；style ∈ `{solid, dashed, dotted}` |
| `border-*`（四边之一） | 同上 |
| `padding` / `margin` | 1 值 = 四边；`[v, h]` = 上下/左右；`[t, r, b, l]` = 四边 |
| `shadow`（inline 不用 token 时） | `{dx, dy, blur, color, opacity}` |
| `cols` / `rows` | 空格分隔 track 列表：`"1fr 2fr auto 1.5in"` |

### A.4 Inline HTML 白名单完整表

（见 §6.4，此处只列标签名）

```
<b> <strong> <i> <em> <u> <s> <del> <sub> <sup> <br>
<code> <a> <span> <mark>
```

### A.5 预置组件清单（v1）

**Layout**：`slide`, `flex`, `grid`, `stack`, `vstack`, `hstack`, `absolute`, `layer`, `padding`, `spacer`

**Atomic**：`text`, `heading`, `title`, `subtitle`, `box`, `circle`, `line`, `arrow`, `image`, `svg`, `icon`, `chart`, `table`, `divider`

**Semantic**：`card`, `kpi`, `callout`, `quote`, `pullquote`, `keypoint`, `stepper`, `timeline`, `figure`, `caption`, `prose`, `note`

**Text runs**（仅 children 内）：`strong`, `em`, `u`, `s`, `code`, `sub`, `sup`, `a`, `br`, `mark`

### A.6 术语表

| 术语 | 定义 |
|---|---|
| **IR** | Intermediate Representation，canonical JSON 树 |
| **Node** | IR 树中的一个节点，六字段结构 |
| **Component** | 已注册的 `type`，决定节点如何渲染 |
| **StyleSheet** | `{selector: style_object}` 的有序字典 |
| **Token** | Theme 中的命名值（`md`, `primary`, `base`） |
| **ComputedStyle** | 节点经 cascade 求解后的最终样式对象 |
| **Cascade** | Theme → Sheet → Deck-local styles → Inline 的层叠求解 |
| **Specificity** | 选择器权重（id/class/tag 加权） |
| **Macro** | `defines` 中的参数化 IR 子树 |
| **Variant** | 同一 IR + 不同 theme/sheet 组合产生的不同视觉版本 |
| **Autofit** | 溢出处理策略（shrink/reflow/clip/error） |
| **Baseline grid** | 全 deck 统一的 y 坐标对齐网格 |

---

## 16. Open Questions

以下议题在 v1 实施阶段需要定案，当前文档先占位：

1. **Stylesheet 是否支持 `@import`** 以便组合复用（v1 暂不支持，可通过 Python 合并）
2. **响应式（针对 aspect-ratio 的 media query）** 是否要引入 —— 倾向于 v2 再考虑
3. **Figma 导入** 作为另一种 authoring 前端 —— 待需求明确
4. **图片自动优化**（JPEG 压缩、尺寸规整）—— 可作为 Codegen 阶段钩子
5. **LaTeX 数学公式**：v1 作为图片（通过 matplotlib 预渲染），v2 可考虑原生 OMML
6. **PDF 输出**：非 v1 目标，但管线设计应保留 Renderer 可替换的能力
7. **模板市场**：Sheet 的分发/安装机制（npm-like？）—— v2 考虑

---

## 17. 修订历史

| 版本 | 日期 | 变更 |
|---|---|---|
| 0.1 | 2026-04-23 | 初稿：三层架构 + JSON IR + MDX/Markdown + 借词不借义的 HTML/CSS 集成策略 |
