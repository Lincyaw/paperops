# PPT大纲（10分钟，12页）

## 论文标题
**Rethinking the Evaluation of Microservice RCA with a Fault Propagation-Aware Benchmark**

---

## Slide 1: Title（30秒）
- 标题
- 作者、单位
- FSE 2026

---

## Slide 2: Motivation - 当系统崩溃时（1分钟）

**真实案例：**
- 2024年CrowdStrike故障：850万台Windows设备蓝屏
- 2023年某云服务商故障：数百万用户无法访问
- 企业停机成本：$23,000/分钟

**技术背景：**
- 现代系统从单体应用拆分为数百个微服务
- 一个用户请求可能经过几十个服务
- 单个故障会像多米诺骨牌一样级联传播
- 人工排查已不可能

**RCA（Root Cause Analysis）任务：**
- 目标：从海量遥测数据中自动定位故障根因
- 输入：Metrics（指标）+ Logs（日志）+ Traces（追踪）
- 输出：根因服务排名
- 评估：Top@K（根因是否在前K个预测中）

---

## Slide 3: The Promise - AI能解决吗？（1分钟）

**研究进展：**
- 早期：单一模态方法（仅用日志、仅用指标、或仅用追踪）
- 近期：多模态融合成为趋势
  - Graph Neural Networks：建模服务依赖图
  - Causal Inference：学习因果关系
  - Deep Learning：端到端学习

**报告的性能：**
- Top@1准确率：80-95%
- MRR（平均倒数排名）：0.85+

**But...**
- 每篇论文用不同的数据集
- 数据集规模小、故障类型单一
- 这些数字真实吗？

---

## Slide 4: Our Question - 基准测试可靠吗？（1分钟）

**核心质疑：**
- SOTA算法的"进步"是真实能力的体现？
- 还是评估基准过于简单的产物？

**验证方法：**
- 设计一个极简的基线方法
- 如果简单方法也能达到高性能，说明基准有问题

**SimpleRCA设计：**
- 完全基于规则，零机器学习
- 模拟运维专家的直觉排查逻辑

**三条检测规则：**
1. **Metrics异常**：3-sigma阈值检测，超限即报警
2. **Traces异常**：P95延迟超过基线3倍即报警  
3. **Logs异常**：统计ERROR/FAIL关键词，超阈值即报警

**根因判定**：报警数量最多的服务 = 根因

---

## Slide 5: The Shocking Discovery（1分钟）

**实验设置：**
- 4个主流公开基准：RE2、RE3、Nezha、Eadro
- 对比：SimpleRCA vs SOTA模型

**结果对比：**
| Dataset | SOTA Best | SimpleRCA |
|---------|-----------|-----------|
| RE2-TT  | 0.67      | **0.80**  |
| RE3-TT  | 0.50      | **0.83**  |
| Nezha-TT| 0.87      | **0.93**  |
| Eadro-TT| 0.99      | 0.81      |

**结论：**
- 简单规则方法匹敌甚至击败复杂AI模型
- 现有基准无法区分简单与复杂方法
- 模型性能被高估了

---

## Slide 6: Root Cause - 为什么现有基准不够好？（1分钟）

**三大局限：**

**1. Limited fault cases（样本不足）**
- 除GAIA外，所有数据集<200个案例
- Eadro和Nezha <100，AIOps-2021仅159
- 复杂模型容易过拟合到有限场景

**2. Simplified request structures（结构简化）**
- 即使有69个服务，请求只覆盖27个
- 调用链最大深度仅2-3跳
- 真实系统通常有7+层深度

**3. Narrow fault spectrum（故障类型单一）**
- 6类故障（CPU/Code/Config/Disk/Memory/Network）
- 但大多数数据集只有1-2种类型
- 严重不平衡

**故障注入模式分析：**
- 86%的案例是Type I：症状仅在注入服务显现
- 根因服务症状最明显
- 简单相关性即可定位，无需复杂因果推理

---

## Slide 7: Our Solution - 故障传播感知基准（1分钟）

**设计目标：**
- 更复杂、更真实、更具挑战性
- 真正测试模型的因果推理能力

**关键数字：**
- 9,152次故障注入 → 1,430个验证案例
- 31种故障类型 → 筛选出25种有影响类型
- 50个微服务，最大7层调用深度
- 动态工作负载（状态机驱动，36种执行路径）

**核心创新 - Impact-Driven Validation：**
- 关键发现：不是所有注入故障都会产生用户影响
- 84.4%的初始注入被归类为"无异常"（静默故障）
- 通过SLI（成功率、延迟）验证用户影响
- 只保留operationally relevant的案例

---

## Slide 8: Six-Stage Framework（1分钟）

1. **System Foundation**: 
   - TrainTicket平台（50服务的大规模开源系统）
   - 增强可观测性（OpenTelemetry）

2. **Dynamic Workload**: 
   - 状态机工作流生成
   - 6×3×2=36种执行路径组合

3. **Fault Injection**: 
   - ChaosMesh框架
   - 31种故障类型，>10^16种配置组合

4. **Multi-modal Collection**: 
   - Metrics + Logs + Traces
   - 154.7M日志行，11.2M追踪

5. **Hierarchical Annotation**: 
   - 服务→Pod→容器→函数级标签
   - 细粒度根因定位

6. **Impact Validation**: 
   - SLI验证，过滤"静默"故障
   - 确保案例的操作相关性

---

## Slide 9: Re-evaluation - 残酷的现实（1分钟）

**11个SOTA模型在新基准上的表现：**

| Metric | Old Benchmarks | Our Benchmark |
|--------|----------------|---------------|
| Top@1  | 0.75           | **0.21**      |
| Best   | 0.90+          | **0.37**      |
| Time   | seconds        | **hours**     |

**SimpleRCA**: 降至0.28
- 证明新基准有区分度
- 复杂场景确实需要复杂方法

**关键发现：**
- 性能断崖式下跌
- 暴露出现有模型的系统性弱点
- 执行时间从秒级暴增至小时级（可扩展性问题）

---

## Slide 10: Deep Dive - 三大关键失效模式（1.5分钟）

**通过262个最困难案例的人工分析：**

**1. Scalability Issues（39.8%）**
- 执行时间随数据量线性增长
- CausalRCA: 8分钟数据需10分钟处理
- Nezha: 需要94.62秒
- 无法满足生产环境实时性要求

**2. Observability Blind Spots（47.4%）**
- **Signal-Loss**: 信号缺失（如PodKill故障，服务突然消失）
- **Infrastructure-Level**: 未监控组件（如数据库内部）
- **Semantic-Gap**: 矛盾信号（错误日志+成功追踪状态）

**3. Modeling Bottlenecks**
- 核心假设在复杂场景失效
- 例：Nezha假设故障增加事件频率
- 但PodKill导致信号消失，模型误判为正常

---

## Slide 11: Implications & Conclusion（1分钟）

**对研究社区的启示：**
- 当前研究可能在"优化错误的目标"
- 基准测试必须与模型协同进化
- 简单基线应成为第一道门槛
- 需要更真实、更具挑战性的评估

**Contributions：**
1. 首次系统揭示公开RCA基准的过度简化问题
2. 提出故障传播感知的新基准框架
3. 大规模重评估识别SOTA模型的三大关键失效模式
4. 建立标准化、可复现的评估生态（11个模型容器化实现）

**Open Source：**
- 框架、数据集、重实现模型全部开源
- GitHub: [URL]

---

## Slide 12: Thank You / Q&A

---

# 时间分配
- Slide 1: 30s
- Slide 2-3: 2min (Motivation + Background)
- Slide 4-5: 2min (SimpleRCA + Discovery)
- Slide 6: 1min (Root Cause Analysis)
- Slide 7-8: 2min (Solution + Framework)
- Slide 9-10: 2min (Results + Deep Dive)
- Slide 11-12: 1.5min (Conclusion)

Total: ~10-11分钟
