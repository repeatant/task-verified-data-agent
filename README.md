# Task-Verified Data Agent

**面向 SQL 数据分析 Agent 的任务核验、反例诊断与反馈修复。**

调用方确认指标与统计范围后，Agent 生成 SQL。核验器检查查询是否满足任务要求，对错误候选生成可重放的反例，并对修复结果执行回归。LangGraph 与 DB-GPT 共用核验核心。

[使用指南](docs/USAGE.md) · [架构与执行契约](docs/ARCHITECTURE.md) · [源码导览](docs/CODE_GUIDE.md) · [评测](docs/EVALUATION.md)

## 快速开始

以下命令在 Windows、Python 3.12 环境下从仓库根目录执行。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/core.lock.txt -r requirements/probes.txt
.\.venv\Scripts\python.exe tools/replay_probe.py tests/fixtures/public_evidence/date_boundary.json
```

示例重放一条日期上界错误。查询把十月一日零点也计入九月，导致添加一条范围外明细后，结果从 `1` 变为 `2`。输出中的 `reproduced: true` 和 `property_id: exclude_end` 表明错误被重现。

运行包含生成、核验、反馈修复及发布的 LangGraph 离线流程。

```powershell
.\.venv\Scripts\python.exe tools/run_graph.py --case d06 --thread boundary-demo
```

两条演示命令均无需模型 API。断点恢复见 [LangGraph 指南](docs/LANGGRAPH.md)，DB-GPT 与真实模型接入见 [使用指南](docs/USAGE.md)。

## 目录结构

```text
dataverify/          Python 源码
  verification/      任务契约、SQL 核验、反例生成与缩减
  agent/             Agent 编排、反馈与断点恢复
  integrations/      DB-GPT 与模型 API 接入
  evaluation/        任务样本、评分与对照实验
  runtime/           预算和运行报告
docs/                安装、架构、评测与贡献指南
requirements/        基础依赖、锁定版本与可选环境
tests/               自动化测试
  fixtures/          可重放的示例数据、评测样例与冻结实现
tools/               离线演示、评测与环境安装入口
```

从 [核验入口](dataverify/verification/probe_guard.py) 或 [LangGraph 执行器](dataverify/agent/probe_langgraph.py) 开始阅读。

## 验证结果

- **LangGraph 闭环**。两条受控错误分别用一次真实调用完成修复。一项自然任务跨进程恢复后通过核验，恢复阶段新增模型请求为零。
- **反馈对照**。`legacy_v1` 的 16 组有效配对中，完整与紧凑反馈均修复 14/16，紧凑反馈的输入 Token 减少 7.0%。另三项自然任务的通过数分别为 3/3 和 2/3。
- **冻结挑战**。原实现通过 21/24 项。修复后，同一批挑战作为回归用例通过 24/24 项。

数据来自固定 `orders` / `order_lines` 表的合成开发任务，覆盖明细净额、逐订单与逐渠道统计。支持的 SQL 语法、完整实验分组与局限见 [执行契约](docs/ARCHITECTURE.md#执行契约)和[评测说明](docs/EVALUATION.md)。

开发与代码检查见 [贡献指南](docs/CONTRIBUTING.md)。第三方依赖、方法参考与许可状态见 [第三方声明](docs/THIRD_PARTY_NOTICES.md)。
