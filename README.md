# Task-Verified Data Agent

**用执行证据核验 SQL，定位分歧，并检查修复结果。**

基于 LangGraph 编排生成、核验与独立核算。模型根据任务编写参考计算和测试数据，本地工具执行 SQL 与参考程序；出现分歧时，再决定修改 SQL、替换参考版本或请求澄清。旧参考与反例始终保留，用于修复后的回归。

[使用指南](docs/USAGE.md) · [动态核验](docs/DISCOVERY.md) · [架构](docs/ARCHITECTURE.md) · [源码导览](docs/CODE_GUIDE.md) · [评测](docs/EVALUATION.md) · [变更记录](docs/CHANGELOG.md)

v1.0.0 的运行路径仅使用 LangGraph；DB-GPT 安装、运行入口及适配层已移除，旧实验单独归档。

## 如何工作

- **隔离上下文**：生成方不读取参考程序；独立核算只读取需求和无答案的数据；规划、归因与常规复核分别构建请求。
- **执行分歧**：SQLite 执行 SQL，受限解释器执行参考程序。Hypothesis 搜索并缩减能区分计算结果的数据。
- **定向修改**：依据需求引文与实际执行记录选择修改对象；替换参考时保留版本及全部反例，不能删除失败测试获得通过。
- **重放与恢复**：检查点绑定任务、数据、配置和代码版本；恢复后重新检查结果，未决付费请求不自动重发。

模型仍可能共同误解需求。`reviewed` 只表示有限检查通过，不是语义正确性证明；当前数据范围为固定的 `orders` / `order_lines` 两张表。

## 快速开始

已验证开发环境为 Windows、Python 3.12。以下命令在仓库根目录执行，均不调用模型 API。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/core.lock.txt -r requirements/probes.txt

# 重放动态核验与修复
.\.venv\Scripts\python.exe tools/evaluate_discovery.py --track controlled
.\.venv\Scripts\python.exe tools/evaluate_discovery.py --track correct
.\.venv\Scripts\python.exe tools/evaluate_reference_audit.py
```

离线响应由脚本提供，用于检查控制流，不衡量模型能力。第三条命令覆盖九种故障注入场景，包含错误参考、错误 SQL、明确舍入要求，以及共同误解且无确认样例时的已知漏检。

预设性质路径可以单独运行，适合观察一条日期边界错误的生成、核验、反馈与修复：

```powershell
.\.venv\Scripts\python.exe tools/run_graph.py --case d06 --thread boundary-demo
.\.venv\Scripts\python.exe tools/replay_probe.py tests/fixtures/public_evidence/date_boundary.json
```

第二条命令的 `reproduced: true` 表示错误被重现，不表示 SQL 通过。真实模型配置见 [使用指南](docs/USAGE.md)，预设性质路径的断点恢复见 [LangGraph 指南](docs/LANGGRAPH.md)。

## 验证结果

v1.0.0 的真实千问联调覆盖主要执行分支：

- **参考与 SQL 分开修复**：三条受控任务分别完成错误参考替换、错误 SQL 修复和明确舍入保留；最终 SQL、当前参考与独立核算均通过七组开发评分数据。
- **自然生成**：订单去重任务的 SQL、参考计划与独立核算均由模型生成，源数据结果为 `3`；三种计算均通过七组开发评分数据。
- **跨进程恢复**：预设性质任务 `d03` 用一次真实调用生成 SQL，暂停后在新进程恢复并核验，恢复阶段新增模型请求为零。

首轮参考任务因非法 DSL 语法未完成；完善通用 AST 诊断后补测，失败记录同样保留。本轮共 26 次真实 HTTP 调用、53,194 Token，属于迭代后的开发联调，不是一次冻结评测的成功率。详见 [本版公开证据](tests/fixtures/discovery/v1_live_validation.json)与[评测说明](docs/EVALUATION.md)。

历史反馈压缩对照来自已移除的 DB-GPT 路径，不归到 LangGraph 或动态核验。实验分组、失败和历史数据见 [评测说明](docs/EVALUATION.md)。

## 目录结构

```text
dataverify/          Python 源码
  agent/             LangGraph 编排、角色上下文与恢复
  verification/      SQL 执行、参考计算、反例与验收
  integrations/      模型 API 配置与客户端
  evaluation/        开发样例与独立评分
  runtime/           预算与运行报告
docs/                使用、架构、评测与贡献指南
  history/           旧实验及来源记录
requirements/        依赖与锁定版本
tests/
  fixtures/          可重放样例、公开证据与冻结历史实现
tools/               离线演示、评分与显式真实模型入口
```

从 [动态图入口](dataverify/agent/discovery_graph.py) 或 [源码导览](docs/CODE_GUIDE.md) 开始阅读。测试约定见 [贡献指南](docs/CONTRIBUTING.md)，依赖与方法来源见 [第三方声明](docs/THIRD_PARTY_NOTICES.md)。
