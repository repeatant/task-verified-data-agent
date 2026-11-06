# 源码阅读指南

先读一次任务的编排入口，再沿调用进入核验与执行。当前 Agent 编排只使用 LangGraph；两种任务契约共用基础工具，但验收条件不同。

## 动态任务：从需求到执行证据

从 [run_discovery](../dataverify/agent/discovery_graph.py) 开始：独立规划参考 → 生成 SQL → 执行检查 → 盲态核算 → 搜索分歧 → 归因 → 定向修改 → 最终重放。

`_DiscoveryNodes` 管理图节点、状态与请求上下文；[ReferenceAuditNodes](../dataverify/agent/discovery_audit.py) 负责独立核算、分歧归因及参考版本替换。生成、核验、核算使用不同客户端；核验的规划、归因、常规复核也重新构建上下文。

[discovery_model.py](../dataverify/agent/discovery_model.py) 定义提示词与 JSON 动作协议；[discovery_runtime.py](../dataverify/verification/discovery_runtime.py) 校验任务、数据、计划并执行只读 SQL。模型提出的参考程序由 [reference_expression.py](../dataverify/verification/reference_expression.py) 解释，不调用 `eval` 或 `exec`。

[reference_audit.py](../dataverify/verification/reference_audit.py) 组织三方执行结果、调用方确认样例与 Hypothesis 区分性数据搜索。执行分歧只说明程序不一致；修改授权还需需求依据，证据不足时澄清。完整过程与限制见 [动态核验指南](DISCOVERY.md)。

## 预设性质任务：生成与修复循环

从 [run_graph_case](../dataverify/agent/probe_langgraph.py) 开始：验证线程绑定 → 加载检查点 → 执行状态图 → 整理结果。`_GraphNodes` 组织节点和路由，活动客户端与数据通过运行依赖传入。

`prepare_call` 保存调用意图，`generate` 生成一次候选，`verify` 委托 Guard，`feedback` 追加诊断，`publish` 重新核验后获取结果。这条路径不生成动态参考程序。

[ProbeGuard.execute](../dataverify/verification/probe_guard.py) 检查任务未变、核验候选、保存反例、回归旧反例、选择反馈并记录审计。每次候选更新先撤销旧接受状态；`publish()` 只返回当前有效结果。

[probe_runtime.py](../dataverify/verification/probe_runtime.py) 的 `_analyse` 分为 SELECT 解析、关联检查、筛选检查和输出检查，再由 `_semantics` 对照已确认任务。语法能力与任务要求分别判断，区分“不支持”与“范围错误”。

## 反馈与恢复

[feedback_strategy.py](../dataverify/agent/feedback_strategy.py) 维护当前任务的候选指纹和失败状态，选择完整或紧凑证据。指纹忽略格式与注释，保留字面量及多语句内容；没有反例时不生成伪证据。

[graph_guard.py](../dataverify/agent/graph_guard.py) 恢复反例与策略状态，不恢复接受状态；[graph_checkpoint.py](../dataverify/agent/graph_checkpoint.py) 管理本地写入锁和累计预算。任务、数据或版本变化时拒绝复用检查点；恢复后必须重新执行候选，未决调用不自动重发。

[graph_model.py](../dataverify/agent/graph_model.py) 负责单次 JSON HTTP 调用，不包含另一套 Agent 循环。角色上下文、预算和检查点是不同边界，详见 [LangGraph 指南](LANGGRAPH.md)。

## 评测与预算

开发任务、参考答案和后置评分位于 `dataverify/evaluation/`。运行时模块不能导入评分答案。

- [reference_audit_scoring.py](../dataverify/evaluation/reference_audit_scoring.py)：同时检查最终 SQL、当前参考和独立核算。
- [probe_grading.py](../dataverify/evaluation/probe_grading.py)：预设性质路径的源数据与附加数据评分。
- [probe_cost_budget.py](../dataverify/runtime/probe_cost_budget.py)：实际 HTTP 前预留额度，记录完整 usage。
- [experiment_budget.py](../dataverify/runtime/experiment_budget.py) 与 [discovery_budget.py](../dataverify/runtime/discovery_budget.py)：累计账本及受限续账审查。
- [reporting.py](../dataverify/runtime/reporting.py)：报告写入、稳定 JSON 摘要和用量汇总。

公开的冻结历史实现保持原样，仅用于离线回放，不属于当前依赖。历史实验不得当作当前代码的验证结果。

## 工程方式的参考

本项目参考官方源码的组织方式，采用短入口、显式状态及独立生命周期，不复制上游业务实现：

- [pytest runner](https://github.com/pytest-dev/pytest/blob/main/src/_pytest/runner.py)：阶段编排与单项运行分开。
- [Pydantic AI RunContext](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/_run_context.py)：显式运行上下文。
- [smolagents memory](https://github.com/huggingface/smolagents/blob/main/src/smolagents/memory.py)：状态记录与序列化边界。
- [SQLGlot 配置](https://github.com/tobymao/sqlglot/blob/main/pyproject.toml)：统一行宽与静态检查。

开发约定见 [贡献指南](CONTRIBUTING.md)，实验范围见 [评测说明](EVALUATION.md)。
