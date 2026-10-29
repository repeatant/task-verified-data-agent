# 源码阅读指南

建议先读一次任务的编排入口，再按调用顺序进入具体阶段。源码保持现有公开接口，历史实验和冻结快照使用原版本的摘要。

## 一次 Agent 任务

从 [run_probe_native_case](../dataverify/probe_agent_bridge.py) 开始：输入验证 → 创建任务资源 → 原生 Agent 运行 → 资源清理 → 运行记录。

`_NativeSession` 表示一次任务拥有的客户端、资源和 Agent；`_NativeOutcome` 收集回复、事件与脱敏后的错误。它们将资源生命周期和最终记录分开，生成与重试仍由 DB-GPT 原生循环负责。

## 候选如何验收

[ProbeGuard.execute](../dataverify/probe_guard.py) 按顺序完成：确认任务未变 → 检查候选 → 记住反例 → 回归已有反例 → 选择反馈 → 保存审计。

每次候选更新先清除旧接受状态。`publish()` 只返回当前仍有效的核验结果。`_RegressionRun` 将回归判定和查询消耗放在一起，避免汇总时丢失预算信息。

## LangGraph 如何编排与恢复

从 [run_graph_case](../dataverify/probe_langgraph.py) 开始：验证线程绑定 → 加载检查点 → 执行状态图 → 整理结果。`_GraphNodes` 把节点与路由放在一起，模型及数据作为运行依赖传入，避免把活动客户端放进检查点。

`prepare_call` 保存调用意图，`generate` 只生成一次候选，`verify` 委托同一个 Guard，`feedback` 追加本轮诊断，`publish` 再次核验并获取结果。

[graph_guard.py](../dataverify/graph_guard.py) 校验并恢复反例与策略状态；接受状态不参与恢复。[graph_checkpoint.py](../dataverify/graph_checkpoint.py) 管理本地写入锁和费用落盘；[graph_model.py](../dataverify/graph_model.py) 只负责单次文本 JSON 调用，不包含另一个 Agent 循环。恢复测试关注请求重复、反例丢失、来源变化与旧结果泄漏，见 [LangGraph 指南](LANGGRAPH.md)。

## SQL 如何检查

[probe_runtime.py](../dataverify/probe_runtime.py) 的 `_analyse` 分为四个阶段：

1. `_parse_select`：读取唯一 SELECT，限制语法范围。
2. `_analyse_relations`：解析表别名，检查登记的父子表关联。
3. `_analyse_where`：收集受支持的筛选条件。
4. `_analyse_projection`：检查度量、输出列及分组键。

随后 `_semantics` 对照调用方确认的任务条件，`execute_sql` 在隔离、只读的 SQLite 中执行。语法许可与业务任务要求分别检查，便于定位“当前不支持”和“答案范围错误”。

## 反馈为何升级

[feedback_strategy.py](../dataverify/feedback_strategy.py) 只维护当前任务的候选指纹和失败状态，输出证据格式与升级原因。SQL 验收、预算和发布由 Guard 继续负责。

`candidate_fingerprint` 忽略格式与注释，同时保留字面量和多语句内容。`FeedbackStrategy.select` 处理观察结果，`metadata` 输出模型可见的策略说明。没有反例时不会生成伪证据。

## 评测和预算

[feedback_experiment.py](../dataverify/feedback_experiment.py) 负责配置、冻结计划、单项执行和报告检查点；[probe_live_benchmark.py](../dataverify/probe_live_benchmark.py) 保留完整／紧凑反例的原有配对实验入口。小型运行上下文集中保存共享状态，单项成功、异常和中断共用报告落盘路径。

[experiment_budget.py](../dataverify/experiment_budget.py) 单独处理账本恢复、截断审查与追加额度；[probe_cost_budget.py](../dataverify/probe_cost_budget.py) 负责每次实际请求前的费用预留及用量记账。外部依赖和模型调用由对应客户端管理。

[reporting.py](../dataverify/reporting.py) 提供无模型依赖的报告写入、稳定 JSON 摘要及用量汇总。未知或不一致的 usage 会单独计数。

## 工程方式的参考

本项目参考以下官方源码的组织方式，按自身规模采用小函数、显式状态及独立生命周期，不复制上游业务实现：

- [pytest runner](https://github.com/pytest-dev/pytest/blob/main/src/_pytest/runner.py)：阶段编排与单项运行分开。
- [Pydantic AI RunContext](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/_run_context.py)：显式描述运行上下文。
- [smolagents memory](https://github.com/huggingface/smolagents/blob/main/src/smolagents/memory.py)：区分状态记录与序列化边界。
- [SQLGlot 配置](https://github.com/tobymao/sqlglot/blob/main/pyproject.toml)：将统一行宽与静态检查放在项目配置中。

修改前的检查步骤见 [贡献指南](../CONTRIBUTING.md)，行为与评测边界见 [评测说明](EVALUATION.md)。
