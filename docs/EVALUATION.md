# 评测说明

当前执行路径只使用 LangGraph。动态参考核验与预设性质核验使用不同契约，分别评价，不合并计算准确率。离线脚本证明控制流可运行；真实模型轨迹说明具体行为，两者不能互换。

## v1.0.0 真实 API 联调（2026-11-07）

本轮在仅含 LangGraph 的运行路径上调用真实千问，验证主要执行分支。首轮和补测分开记录，不以最后成功的轨迹替代失败历史。

- 首轮 `wrong_reference_rounding`：3 次请求后因参考程序使用非法 DSL 语法未完成。
- 通用 AST 诊断完善后，同一任务补测使用 6 次请求：保留正确 SQL，替换错误参考并保留旧版本。
- `wrong_sql_grain` 使用 6 次请求修复 SQL 的统计粒度，保留正确参考。
- `requested_rounding` 使用 5 次请求保留需求明确要求的舍入，未把所有舍入都当作错误。
- `distinct_orders` 使用 5 次请求完成自然生成，源数据输出 `[[3]]`；除内置五组 SQL 评分外，SQL 与两份参考通过七组开发评分数据。
- 固定路径 `d03` 使用 1 次请求自然生成 SQL，在 `verify` 后暂停并跨进程恢复；恢复新增请求为零。

三条完成的参考审查任务和自然任务在运行结束后同时核验最终 SQL、当前参考和独立核算，均通过七组开发评分数据：源数据及六组补充数据，覆盖粒度、空输入、循环小数、半值舍入、退款和边界。初始 SQL 与首版参考在三条审查任务中为受控注入；自然任务没有这两项注入。检查点的两份进程报告属于同一个恢复任务，不能当作两个独立样本。

本轮合计 26 次 HTTP 调用、53,194 Token，全部 usage 完整。沿用本地账本标价估算的增量为 0.0610118 元，累计估价为 0.4589466 元；保守预留累计为 4.9928016 / 5 元。估价与预留不是云控制台实际扣费，不通过重新建立账本重置累计消费。

脱敏记录见 [v1.0.0 联调证据](../tests/fixtures/discovery/v1_live_validation.json)。原始私有报告保存在本地 `reports/v1-live-20261107/`，不随源码发布。这些是包含诊断改动和补测的开发联调，没有覆盖所有任务组合，也不能推算模型准确率。

## 历史参考审查开发记录

九种离线场景检查错误参考替换、正确 SQL 保留、错误 SQL 修复、明确舍入要求和共同误解。无确认样例时仍可能错误放行，作为已知失败保留。

真实千问实验受控注入初始 SQL／参考，随后独立核算、归因与修改调用模型。六条开发记录包含格式中止、半值舍入错误和后续补测；三条最终选定轨迹分别完成参考替换、SQL 修复和明确舍入保留，不能按这三条计算冻结成功率。

[reference_audit_scoring.py](../dataverify/evaluation/reference_audit_scoring.py) 在运行后同时检查 SQL、当前参考和独立核算。三条最终轨迹通过七组开发数据，包含粒度、空数据、精度、退款及边界；评分答案不进入当次模型上下文，但数据参与了迭代分析，不是独立留出集。

方法、开销、失败及上下文边界见 [动态核验指南](DISCOVERY.md#参考审查改造的实测与局限)，原始投影见 [参考审查证据](../tests/fixtures/discovery/reference_audit_evidence.json)。改造前曾出现 SQL 评分通过、参考程序仍错误舍入的问题，见 [历史动态证据](../tests/fixtures/discovery/live_evidence.json)。

## 预设性质 LangGraph 执行与恢复

三项历史千问联调使用 `qwen3.8-flash`、`capability_v2`、自适应反馈及最多三个候选尝试：

- `d06`：受控日期上界错误，一次真实调用修复并通过历史反例回归。
- `d10`：受控 LEFT JOIN 过滤错误，一次真实调用修复并保留零值订单。
- `d03`：自然生成后暂停，在另一进程恢复、重新核验并发布；恢复阶段新增 HTTP 为零。

四份进程报告对应三个任务。两个受控首轮不算模型生成，实际 HTTP 共 3 次、3,479 Token，usage 完整。源数据、发布 SQL 一致性及附加边界数据评分通过，见 [联调证据](../tests/fixtures/public_evidence/langgraph_validation.json)。

`tools/evaluate_graph.py` 对已保存的进程报告离线评分。恢复线程按一次任务统计，实际调用分别保留；评分器不进入运行时核验器或模型上下文。这项验证说明图路由与检查点连通，不证明框架更换提升修复率，也不承诺未决 HTTP 的自动恢复。

## 离线回归与冻结历史

```powershell
.\.venv\Scripts\python.exe tools/evaluate_reference_audit.py
.\.venv\Scripts\python.exe tools/evaluate_discovery.py --track controlled
.\.venv\Scripts\python.exe tools/evaluate_discovery.py --track correct
.\.venv\Scripts\python.exe tools/replay_probe.py tests/fixtures/public_evidence/date_boundary.json
.\.venv\Scripts\python.exe tools/replay_initial_challenge.py
.\.venv\Scripts\python.exe tools/evaluate_independent_holdout.py
```

最后两条分别重放冻结旧实现的 21/24，以及修复后的同题 24/24。新工具名仍带 `holdout`，但当前题目已经公开，运行结果是已知回归，不是新的留出成绩。当前回归会记录实际使用的核验源码摘要。

`tests/fixtures/independent_holdout/initial_verifier/` 及其 manifest 保持原样，包含旧适配文件；历史回放只加载纯核验组件，不需要 DB-GPT。旧 JSON 报告可用 `tools/audit_probe_workflow.py` 离线审计，它不运行当前 Agent，也不产生新版指标。

## 历史 DB-GPT 实验

旧反馈实验中，16 组有效配对的完整与紧凑反馈均修复 14/16；诊断字符减少 29.1%，实际输入 Token 减少 7.0%。另三项自然任务分别通过 3/3、2/3，因此不能宣称整体成功率不降。

这些结果及反馈消融来自已移除的 DB-GPT 路径，未在当前 LangGraph 版本重跑。完整分组、失败、范围及历史测试口径见 [历史实验档案](history/LEGACY_EVALUATION.md)。本次迁移不重写旧证据，不把框架移除计作算法提升。

## 适用范围

实验使用固定 `orders` / `order_lines` 表与合成开发数据，尚无跨 schema 或生产准确率证据。预设检查是必要条件；动态生成的参考也可能错。上下文分离降低部分信息诱导，但同一模型可能共同误解；多个程序一致不构成业务正确性证明。

2026-11-07 最终离线回归 1,077 项通过，Ruff 检查及格式检查通过。工程回归在未安装 DB-GPT 的环境中运行；额外隔离测试阻断其导入、凭据读取和模型 HTTP 请求，并实际运行预设与动态两条 CLI 的修复流程。测试通过数衡量工程回归，不是模型准确率。旧版测试数量保留在历史档案中，不沿用为本版通过数。

本地生成的测试报告为 `reports/v1-live-20261107/final-regression.xml`，不随源码上传。测试另有一条 Hypothesis 关于 `.hypothesis` 收集排除的非失败警告。
