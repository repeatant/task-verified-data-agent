# 反馈与修复

预设性质路径将错误原因、支持语法和可重放数据交给生成方；动态路径还要区分 SQL 错误与参考错误。两者共用部分工具，但修改权限不同。

## 预设性质反馈

`ProbeGuard` 保持任务与验收条件，反馈配置只改变模型看到的信息：

- `legacy_v1`：基础 `supported_sql` 和原反馈。
- `capability_only`：额外完整执行契约。
- `guidance_only`：额外分类修复指导。
- `capability_v2`：完整契约与分类指导。

`full` 和 `compact` 固定证据格式。`adaptive` 从紧凑证据开始，遇到重复候选、相邻同类失败或连续两次失败时升级为完整证据，在当前任务保持。AST 指纹忽略格式与注释，保留字面量；状态不跨任务共享。

没有反例时记录 `evidence_available=false`。升级不会制造证据、放宽验收或增加修复次数。最后一轮才产生的反馈也不能计为模型已使用。

```powershell
.\.venv\Scripts\python.exe tools/run_graph.py --case d06 --thread feedback-demo --mode adaptive
```

该命令离线运行一条错误候选及修复候选，验证反馈与回归接线；它不保证触发升级，也不衡量模型能力。重复失败的历史脚本轨迹见 [升级记录](../tests/fixtures/public_evidence/adaptive_replay.json)。

## 动态参考核验反馈

执行不一致先进入归因。核验角色必须引用需求原文和真实 `check_id`，工具检查支持方是否在保留数据上相符，以及是否违反调用方确认样例。只有获准修改 SQL 时才反馈给生成方；错误参考可以替换为独立核算版本，旧版本与反例不删除。

常规 `repair` 动作不能绕过归因直接改 SQL。引文可追溯不等于推理正确，无外部确认的授权仍为 `model_supported_not_proven`。详见 [动态核验指南](DISCOVERY.md#修改权限与业务依据)。

## 实验边界

旧版完整／紧凑反馈、四组消融和三策略对照运行于 DB-GPT。相关 CLI 已移除，原结果在 [历史实验](history/LEGACY_EVALUATION.md) 中保留，不作为 LangGraph 结果。那轮真实三策略对照未触发自适应升级，因此没有证明升级提高成功率。

当前真实调用、费用续账和恢复入口见 [LangGraph 指南](LANGGRAPH.md)。每次调用都记录实际请求；离线脚本响应、注入候选和模型自然生成分别统计。

## 源码

- [probe_capabilities.py](../dataverify/verification/probe_capabilities.py)：信息开关及分类指导。
- [feedback_strategy.py](../dataverify/agent/feedback_strategy.py)：候选指纹、升级状态与原因。
- [probe_guard.py](../dataverify/verification/probe_guard.py)：反例、验收及发布。
- [probe_langgraph.py](../dataverify/agent/probe_langgraph.py)：预设性质图与反馈路由。
- [discovery_audit.py](../dataverify/agent/discovery_audit.py)：归因与修改权限。
