# 反馈机制与对照实验

SQL 被拒绝后，Agent 需要知道失败原因、可用语法及能揭露错误的数据。本模块将这三类信息分开组织，并记录反馈是否进入下一次模型请求。

## 四组消融

所有组都保留原有的基础 `supported_sql` 描述，核验和发布条件完全一致。四组只改变额外完整执行契约与分类修复指导：

- `legacy_v1`：基础说明和原反馈。
- `capability_only`：额外提供完整 `execution_contract`。
- `guidance_only`：失败时提供对应的 `repair_guidance`。
- `capability_v2`：同时提供上述两项。

这项实验回答两种新增信息的边际作用及交互，不能解释为“完全没有能力说明”与“有能力说明”的比较。

小规模开发对照已运行。左关联筛选错误中，基础反馈用了两次真实修复调用，三种增强组各用一次；另一条分组错误四组均一次成功。这个结果支持具体案例的修复路径分析，不足以给出普遍成功率提升。

## 自适应反馈

`full` 与 `compact` 在任务内固定反馈格式。`adaptive` 从紧凑证据开始，观察以下运行状态：

1. 规范化 AST 与历史失败候选重复；保留字面量，避免把不同筛选条件合并。
2. 相邻失败使用相同错误码。
3. 连续两次失败，即使 SQL 和错误类别不同。

满足任一条件，从本次失败反馈起升级到完整证据，并在该任务内保持。新任务使用新的 Guard，历史不跨任务共享。成功会清除连续失败计数；已经升级的任务不再降级。

策略只依赖候选与运行反馈，不读取题号、标准 SQL 或正确答案。没有生成反例时，反馈明确记录 `evidence_available=false`；选择完整模式不会凭空产生反例，也不会改变验收条件或额外增加修复轮数。

`feedback_strategy` 记录选择的格式、升级原因与证据可用性。实验另区分模式升级、完整失败反馈的发出，以及它是否进入后续实际 HTTP 请求。最后一次尝试才产生的升级反馈不能计作模型使用过的反馈。

本轮真实三策略对照均完成 3/3，自适应组没有触发升级，因此升级效果仍待出现连续失败的任务验证。默认模式保持 `compact`。重复失败的离线接线记录见 [升级示例](../examples/public_evidence/adaptive_replay.json)。

## 离线演示

在已安装原生 Agent 依赖的环境执行：

```powershell
.\.venv-dbgpt\Scripts\python.exe tools/evaluate_feedback_strategies.py --stage strategy --controlled-case-ids d06 --natural-case-ids --seeds 53 --max-attempts 3
```

该命令使用脚本依次返回错误候选、重复错误候选和正确候选，演示紧凑反馈、升级与回归。它不调用模型 API，只证明接入流程可运行，不代表模型修复能力。

运行后查看终端给出的报告目录：

- `frozen_manifest.json`：运行前固定的任务、实验分组、预算和源码摘要。
- `summary.json`：各组结果、实际调用、用量、失败及未运行项。
- `traces/`：逐项候选、核验、反馈和发布轨迹。
- `request_audit.json`：本次实际 HTTP 审计；离线为空。

## 真实实验与预算

入口 `tools/evaluate_feedback_strategies.py` 默认离线，`--live` 才发送模型请求。继续真实实验必须通过 `--prior-budget-json` 指向最近一次完整账本，保留原有请求、Token、费用预留及用量记录。

每次 POST 前分别检查累计预算和本阶段追加额度。追加额度耗尽时保留未运行项，不能把它们从预定分母中删除。费用为按标准价计算的本地估算；定价见 [阿里云模型说明](https://help.aliyun.com/zh/model-studio/qwen3-8-flash)。

一次响应截断曾中止实验。`--resume-after-truncation` 是窄范围续账入口：只允许末次 HTTP 200、900 输出 Token 截断，且所有事件、用量与历史账本逐项匹配。续账保留失败单元、费用、来源报告摘要和原停止原因；未知用量、传输错误与其他 provider 错误仍拒绝恢复。`--arms` 可只执行预定但尚未运行的组，补充报告单独保存。

消融与反馈策略分阶段运行，各组使用相同任务、源数据、模型与最大尝试数，组顺序按任务轮换。受控首轮是注入测试输入，后续真实调用单独计数；自然任务从首次生成开始调用真实模型。

当前小规模实验使用已有开发任务，不作为未知模板、跨 schema 或生产表现的证据。生成种子改变测试数据，不是模型采样种子。具体结果见 [评测说明](EVALUATION.md)。

## 源码

- [probe_capabilities.py](../dataverify/probe_capabilities.py)：独立信息开关与分类指导。
- [probe_guard.py](../dataverify/probe_guard.py)：运行状态、证据选择与发布控制。
- [probe_agent_bridge.py](../dataverify/probe_agent_bridge.py)：原生资源和反馈接入。
- [feedback_experiment.py](../dataverify/feedback_experiment.py)：分组、评分、预算续接和轨迹记录。
