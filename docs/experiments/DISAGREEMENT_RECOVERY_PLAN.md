# 分歧处理恢复方案

状态：2026-11-07 的研究与实施决策。保留 `51a544c` 的主流程和全部历史结果；
本分支 `experiment/evidence-recovery` 为隔离的实验开发，不是已验证的新默认版本。
三方分歧处理器已作为显式实验入口实现，并完成六条受控故障轨迹的真实模型开发对照；
结果及限制见[恢复实验记录](DISPUTE_RECOVERY_RESULTS.md)。初版 SQL-only 原型仍保留，
其离线测试不代表三方分歧解决效果。下文保留最初的研究依据和实施边界。

## 本次只解决什么

在 SQL、首参考与独立核算产生分歧时，系统应能寻找有区分力的新证据，识别需要
修改的具体程序，保留修订历史并重验。三方都可能出错，不预先赋予独立核算真值地位。
保留 SQL 数据分析的输入输出、LangGraph、隔离角色上下文、受限执行、反例与版本回放；
不增加另一套 Agent 框架、无界辩论或训练裁判模型。

## 已观察到的问题

根据冻结扩展实验与当前源码，以下是事实，而非推测：

- 受控独立核算错误组正确交付 0/20；归因动作只支持 candidate / reference /
  ambiguous，没有 auditor 修复出口。
- `ReferenceAuditNodes.replace_reference` 将 `audit_plan` 直接用作新参考。
  这种替换会降低两份程序的独立性；替换后的相等不能成为新增独立证据。
- 完整协议拒绝了 26 条初始正确公开 SQL 中的 23 条；最终已交付 12 条中仍有
  8 条错误。因此当前问题同时包含不必要拒绝和错误放行，不能只追求减少拒绝。
- 公开评测时，模型运行时的数据源是少量记录预览，最终评分使用完整数据库。
  新的主动取证应访问同一份主机绑定的只读原库，并为基线提供相同访问范围。
  不能把原库 SQL 结果与在预览上计算的参考结果直接比较。
- JSON Schema 确认实验未改善正确交付。协议动作错误、DSL 限制和真正语义分歧
  必须分别记录，不能都算作用户需求不明确。

对应源码入口：`agent/discovery_audit.py`、`agent/discovery_model.py`、
`verification/reference_audit.py`、`agent/discovery_graph.py`。
历史数据参见 [扩展评测](EXPANDED_RESULTS.md) 与
[结构化输出确认](STRUCTURED_CONFIRMATION_RESULTS.md)。

## 开源依据与取舍

下列是截至 2026-11-07 实际查阅的官方仓库、源码或论文。
只迁移可验证的方法；未复制第三方实现，也不继承其论文指标。

### DPC：在真正产生不同结果的数据上比较

[官方仓库](https://github.com/HKUSTDial/DPC) 与
[ACL 论文](https://aclanthology.org/2026.acl-long.313.pdf) 使用最小区分数据库，
让候选 SQL 的差异显现，再用 Python/Pandas 的独立解辅助选择。
仓库还提供
[Solver 可靠性分析](https://github.com/HKUSTDial/DPC/blob/master/baseline/run_solver_reliability_experiment.py)。

迁移点：优先找能区分当前两个假设的输入，复用现有 Hypothesis 搜索与缩减；
对模型构造的数据，先检查类型、主外键和已声明的业务约束。
边界：区分数据证明程序不等价，不自动证明哪一个符合自然语言需求。
Python 也是模型生成的代理答案；合成数据可能违反隐含业务约束。
因此不能直接把 DPC 的 Solver 当作我们新的不可质疑裁判。

### CRITIC：获取工具反馈后再修改

[官方代码](https://github.com/microsoft/ProphetNet/tree/master/CRITIC) 与
[论文](https://arxiv.org/abs/2305.11738) 通过外部工具反馈支持批评与迭代修改。
[程序任务实现](https://github.com/microsoft/ProphetNet/blob/master/CRITIC/src/program/critic.py)
保存逐轮代码、执行结果，并能对比有无工具反馈。

迁移点：调查角色按当前疑点选择只读查询，获取观察后再更新判断。
例如先查询连接基数、字段值分布或过滤前后的记录，而不是只读取三段推理。
边界：查询结果证明数据库中的事实，不会自行确定用户对“收入”等词的业务定义。
观测 ID 真实存在也不意味着模型正确解释了它。

### Magentic-One：事实、猜测与进展分开记录

[官方文档](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/magentic-one.html)、
[协调器实现](https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_magentic_one/_magentic_one_orchestrator.py)
与 [提示词](https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_magentic_one/_prompts.py)
区分已知事实、待查事实和猜测，并记录循环、进展和重新规划。

迁移点：在现有 LangGraph 状态中记录争议及已执行调查；同一程序版本和数据源上的
重复查询可返回缓存，但模型调用和控制循环仍计入固定上限，并记录 no_progress。
连续没有新信息时改换调查方式；达到固定上限后明确退出，不能无限免费重试到一致。
边界：进度管理解决空转，不证明裁判正确。不引入 AutoGen 运行依赖。

### CHESS 与 CHASE-SQL：局部问题和候选比较

[CHESS](https://github.com/ShayanTalaei/CHESS) 生成自然语言测试来比较 SQL 候选。
已核对其
[测试生成](https://github.com/ShayanTalaei/CHESS/blob/main/src/workflow/agents/unit_tester/tool_kit/generate_unit_test.py)
与 [评估源码](https://github.com/ShayanTalaei/CHESS/blob/main/src/workflow/agents/unit_tester/tool_kit/evaluate.py)：
评分依然来自 LLM，不是自动得到真值的 pytest 测试。
[CHASE-SQL 论文](https://arxiv.org/abs/2410.01943) 使用多样候选及经训练的两两选择器。

迁移点：把“哪个程序对”缩小成“此处应按订单还是明细计数”等局部差异。
暂不迁移多候选锦标赛或训练选择器，其候选规模、训练和费用不适合本次最小恢复。

### AgentAuditor：审查关键分歧点

[论文](https://arxiv.org/abs/2602.09341) 比较推理分叉，研究多数共识也会出错的问题。
本文仅将其作为局部分歧审查的研究支持，未核实可复用的作者代码，不将其列为
已接入开源组件，也不复现其训练策略或宣称相同增益。

## 选定的最小处理流程

1. **登记分歧。** 固定任务与数据源，记录 candidate / reference / auditor 的版本、
   相同输入上的不同输出、具体疑点、需求依据和仍未验证的假设。诊断文字仅是观点。
2. **选择下一项调查。** 调查角色提出能区分假设的查询或小样本；主机决定安全性、
   可访问范围和资源上限。先收集事实，再决定修改对象，避免总是修改 SQL。
3. **执行并保存观察。** 只接受工具实际返回的观察，记录输入、数据范围、程序版本、
   错误和截断。模型不能提供伪造结果；运行在预览、合成数据和原库的证据分别标记。
4. **提出局部修订。** 允许 target 为 candidate / reference / auditor。生成新的
   目标程序并解释它解决哪一个分歧，引用现有观察及需求。不能仅因为另两方一致，
   就复制它们覆盖少数方。旧版本始终保留。
5. **重放。** 在同一输入上重跑三份当前程序和全部已保留反例；检查新的执行错误、
   旧反例回归和当前争议。历史模型给出的预期值不因保留而变成真值；若该预期错误，
   可带理由另存新解释，但不删除原数据、原判断和修改记录。
6. **结束或继续。** 当前调查解决后结束有限核验；出现新分歧时继续受预算约束的调查。
   无新证据为 unresolved，工具或协议故障为 error，真正需要额外业务定义才是
   needs_clarification。不能为了提高交付数而静默放行，也不能把所有分歧都交给用户。

最小实施时，三方数值比较限于相同的有界 fixture；现有 DSL 执行器并不直接执行
完整 SQLiteSource。原库查询单独作为事实观察记录，不与预览参考的数值混比，
也不为本次恢复扩建全量 DSL 执行引擎。

有限核验的退出状态沿用 reviewed，必要条件为：所有采用的版本合法且可执行；
三方在保留的有效 fixture 上一致；主机确认样例未回归；已登记的分歧有实际重放记录；
没有未决工具失败或未解决的业务定义。该状态表示协议允许交付，不叫 verified，
不证明语义正确，三方仍可能共同误解。错误地将分歧标为已解决也是评测要揭露的错误。
真实正确性由与运行上下文隔离的评分器或人工确认判断。

最小争议记录需要：任务/数据来源标识、三份程序版本、被质疑的逻辑点、竞争假设、
实际观察及其范围、修订对象、修改理由、重放结果、退出原因。
暂不建立复杂推理树、知识图谱或多个新的长期 Agent。

### 例子：同一份退款记录，误用了绝对值

假设用户明确要求“将带符号金额求和，退款记录按其负值计入”，有效样本中有
100 和 -20。SQL 与首参考得到 80，独立核算因为使用 abs 得到 120。

新版应定位到“是否对金额取绝对值”的差异，查询或构造有效的正负值记录，再根据
明确的需求修改独立核算，重放得到 80。不能为了迎合独立核算，把原本正确的 SQL
也改为 abs。此示例是机制说明，不是新实验成功结果。
若用户仅说“统计收入”而退款口径未知，数据库有负数也不能替用户决定口径。

## 实现边界和当前进度

已在隔离分支实现的组件：

- `verification/sql_tools.py`：主机绑定原库、只读与资源限制、真实观察记录。
- `agent/evidence_model.py` / `agent/evidence_graph.py`：SQL-only 的 query / revise /
  finish 原型，验证独立上下文如何取证和保存版本，不等于三方分歧处理器。
- `verification/dispute_runtime.py`：三份程序的版本、执行分歧、修订尝试与样本回放。
- `agent/dispute_model.py` / `agent/dispute_graph.py`：可查询数据、修订任一程序的
  有界诊断流程。原 `run_discovery` 不变，新入口需要调用方显式传入三份初始程序。
- `evaluation/dispute_recovery.py`：同输入的三方法开发对照。真实实验已完成，
  未替换旧实验；结束动作的引用兼容修复另用原始响应离线回放。

本次复用既有模型适配、DSL 执行、评分和计费。实现了主机绑定的只读查询，
尚未在新诊断循环中加入模型自适应生成区分样本的动作；不能把实验预先提供的
诊断样本说成 Agent 自动发现的证据。下一步只验证真实数据查询的增益，先不扩展架构。

## 验证顺序与成功判据

先在已使用的开发输入上验证机制：SQL 单错、首参考单错、独立核算单错、两方共同错、
三方一致但错误、全部正确、缺少业务定义、查询无进展、无效 DSL 和工具失败。
离线脚本回答只验证状态机与约束，不能代替真实模型效果。
三方一致但错误是分歧触发流程的已知盲点及负对照；不能为了让该组通过，再临时
加一套无分歧审查或个案规则。本次目标是有分歧后的解决能力，不声称覆盖全部错误。

开发联调通过后再冻结方法与题集，在授权余额内进行小规模真实模型配对实验。
固定相同初始 SQL、原库权限和资源限额，比较：

- 简单自修复，允许相同原库执行反馈；
- 无新增主动取证的对应流程；
- 含主动取证与三方修订出口的流程。

分别报告错→对、对→错、正确但未交付、错误交付、未决、请求数和 token；
“保留正确 SQL 并完成交付”与“修好了错误 SQL”分别统计。
数据量和置信区间如实列出。新公开题只算既有 schema 内的新题，不说成新数据库泛化。
开发无净收益时不动留出集，不加个案规则后继续冒充冻结测试。

所有付费调用复用累计 ¥120 授权和原有四条账本，不退款预留、不重新开额度。
本方案不承诺提升幅度；完成标准是能用实际记录说明改进了哪些分歧，以及仍失败在哪。
