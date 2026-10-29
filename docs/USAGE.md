# 使用指南

## 环境

已验证环境为 Windows、Python 3.12。核验组件与 LangGraph 路径使用基础环境，DB-GPT 路径另有原生环境。安装命令在仓库根目录执行。

## 重放一个反例

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt -r requirements-probes.txt
.\.venv\Scripts\python.exe tools/replay_probe.py examples/public_evidence/date_boundary.json
```

该示例固定一条错误候选 SQL，在原数据及加入一条范围外记录后的数据上各运行一次。`reproduced=true` 表示重新观察到同一性质违规；这是反例重放成功，不是候选 SQL 通过。

输出中 `before=[[1]]`、`after=[[2]]` 表示日期上界错误把新记录计入。`record_count=4` 是原数据和新增数据的去重记录总数：两个订单、两条明细。

## 运行 LangGraph 闭环

完成上面的基础依赖安装后：

```powershell
.\.venv\Scripts\python.exe tools/run_graph.py --case d06 --thread boundary-demo
```

该命令用离线响应走完整生成、核验、反馈、回归与发布链路。检查点保存在本地 SQLite，命令输出结果状态和报告位置。断点恢复与真实模型命令见 [LangGraph 指南](LANGGRAPH.md)。

## 运行 DB-GPT 闭环

另需 Git 和带 pip 的 Python 3.12：

```powershell
.\tools\setup_dbgpt.ps1 -Python python -Git git
.\.venv-dbgpt\Scripts\python.exe tools/run_probe_workflow.py
```

安装脚本获取固定版本 DB-GPT 源码，创建 `.venv-dbgpt` 并安装原生链路与自动测试依赖。运行命令使用脚本客户端，检查候选、反馈、重试与结果发布的接入流程，不调用模型 API。

运行结束后，终端给出新报告目录。`workflow.json` 记录阶段状态，子目录的 `summary.json` 给出任务结果，`traces/` 保存逐步轨迹。应用输出读取 `publication`：

- `status=verified`：候选满足本次登记检查，`rows` 为核验器持有的结果。
- `status=unverified`：检查失败或任务未完成，公开结果不包含候选数值。

这些状态不等于任意 SQL 的完整语义证明；支持范围见 [架构与执行契约](ARCHITECTURE.md#执行契约)。

## 接入真实模型

已验证普通百炼 `qwen3.8-flash`，通过文本 JSON Action 协议接入 DB-GPT。API 配置只从本机环境读取；`.env.example` 是字段说明，不会自动加载。

先在本机环境变量界面设置 `DASHSCOPE_API_KEY`，随后新开 PowerShell，再配置服务地址和模型。密钥不写入项目文件：

```powershell
$env:DASHSCOPE_BASE_URL = "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
$env:DASHSCOPE_MODEL = "qwen3.8-flash"
```

将占位地址替换为普通 API 业务空间地址。配置完成后，可运行一项自然任务：

```powershell
.\.venv-dbgpt\Scripts\python.exe tools/evaluate_probe_agent.py --live --natural-only --case-ids d01 --seeds 17 --modes compact --max-requests 2 --max-attempts 2 --money-limit-cny 0.1
```

`--live` 会产生 API 用量。上例最多两次实际 HTTP 尝试；调用前做费用预留，usage 完整时记录标准价估算。Token Plan 订阅密钥与普通按量接口是不同配置，此入口使用普通 API。API 参数说明见 [阿里云兼容接口](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)。

## 运行评测

以下命令离线执行：

```powershell
# 当前验证器对已知挑战的回归
.\.venv\Scripts\python.exe tools/evaluate_independent_holdout.py

# 在独立进程加载修改前源码，重现初次挑战结果
.\.venv\Scripts\python.exe tools/replay_initial_challenge.py

# 组件数据生成、缩减与诊断对照
.\.venv\Scripts\python.exe tools/evaluate_probes.py --seeds 17 31
```

公开示例存在时，第一条命令默认做修复后的已知回归，第二条命令重放修改前的21/24结果。它们使用同一组题目，不能当作两批独立样本。完整方法见 [评测说明](EVALUATION.md)。

反馈机制另有四组消融和三种策略对照入口。以下命令离线演示连续失败后的证据升级：

```powershell
.\.venv-dbgpt\Scripts\python.exe tools/evaluate_feedback_strategies.py --stage strategy --controlled-case-ids d06 --natural-case-ids --seeds 53 --max-attempts 3
```

这项离线演示使用脚本响应，不计为模型修复结果。开关、真实实验的续账要求和指标见 [反馈机制说明](FEEDBACK.md)。

## 运行测试

核验组件及早期服务：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

原生 Agent 环境未安装 LangGraph 时，其专属集成测试会跳过；不安装早期 FastAPI 服务时，运行对应模块：

```powershell
.\.venv-dbgpt\Scripts\python.exe -m pytest -q --ignore=tests/test_engine.py --ignore=tests/test_interfaces.py --ignore=tests/test_source_consistency.py
```

`pytest.ini` 将默认收集范围限定为正式 `tests/`，避免收集实验快照、依赖或发布副本中的同名文件。Windows 中异步测试需允许本机回环。
