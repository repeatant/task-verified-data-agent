# 使用指南

## 环境

当前两个核验流程均由 LangGraph 编排，使用同一个 Python 环境。已验证开发环境为 Windows、Python 3.12；命令在仓库根目录执行。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/core.lock.txt -r requirements/probes.txt
```

## 动态核验

模型提出参考计算与测试，通过执行差异选择修改 SQL、替换参考或澄清。先运行离线开发用例：

```powershell
.\.venv\Scripts\python.exe tools/evaluate_discovery.py --track controlled
.\.venv\Scripts\python.exe tools/evaluate_discovery.py --track correct
.\.venv\Scripts\python.exe tools/evaluate_reference_audit.py
```

前两条分别检查错误候选修复与正确候选保留；第三条覆盖九种参考审查场景。模型响应为预写脚本，结果用于验证工程接线，不代表真实模型正确率。报告写入新目录，终端给出位置。

动态入口是库函数 `dataverify.agent.discovery_graph.run_discovery`，命令行通过 `tools/evaluate_discovery.py` 调用。自定义任务、角色边界与真实实验参数见 [动态核验指南](DISCOVERY.md)。

## 预设性质核验

调用方确认任务契约后，检查六类已登记性质。该路径保持原有验收条件，可独立运行：

```powershell
.\.venv\Scripts\python.exe tools/run_graph.py --case d06 --thread boundary-demo
.\.venv\Scripts\python.exe tools/replay_probe.py tests/fixtures/public_evidence/date_boundary.json
```

第一条使用离线响应走完生成、核验、反馈、回归与发布。检查点保存在本地 SQLite；新任务需新的 `--thread`。暂停、恢复和真实调用见 [LangGraph 指南](LANGGRAPH.md)。

第二条重放日期上界错误。`before=[[1]]`、`after=[[2]]` 表示范围外记录被错误计入；`reproduced=true` 表示反例重现，不是候选通过。`record_count=4` 是两次数据去重后的记录总数。

## 接入真实模型

模型调用采用文本结构化 JSON 协议，由本地代码解析和分发；不是供应商原生 Function Calling。现有开发证据使用普通百炼 `qwen3.8-flash`，凭据只从本机环境读取，`.env.example` 不会自动加载。

在本机环境变量中设置 `DASHSCOPE_API_KEY`，新开终端后配置普通 API 业务空间地址和模型：

```powershell
$env:DASHSCOPE_BASE_URL = "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
$env:DASHSCOPE_MODEL = "qwen3.8-flash"
```

替换占位地址；密钥不写进源码、报告或 Git。此配置使用普通按量 API，不使用 Token Plan 订阅密钥。接口说明见 [阿里云兼容接口](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)。

首次使用且没有既有实验开销时，以下命令创建本地 0.1 元额度，最多追加两次 HTTP：

```powershell
.\.venv\Scripts\python.exe tools/run_graph.py --live --track natural --case d01 --thread first-live --budget-file artifacts/langgraph/budget.json --new-budget-cny 0.1 --max-extra-requests 2
```

`--live` 才会产生调用。已有实验必须续接累计账本，不能通过新建文件重置开销。调用前做费用预留，完整 usage 用于本地估算；技术额度不是服务商侧扣费上限。续账和恢复见 [LangGraph 指南](LANGGRAPH.md#真实千问调用)；动态角色共用预算，详见 [动态核验指南](DISCOVERY.md#运行)。

## 离线评测与历史回放

```powershell
# 当前预设核验器对已知挑战的回归
.\.venv\Scripts\python.exe tools/evaluate_independent_holdout.py

# 在独立进程加载冻结旧源码
.\.venv\Scripts\python.exe tools/replay_initial_challenge.py

# 生成、缩减与诊断对照
.\.venv\Scripts\python.exe tools/evaluate_probes.py --seeds 17 31
```

前两条分别检查修复后 24/24 与初次 21/24；同一批题目不能当作两批独立样本。当前回归默认将本次实际核验源码摘要写入报告目录，不沿用历史 manifest 充当当前版本。冻结旧源码含旧适配文件，但回放只加载纯核验模块，无需安装 DB-GPT。旧反馈实验不再提供当前执行入口，其结果见 [历史实验](history/LEGACY_EVALUATION.md)。

## 运行测试

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

`pytest.ini` 将收集范围限定为正式 `tests/`，不收集冻结历史实现和发布副本。测试默认离线；Windows 中异步测试需要本机回环可用。代码检查见 [贡献指南](CONTRIBUTING.md)。
