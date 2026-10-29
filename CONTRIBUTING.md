# 开发与代码约定

## 本地检查

在仓库根目录安装开发工具。模型运行依赖按 [使用指南](docs/USAGE.md) 单独安装。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m ruff check dataverify tools tests
.\.venv\Scripts\python.exe -m ruff format --check dataverify tools tests
.\.venv\Scripts\python.exe -m pytest -q
```

原生 DB-GPT 环境的测试范围与早期服务不同，命令见 [运行测试](docs/USAGE.md#运行测试)。测试默认离线，真实模型实验必须显式指定 `--live` 并继续已有用量账本。

## 组织代码

- 入口函数表达执行顺序，具体阶段交给有明确名称的辅助函数。共享运行状态较多时，用小型 dataclass 表达归属。
- 同一模块围绕一种职责组织：SQL 语法与语义、性质测试、验收与发布、反馈选择、模型接入、实验评分、预算续账分别处理。
- 参数和返回值补充类型，公共函数用简短 docstring 说明作用。外部 JSON 保留显式校验，类型注解不代替运行时检查。
- 注释解释约束和取舍，例如为什么修改候选前必须撤销旧结果、为什么未知用量不能记为零；避免逐句复述代码。
- 报告序列化、JSON 摘要与用量汇总复用 `dataverify.reporting`，避免评测模块互相导入私有函数。
- 原始异常和模型响应经过既有脱敏边界处理。新增日志只记录定位问题需要的字段。

## 修改行为前后

保持公开接口、错误码和报告字段稳定。涉及任务验收、SQL 规则或反馈选择时，覆盖合法候选、失败候选、历史反例回归及旧结果失效。

评测答案只存在于评测侧；运行时模块不能导入参考 SQL、故障标签或预期结果。预算和发布条件共用同一套检查，不能因为调整反馈方式而放宽。

`examples/independent_holdout/initial_verifier/` 是历史实现快照，`reports/` 中的冻结数据与原始实验记录也保留原样。格式化和静态检查已排除这些目录；新验证使用新输出目录，不覆盖旧证据。

## 源码入口

按 [源码阅读指南](docs/CODE_GUIDE.md) 可以从一次任务入口追踪到最终结果，并找到各阶段的测试。
