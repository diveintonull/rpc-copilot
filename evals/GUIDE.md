# GRC 评测集编写规范

## 目的

`dataset.jsonl` 是 GRC Copilot 的固定验收题库。它在 RAG 和 Agent 实现之前定义期望行为，用于比较不同检索、Prompt、Graph 和 Skill 配置，而不是为某一次模型输出量身定制标准答案。

禁止把当前 Agent 的回答直接复制为 `gold_points`。Gold 必须来自已收录法规原文、明确的任务边界或可人工判断的拒答规则。

## 一行一个样例

每行是一个 JSON 对象，必须且只能包含以下字段：

| 字段 | 含义 |
|---|---|
| `id` | 全数据集唯一的稳定编号 |
| `question` | 交给系统的原始用户问题 |
| `task_type` | `regulation_qa`、`clause_comparison`、`gap_analysis` 或 `unsupported` |
| `gold_points` | 合格答案必须覆盖的事实或行为要点；拒答题记录拒答原因 |
| `gold_citations` | 支持要点的版本化父章节 ID；拒答题可以为空 |
| `should_refuse` | 是否应该拒绝生成法规结论 |
| `source_versions` | 本题允许使用的 `source_id@version` 列表 |
| `tags` | 数据切片标签，例如 `version_trap`、`colloquial` |

## 引用粒度

- Gold 引用使用父章节 ID，例如 `GBT-22239@2019#7.1.4.1`，不用 Qdrant 内部点 ID或子块序号。
- 每个引用必须属于 `source_versions` 中声明的一个版本。
- 跨法规对比必须至少保留两侧证据，不能只引用其中一侧后生成另一侧结论。
- 同一编号在不同版本中视为不同引用，禁止省略版本。
- 引用存在只证明定位有效；Gold 要点还必须人工核对原文含义。

## 出题来源

1. 从 `data/parsed/_parents_store.json` 选择真实父章节。
2. 根据章节正文编写问题和最小必要 `gold_points`。
3. 把父章节 ID 放入 `gold_citations`，把 `#` 前的文档 ID 放入 `source_versions`。
4. 运行校验器确认字段、ID 和版本关系。
5. 人工对照原文复核问题、要点和引用，不用模型当前答案反推 Gold。

## 拒答规则

以下情况应设置 `should_refuse=true`：

- 问题超出当前法规知识库范围，例如医疗诊断或税务计算。
- 用户要求不存在或受版权限制且未收录的原文。
- 用户要求在没有企业现状和证据时直接证明“完全合规”。
- 用户要求无法由当前语料证明的未来事件或确定性预测。

拒答不是简单返回空字符串。`gold_points` 应记录为什么不能回答，以及需要用户补充什么信息。

## 版本陷阱

`version_trap` 样例必须填写 `source_versions`。答案应明确当前知识库实际收录的版本、替代关系或版本缺口，不能把生效年份、修订年份和法规编号混为一谈。

## v0 分布

当前 30 条数据使用六个互斥主标签：

- `single_regulation`: 10
- `cross_regulation`: 5
- `refusal`: 5
- `colloquial`: 4
- `version_trap`: 3
- `control_gap`: 3

## 校验

```powershell
uv run python -m evals.validate_dataset evals/dataset.jsonl
```

期望输出：

```text
valid=30 invalid=0
```

校验器检查结构，不代替法律或合规专家判断。Task 5 至少人工复核 3 条代表样例；最终评测前必须完成全量 Gold 复核并冻结数据集哈希。

