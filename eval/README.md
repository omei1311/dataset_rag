# 可复现的检索评测

目录包含两个入口，共用 [metrics.py](metrics.py) 的严格指标实现：

- `offline_eval`：只读取已保存的候选排名，纯 Python 标准库；不加载 `.env`、模型或 Milvus。
- `run_eval`：按已标注的 `item_names` 运行真实检索配置；需要项目依赖、模型与服务。它是给定检索范围的 oracle 实验，不是端到端问答评测。

仓库没有提供真实论文题集上的效果成绩。自带题集和候选列表是手写合成数据，用于确认评测代码能运行。

## 离线演示

在仓库根目录执行，无需 `pip install`：

```bash
python -m eval.offline_eval --questions eval/questions.example.json --candidates eval/candidates.example.json --output-dir eval/reports/demo
```

也可用 `python -S -m eval.offline_eval` 验证不依赖 site-packages。输出 `report.json` 与 `report.md`，两份都明确标注 **SYNTHETIC / 合成演示，不是实际模型成绩**。默认计算 K=1、3、5，可用 `--ks 1,5,10` 修改；报告会覆盖指定目录内同名结果文件。

```bash
python -m unittest discover -s offline_tests -v
```

上述测试覆盖指标边界、问题集验证与命令行输出。实际服务检查仍需独立执行，离线测试通过不说明模型效果达标。

## 问题集格式

顶层是非空 JSON 数组，每个 `question_id` 唯一：

```json
[
  {
    "question_id": "real-q001",
    "question": "需要从真实文档验证的问题",
    "answerable": true,
    "relevant_chunks": [
      {"document_id": "真实文档的标识或file_title", "chunk_id": 0}
    ]
  },
  {
    "question_id": "real-q002",
    "question": "已人工确认语料无法回答的问题",
    "answerable": false,
    "relevant_chunks": []
  }
]
```

- 可回答题至少标注一个真实证据片段；跨文档题可在 `relevant_chunks` 标多个文档与片段。
- 文档字段优先使用 `document_id`。当前本地语料若只存 `file_title`，运行归一化会将它用作文档标识；该字段应在评测语料范围内唯一。文件标题重名时，应先补稳定文档 ID，不能声称能区分这类文档。
- 也可明确标注 `{"file_title":"实际标题","chunk_id":0}`，此时候选必须携带相同的 `file_title`，缺失时不放宽匹配。
- `chunk_id=0` 合法；整数 ID 和等值字符串 ID 会统一成字符串。布尔值、浮点值与空 ID 被拒绝。
- 无答案题需 `answerable:false` 且证据列表为空。空标注不能自动变成正确拒答。
- 为兼容旧标注，允许 `expected_chunk_ids` 搭配 `expected_file_title` 或 `expected_document_id`；仅关键词或裸 chunk ID 标注不接受。历史说明书问题模板也不代表论文问答测试集。
- 真正的 live 题集应补充每题 `item_names`，并删除示例中的 `synthetic:true`；这些检索范围来自人工标注，因此必须披露 oracle 条件。

## 候选文件格式

```json
{
  "synthetic": false,
  "configs": {
    "baseline": {
      "real-q001": [{"source":"local","document_id":"真实文档的标识或file_title","chunk_id":0}],
      "real-q002": []
    }
  }
}
```

每个配置都要覆盖题集的全部 question_id，未知或遗漏的 ID 会报错，避免只提交容易的问题。每题候选按排名顺序保存，支持附加文本与分数，但命中不依赖关键词。网页候选标记 `source:"web"`，不命中当前本地证据标注。缺文档或 chunk 元数据的本地候选会计入诊断项并按未命中处理，不能因元数据丢失反而得到宽松分数。

## 指标口径

| 指标 | 计算方法 |
|---|---|
| Hit@K | 可回答题中，前 K 个不同候选命中任一标注片段的比例 |
| Recall@K | 每题前 K 个候选命中的不同标注片段数 / 该题标注片段总数，再对可回答题做宏平均 |
| MRR | 可回答题中首次命中的倒数排名均值，未命中为 0；排名取完整提交候选列表，非 MRR@K |
| 无答案题空候选率 | 无答案题中返回空候选列表的比例；不称为拒答准确率 |

先按文档与 chunk 证据身份去重、保留首次出现，再计算排名和 K；重复候选不能增加 Recall。只包含无答案题时，Hit/Recall/MRR 为 `null`，不输出虚假的 100%。这些指标仅评价检索，不评价回答事实正确性、引用是否支持结论或模型拒答能力。标注不完整时，未标注候选会按不相关计，相关性与召回估计可能有偏，结果需结合逐题检查解释。

## Live：给定范围的检索实验

导出自己的真实语料以便人工标注：

```bash
python eval/run_eval.py --dump-corpus --output-dir eval/reports/annotation
python eval/run_eval.py --questions eval/questions.real.json --check-questions
python eval/run_eval.py --questions eval/questions.real.json --configs C0,C1,C2 --skip-web --topk 10 --output-dir eval/reports/live
```

`questions.real.json` 由使用者根据导出的证据标注，仓库不伪造该文件。导出会包含语料原文，结果默认不提交版本库。`--help` 和导入 live 模块不加载应用依赖；真正导出、校验或检索时才加载。

| 配置 | 实际路径 |
|---|---|
| C0 | BGE dense+sparse 混合检索 |
| C1 | C0 + HyDE 检索，再用 RRF 融合 |
| C2 | C1 + 运行节点相同的重排与分数截断策略 |
| C3 | C2 中加入网页候选；网页不命中本地证据 GT |

报告明确记录以下限制：

- `item_names` 直接来自标注，跳过产品/实体识别和问题重写，不是端到端测试；不运行回答生成。
- 检索和 RRF 的 topk 可由命令指定；精排遵循运行节点的最大候选数与分数截断，所以配置间最终列表长度可能不同。
- 重排模型失败时使用与运行节点相同的降级策略，单独统计降级次数，不能把降级结果称为重排模型成绩。
- 运行失败的问题保留在分母，候选按空列表处理并记录错误；CLI 返回非零状态，不把失败题悄悄删掉。
- 真实调用的耗时单独保存在 JSON 的 runtime 字段中，是本机调用计时，不代表生产延迟、吞吐或稳定性结论。

先保留逐题结果，再分析哪类问题获益、哪一步增加成本；没有真实题集与实际运行记录前，不在简历中写召回提升百分比。
