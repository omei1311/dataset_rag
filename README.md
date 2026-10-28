# 图文文档 RAG 问答系统

基于 Python、FastAPI、LangGraph 和 Milvus 的图文文档检索问答项目。将 PDF 解析为 Markdown，结合图片附近上下文生成图片摘要，再通过混合检索、HyDE、RRF 和重排序组织回答依据。

当前实体识别与示例主要面向产品说明书。通用 PDF 处理流程可以作为论文问答的基础，但本仓库没有已验证的学术论文端到端效果或生产用户规模，不将示例结果当成这些能力的证明。

## 已实现的链路

```text
PDF → MinerU 解析 → 图片及上下文摘要 → 分层切分 → BGE-M3 → Milvus
问题 → 主体确认 / 问题重写 → 混合检索 ─┐
                         → HyDE 检索 ─┼→ RRF → BGE 重排 → 有编号证据的回答
                         → Web 搜索 ─┘
```

- 图片保存在 MinIO，会话历史保存在 MongoDB；运行节点通过 SSE 返回进度。
- RRF 按来源文档与 chunk 组合去重，保留合法的 chunk ID `0`；同一路重复结果不会重复计票，多路召回的共同证据仍可累加分数。
- 重排与降级不修改输入候选。模型加载失败、分数数量不匹配或出现非有限数时，保留本地 RRF、随后 Web 的原顺序，并标记 `rerank_degraded`。
- 两条本地检索路线均保留 `file_title`，融合与重排后继续携带 `document_id`、`chunk_id`、章节、父章节、来源等信息；原始数据没有页码时不会编造页码。
- 回答使用 `[1]` 等证据编号。同步结果和 SSE `final` 事件提供 `sources`，且只包含实际进入字符预算内上下文的证据；截断片段标记 `truncated`。空证据直接说明依据不足，不调用模型生成答案。
- 图片输出同样只来自实际保留的上下文或来源链接，截断尾部和未进入上下文的文档不会额外返回图片。
- 同一会话正在查询时拒绝重入（HTTP 409）；每轮清理旧的内存结果，失败时清除部分回答和证据，同步调用返回 HTTP 500。MongoDB 对话历史不随本轮清理而删除。

`sources` 是送入模型的证据清单，**不等于已验证的引用正确率、事实准确率或模型置信度**。最终答案仍需用实际题集检查。

## 无外部服务的验证

Python 3.11+ 即可运行以下命令，不需要安装运行时依赖、配置密钥、下载模型或启动数据库：

```bash
python -S -m unittest discover -s offline_tests -v
python -m eval.offline_eval --help
python -m eval.offline_eval --questions eval/questions.example.json --candidates eval/candidates.example.json --output-dir eval/reports/demo
```

离线回归覆盖跨文档相同 ID、ID 0、RRF 重复计票、候选元数据、输入不可变性、重排失败及非法分数、上下文截断、图片证据边界、无证据回答、SSE 最终证据输出、同会话重入与失败后的旧结果隔离等边界。外部服务在测试中使用替身；这些测试不能证明真实模型效果或在线服务稳定性。

评测示例是**人工构造的合成 smoke fixture**，只用于验证指标计算和报告输出，不能将其分数写成真实 RAG 准确率或性能提升。

## 完整服务运行

以下命令从仓库根目录运行。完整运行时需要额外依赖、模型与外部服务；`uv.lock` 对应现有依赖环境，首次安装与模型准备可能耗时。

1. 安装 Python 3.11+ 和 uv，运行 `uv sync --frozen`，激活 `.venv`（或在命令前使用 `uv run`）。
2. 复制配置模板：PowerShell 使用 `Copy-Item .env.example .env`，Linux/macOS 使用 `cp .env.example .env`。
3. 将 `.env` 中所有占位符替换为自己的服务配置。准备 Milvus、MongoDB、MinIO、文本/视觉模型 API、MinerU 解析服务及联网搜索 MCP；准备 BGE-M3 与 BGE 重排模型的本地目录。仅修改模型名并不能保证所有 API 提供商参数兼容。
4. 在两个终端分别启动导入服务与查询服务：

```bash
python -m uvicorn app.import_process.api.import_process:app --host 127.0.0.1 --port 8000
python -m uvicorn app.query_process.api.query_server:app --host 127.0.0.1 --port 8001
```

先打开 `http://127.0.0.1:8000/import` 上传并导入文档，再访问 `http://127.0.0.1:8001/chat.html`。两个服务分别在 `/docs` 提供接口文档。

| 服务 | 方法与路径 | 作用 |
|---|---|---|
| 导入 | `POST /upload` | 上传文档并启动后台任务 |
| 导入 | `GET /status/{task_id}` | 查询导入节点和任务状态 |
| 查询 | `POST /query` | 同步回答，或启动 SSE 查询 |
| 查询 | `GET /stream/{session_id}` | SSE 进度、增量回答和最终证据 |
| 查询 | `GET /history/{session_id}` | 查询会话历史 |
| 查询 | `DELETE /history/{session_id}` | 删除会话历史 |

同步请求示例：

```bash
curl -X POST http://127.0.0.1:8001/query -H "Content-Type: application/json" -d '{"query":"说明书A中的操作步骤是什么？","is_stream":false}'
```

响应保留原有 `answer`、`session_id` 等字段，并增加：

```json
{
  "sources": [
    {
      "citation": 1,
      "source": "local",
      "document_id": "实际来源文件名",
      "file_title": "实际来源文件名",
      "chunk_id": 0,
      "title": "实际章节",
      "parent_title": "实际父章节",
      "url": "",
      "truncated": false
    }
  ],
  "rerank_degraded": false
}
```

这是字段格式示意，不是真实文档、真实请求结果或质量评测。

## 评测口径

题集/候选格式、指标定义与 live 命令详见 [评测说明](eval/README.md)。

- 离线评测对明确标注的来源文档与 chunk 做严格匹配，分别报告 Hit@K、MRR、Recall@K；重复候选不重复贡献分数，缺少必要来源元数据不能放宽命中条件。
- 无答案问题单独统计，不能将空检索误算为普通问题的 Recall。仅凭检索是否为空，不能证明最终回答的拒答质量。
- 可选 live 检索评测需要运行时依赖和外部服务。旧式题集中的 `item_names` 是预先给定的检索范围，问题重写被跳过，因此属于 **oracle 范围内的检索评测**，不代表主体识别、多轮对话或端到端回答效果。
- 标准 live 配置为基础混合检索、加 HyDE/RRF、加重排、再加 Web。只有使用固定真实语料、人工核验相关证据和一致候选判定后，才能比较不同配置的效果、降级比例及耗时。
- 本仓库没有声称已经跑出真实模型准确率、真实语料提升比例或上线收益。生成的实际报告默认保存在 `eval/reports/`，不会自动提交到 Git。

建议实际题集覆盖单文档定位、跨文档比较、图表信息和无答案问题，并在调参集之外保留独立测试集。页码标注只有原始解析数据提供时才可使用；目前优先用文件、章节和 chunk ID 回溯。

## 当前运行边界

- API、演示页面和上传逻辑沿用本地开发方案，没有完整用户鉴权、租户隔离或生产任务队列；内存任务和 SSE 队列也不适合多 worker 状态共享。以上启动命令只监听本机。
- 现有 MinIO 客户端会配置匿名只读桶策略供图片展示。请仅使用专用测试桶和可公开的样例内容；真实业务部署应先完成访问控制、上传文件校验、私有对象访问及服务隔离。
- `.env`、模型、本机日志、导出文件和服务卷不应进入新的提交。`.gitignore` 防止新增误提交，不会移除仓库历史中已经跟踪的产物或凭据；曾公开的凭据应在服务端轮换。
- 离线测试使用真实的纯转换代码与受控替身，未替代 MinerU、Milvus、MongoDB、MinIO、模型 API 和 MCP 的实际联调。

## 代码位置

| 路径 | 内容 |
|---|---|
| `app/import_process/` | PDF 解析、图文处理、切分、向量入库 |
| `app/query_process/` | 主体确认、检索、融合、重排和回答 |
| `app/query_process/agent/nodes/retrieval_common.py` | 无依赖的证据身份、融合与上下文构建 |
| `eval/` | 离线指标、合成示例与可选 live 检索评测 |
| `offline_tests/` | 不连服务、不加载模型的回归验证 |
| `prompts/` | 实体提取、图片摘要、HyDE 和回答提示词 |
