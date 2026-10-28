"""Score saved retrieval rankings without importing any application dependencies."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .metrics import evaluate_dataset, validate_ks


EVAL_DIR = Path(__file__).resolve().parent


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON value is not allowed: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8-sig"),
        parse_constant=reject_constant, object_pairs_hook=unique_object,
    )


def configure_console():
    # Redirected Windows output otherwise depends on the machine's code page.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value):
    return "—" if value is None else f"{value:.4f}"


def render_markdown(report):
    synthetic = report.get("synthetic", False)
    lines = ["# 检索评测报告", ""]
    if synthetic:
        lines += ["**SYNTHETIC 合成数据演示：仅验证评测代码，不是实际模型或语料的检索成绩。**", ""]
    lines += [
        f"模式：{_cell(report.get('execution_scope', 'saved-candidates / offline'))}",
        f"问题数：{report['question_count']}；每个配置必须提交同一批完整问题。", "",
        "| 配置 | 可回答题数 | 无答案题数 | MRR（完整候选列表） | 无答案题空候选率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, block in report["configs"].items():
        m = block["metrics"]
        lines.append(f"| {_cell(name)} | {m['answerable_questions']} | {m['no_answer_questions']} | {_number(m['mrr'])} | {_number(m['no_answer_empty_candidate_rate'])} |")
    lines += ["", "| 配置 | K | Hit@K | Recall@K（宏平均） |", "|---|---:|---:|---:|"]
    for name, block in report["configs"].items():
        for k in report["k_values"]:
            m = block["metrics"]
            lines.append(f"| {_cell(name)} | {k} | {_number(m[f'hit_at_{k}'])} | {_number(m[f'recall_at_{k}'])} |")
    lines += [
        "", "## 口径与限制", "",
        "- 相关性严格匹配本地文档身份和 chunk_id；0 是有效 ID，缺文档元数据不会放宽判定。",
        "- 同一证据先去重再计算 K 和排名；Recall 统计命中的不同标注片段，重复不能增加命中数。",
        "- 无答案题不进入 Hit/Recall/MRR 分母；空候选率不等于模型拒答准确率。",
        "- 没有可回答题时，相关指标为空，而不是记为满分。",
        "- 未标注候选按不相关计；不完整标注会使相关性与召回估计有偏。这里只评检索，不评答案正确性。",
        "- 保存的候选排名评测不包含模型调用耗时、问题改写、回答生成或生产负载测试。",
    ]
    if report.get("scope_notes"):
        lines += ["", *[f"- {_cell(note)}" for note in report["scope_notes"]]]
    lines += ["", "## 逐题结果", "", "| 配置 | 问题 | 可回答 | 首次命中排名 | 原始/去重候选数 | 缺身份候选数 |", "|---|---|---|---:|---:|---:|"]
    for name, block in report["configs"].items():
        for row in block["per_question"]:
            rank = row["first_relevant_rank"] if row["first_relevant_rank"] is not None else "—"
            lines.append(f"| {_cell(name)} | {_cell(row['question_id'])} | {row['answerable']} | {rank} | {row['raw_candidate_count']}/{row['unique_candidate_count']} | {row['missing_local_identity_count']} |")
    return "\n".join(lines) + "\n"


def write_reports(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = output_dir / "report.json", output_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description="纯标准库：评测保存的检索候选排名；不连接模型或 Milvus")
    parser.add_argument("--questions", default=str(EVAL_DIR / "questions.example.json"))
    parser.add_argument("--candidates", default=str(EVAL_DIR / "candidates.example.json"))
    parser.add_argument("--ks", default="1,3,5", help="逗号分隔的正整数 K")
    parser.add_argument("--output-dir", default=str(EVAL_DIR / "reports" / "demo"))
    args = parser.parse_args(argv)
    try:
        ks = validate_ks(int(value.strip()) for value in args.ks.split(","))
        questions = read_json(args.questions)
        predictions = read_json(args.candidates)
        if not isinstance(predictions, dict) or not isinstance(predictions.get("synthetic"), bool):
            raise ValueError("candidate file must contain boolean synthetic and a configs object")
        report = evaluate_dataset(questions, predictions.get("configs"), ks)
        report.update(
            synthetic=predictions["synthetic"] or any(q.get("synthetic", False) for q in questions),
            execution_scope="saved-candidates / offline; retrieval metrics only",
            generated_at=datetime.now(timezone.utc).isoformat(),
            questions_path=str(args.questions), candidates_path=str(args.candidates),
        )
        paths = write_reports(report, args.output_dir)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(render_markdown(report))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
