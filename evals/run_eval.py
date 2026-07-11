"""Run the reproducible Dense RAG evaluation and write a complete Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from evals.metrics import (
    EvaluationRecord,
    citation_coverage,
    citation_precision,
    mean_reciprocal_rank,
    recall_at_k,
    summarize,
)
from evals.schema import EvaluationCase
from ingest.index import MODEL_NAME
from rag.qa import MAX_CONTEXT_HITS, answer_from_hits
from rag.retrieve import retrieve
from rag.types import RetrievalConfig, SearchHit


CITATION_PATTERN = re.compile(r"\[(\d+)]")
RetrieveFn = Callable[[str, RetrievalConfig], list[SearchHit]]


@dataclass(frozen=True, slots=True)
class RunMetadata:
    dataset_path: str
    dataset_sha256: str
    model: str
    parameters: dict[str, Any]
    started_at: str
    finished_at: str
    duration_seconds: float


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load every non-empty JSONL row through the Task 5 schema contract."""
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("row must be a JSON object")
            case = EvaluationCase.from_dict(payload)
            if case.id in seen_ids:
                raise ValueError(f"duplicate id {case.id!r}")
            seen_ids.add(case.id)
            cases.append(case)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid dataset line {line_number}: {exc}") from exc
    return cases


def cited_parent_ids(answer: str, sources: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    """Map unique answer citation numbers to parent IDs and retain invalid numbers."""
    by_number = {int(source["n"]): str(source["parent_id"]) for source in sources}
    result: list[str] = []
    seen: set[str] = set()
    for raw_number in CITATION_PATTERN.findall(answer):
        number = int(raw_number)
        citation = by_number.get(number, f"invalid:[{number}]")
        if citation not in seen:
            seen.add(citation)
            result.append(citation)
    return tuple(result)


def build_run_metadata(
    dataset_path: Path,
    *,
    model: str,
    parameters: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
) -> RunMetadata:
    """Freeze model/configuration identity and exact dataset bytes for one run."""
    return RunMetadata(
        dataset_path=dataset_path.as_posix(),
        dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        model=model,
        parameters=dict(parameters),
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        duration_seconds=(finished_at - started_at).total_seconds(),
    )


def build_parameters(
    config: RetrievalConfig,
    *,
    generation_model: str,
    embedding_model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Describe every model and runtime parameter that identifies the baseline."""
    return {
        "generation_model": generation_model,
        "embedding_model": embedding_model,
        "dense_k": config.dense_k,
        "fused_k": config.fused_k,
        "expand_parent": config.expand_parent,
        "use_sparse": config.use_sparse,
        "use_rerank": config.use_rerank,
        "max_context_hits": MAX_CONTEXT_HITS,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def evaluate_case(
    case: EvaluationCase,
    *,
    config: RetrievalConfig,
    client: Any,
    model: str,
    temperature: float,
    max_tokens: int,
    retrieve_fn: RetrieveFn = retrieve,
) -> EvaluationRecord:
    """Run retrieval and generation once while retaining errors as failed records."""
    started = perf_counter()
    hits: list[SearchHit] = []
    total_tokens = 0
    try:
        hits = retrieve_fn(case.question, config)

        def generate_with_usage(messages: list[dict[str, str]]) -> str:
            nonlocal total_tokens
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            usage = getattr(response, "usage", None)
            value = getattr(usage, "total_tokens", None)
            if value is None:
                raise RuntimeError("model response did not include usage.total_tokens")
            total_tokens = int(value)
            return response.choices[0].message.content or ""

        result = answer_from_hits(
            case.question,
            hits,
            generator=generate_with_usage,
        )
        answer = str(result["answer"])
        sources = list(result["sources"])
        predicted_refused: bool | None = bool(result["refused"])
        predicted_citations = cited_parent_ids(answer, sources)
        error = None
    except Exception as exc:  # retain infrastructure/model failures in the report
        answer = ""
        predicted_refused = None
        predicted_citations = ()
        error = f"{type(exc).__name__}: {exc}"

    return EvaluationRecord(
        case_id=case.id,
        question=case.question,
        should_refuse=case.should_refuse,
        predicted_refused=predicted_refused,
        gold_citations=case.gold_citations,
        retrieved_citations=tuple(hit.parent_id for hit in hits),
        predicted_citations=predicted_citations,
        answer=answer,
        latency_ms=(perf_counter() - started) * 1000,
        total_tokens=total_tokens,
        error=error,
    )


def evaluate_cases(
    cases: Sequence[EvaluationCase],
    *,
    config: RetrievalConfig,
    client: Any,
    model: str,
    temperature: float,
    max_tokens: int,
    retrieve_fn: RetrieveFn = retrieve,
) -> list[EvaluationRecord]:
    """Evaluate all supplied cases in order and print non-destructive progress."""
    records: list[EvaluationRecord] = []
    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] {case.id}", flush=True)
        records.append(
            evaluate_case(
                case,
                config=config,
                client=client,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                retrieve_fn=retrieve_fn,
            )
        )
    return records


def failure_reasons(record: EvaluationRecord) -> tuple[str, ...]:
    """Classify observable baseline failures without judging unmeasured answer prose."""
    reasons: list[str] = []
    if record.error:
        reasons.append(f"运行错误：{record.error}")
    if record.predicted_refused is None or (
        record.predicted_refused != record.should_refuse
    ):
        reasons.append("拒答状态与 gold 不一致")
    if not record.should_refuse and record.gold_citations:
        recall_20 = recall_at_k(
            record.retrieved_citations, record.gold_citations, k=20
        )
        recall_5 = recall_at_k(
            record.retrieved_citations, record.gold_citations, k=5
        )
        if recall_20 < 1:
            reasons.append("检索未在 Top-20 找到全部 gold 引用")
        elif recall_5 < 1:
            reasons.append("gold 引用进入 Top-20 但未全部进入 Top-5")
        if citation_precision(
            record.predicted_citations, record.gold_citations
        ) < 1:
            reasons.append("答案包含非 gold 引用或没有有效引用")
        if citation_coverage(
            record.predicted_citations, record.gold_citations
        ) < 1:
            reasons.append("答案未覆盖全部 gold 引用")
    return tuple(reasons)


def _ratio(value: float) -> str:
    return f"{value:.4f} ({value * 100:.2f}%)"


def _md_list(values: Sequence[str]) -> str:
    return "\n".join(f"- `{value}`" for value in values) if values else "- （无）"


def _refusal_label(value: bool | None) -> str:
    if value is None:
        return "ERROR"
    return "yes" if value else "no"


def render_report(
    metadata: RunMetadata, records: Sequence[EvaluationRecord]
) -> str:
    """Render aggregate numbers plus every case, answer, retrieval, and failure."""
    summary = summarize(records)
    failed = [(record, failure_reasons(record)) for record in records]
    failed = [(record, reasons) for record, reasons in failed if reasons]
    lines = [
        "# Dense RAG Baseline（Task 7）",
        "",
        "> 本报告由统一评估 CLI 生成。30 条样例按数据集原顺序全部保留；失败不会被筛除。",
        "",
        "## 运行元数据",
        "",
        "| 字段 | 值 |",
        "|---|---|",
        f"| 生成模型 | `{metadata.model}` |",
        f"| Embedding 模型 | `{metadata.parameters.get('embedding_model', '未记录')}` |",
        f"| 参数 | `{json.dumps(metadata.parameters, ensure_ascii=False, sort_keys=True)}` |",
        f"| 数据集 | `{metadata.dataset_path}` |",
        f"| 数据集 SHA-256 | `{metadata.dataset_sha256}` |",
        f"| 开始时间 | `{metadata.started_at}` |",
        f"| 结束时间 | `{metadata.finished_at}` |",
        f"| 总运行时间 | {metadata.duration_seconds:.3f} s |",
        f"| 样例数 | {summary.total_cases} |",
        "",
        "## 指标口径",
        "",
        "- Recall@K：每道非拒答题的 gold 父章节中，进入检索 Top-K 的比例，再对题目取平均。",
        "- MRR：每道非拒答题第一个 gold 父章节排名的倒数，未命中为 0，再取平均。",
        "- 引用准确率：答案实际使用的唯一引用中，属于 gold 的比例。无引用为 0。",
        "- 引用覆盖率：gold 引用中，被答案实际使用的比例。",
        "- 拒答准确率：30 道题的 `refused` 状态与 `should_refuse` 完全一致的比例；运行错误计错。",
        "- P50/P95：端到端（检索 + 生成）的最近秩延迟百分位；首次模型加载计入真实延迟。",
        "- 平均 Token：模型返回的 `usage.total_tokens` 按全部题目平均；未调用或失败调用记 0。",
        "",
        "## 汇总结果",
        "",
        "| 指标 | 结果 | 分母/单位 |",
        "|---|---:|---|",
        f"| Recall@5 | {_ratio(summary.recall_at_5)} | {summary.answer_cases} 道非拒答题 |",
        f"| Recall@20 | {_ratio(summary.recall_at_20)} | {summary.answer_cases} 道非拒答题 |",
        f"| MRR | {summary.mrr:.4f} | {summary.answer_cases} 道非拒答题 |",
        f"| Citation precision | {_ratio(summary.citation_precision)} | {summary.answer_cases} 道非拒答题 |",
        f"| Citation coverage | {_ratio(summary.citation_coverage)} | {summary.answer_cases} 道非拒答题 |",
        f"| Refusal accuracy | {_ratio(summary.refusal_accuracy)} | {summary.total_cases} 道题 |",
        f"| P50 latency | {summary.p50_latency_ms:.2f} ms | 端到端 |",
        f"| P95 latency | {summary.p95_latency_ms:.2f} ms | 端到端 |",
        f"| Average tokens | {summary.average_tokens:.2f} | total tokens/题 |",
        "",
        "## 全部样例结果",
        "",
        "| ID | R@5 | R@20 | RR | Cit.P | Cit.C | Gold拒答 | 预测拒答 | 延迟ms | Token | 失败项 |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for record in records:
        if record.should_refuse or not record.gold_citations:
            r5 = r20 = rr = cp = cc = "—"
        else:
            r5 = f"{recall_at_k(record.retrieved_citations, record.gold_citations, k=5):.2f}"
            r20 = f"{recall_at_k(record.retrieved_citations, record.gold_citations, k=20):.2f}"
            rr = f"{mean_reciprocal_rank(record.retrieved_citations, record.gold_citations):.3f}"
            cp = f"{citation_precision(record.predicted_citations, record.gold_citations):.2f}"
            cc = f"{citation_coverage(record.predicted_citations, record.gold_citations):.2f}"
        lines.append(
            f"| {record.case_id} | {r5} | {r20} | {rr} | {cp} | {cc} | "
            f"{_refusal_label(record.should_refuse)} | {_refusal_label(record.predicted_refused)} | "
            f"{record.latency_ms:.2f} | {record.total_tokens} | {len(failure_reasons(record))} |"
        )

    lines.extend(["", "## 完整逐题记录", ""])
    for record in records:
        reasons = failure_reasons(record)
        lines.extend(
            [
                f"### {record.case_id}",
                "",
                f"- 问题：{record.question}",
                f"- Gold 拒答：{_refusal_label(record.should_refuse)}",
                f"- 预测拒答：{_refusal_label(record.predicted_refused)}",
                f"- 延迟：{record.latency_ms:.2f} ms",
                f"- Total tokens：{record.total_tokens}",
                f"- 结论：{'失败：' + '；'.join(reasons) if reasons else '本报告所测指标均通过'}",
                "",
                "Gold 引用：",
                "",
                _md_list(record.gold_citations),
                "",
                "检索排名（最多 20 条）：",
                "",
                _md_list(record.retrieved_citations[:20]),
                "",
                "答案实际引用：",
                "",
                _md_list(record.predicted_citations),
                "",
                "模型答案：",
                "",
                record.answer or "（空）",
            ]
        )
        if record.error:
            lines.extend(["", f"运行错误：`{record.error}`"])
        lines.append("")

    lines.extend(["## 失败案例分析", ""])
    if not failed:
        lines.append("本次运行没有发现由既定指标定义的失败。")
    else:
        for record, reasons in failed:
            lines.extend(
                [
                    f"### {record.case_id}",
                    "",
                    *[f"- {reason}" for reason in reasons],
                    "",
                ]
            )
    lines.extend(
        [
            "## 解释边界",
            "",
            "这些数字只测检索、引用、拒答、延迟和 Token，不直接证明答案文字覆盖了所有 `gold_points`，也不构成法律意见。引用命中 gold 只说明定位相同，仍需人工核对答案是否忠实表达原文。",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evals/dataset.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline.md"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--fused-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--no-expand-parent", action="store_true", help="evaluate child hits directly"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    load_dotenv()
    args = build_parser().parse_args(argv)
    model = args.model or os.environ.get("LLM_MODEL")
    api_key = os.environ.get("LLM_API_KEY")
    if not model or not api_key:
        print("LLM_MODEL and LLM_API_KEY must be configured", file=sys.stderr)
        return 2
    if args.dense_k <= 0 or args.fused_k <= 0 or args.max_tokens <= 0:
        print("dense-k, fused-k, and max-tokens must be positive", file=sys.stderr)
        return 2

    from openai import OpenAI

    cases = load_cases(args.dataset)
    config = RetrievalConfig(
        dense_k=args.dense_k,
        fused_k=args.fused_k,
        use_sparse=False,
        use_rerank=False,
        expand_parent=not args.no_expand_parent,
    )
    parameters = build_parameters(
        config,
        generation_model=model,
        embedding_model=MODEL_NAME,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL") or None,
    )
    started_at = datetime.now(timezone.utc)
    records = evaluate_cases(
        cases,
        config=config,
        client=client,
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    finished_at = datetime.now(timezone.utc)
    metadata = build_run_metadata(
        args.dataset,
        model=model,
        parameters=parameters,
        started_at=started_at,
        finished_at=finished_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(metadata, records), encoding="utf-8")
    summary = summarize(records)
    print(f"wrote {args.output} cases={summary.total_cases}")
    print(
        f"Recall@5={summary.recall_at_5:.4f} "
        f"Recall@20={summary.recall_at_20:.4f} MRR={summary.mrr:.4f}"
    )
    return 1 if any(record.error for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
