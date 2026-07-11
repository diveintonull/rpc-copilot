"""Run the Task10 retrieval-only ablation matrix on isolated Qdrant indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Literal

from ingest.chunk import iter_spans
from ingest.chunk_parent import PARSED
from evals.schema import EvaluationCase
from rag.types import SearchHit


Chunking = Literal["fixed_window", "parent_child"]
FIXED_COLLECTION = "grc_kb_ablation_fixed"
PARENT_CHILD_COLLECTION = "grc_kb_ablation_parent_child"


@dataclass(frozen=True, slots=True)
class AblationConfig:
    chunking: Chunking
    use_sparse: bool
    use_rerank: bool
    dense_k: int = 20
    sparse_k: int = 20
    fused_k: int = 20
    rerank_k: int = 5
    window_size: int = 500
    window_overlap: int = 100
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if self.chunking not in {"fixed_window", "parent_child"}:
            raise ValueError(f"unsupported chunking strategy: {self.chunking}")
        if min(self.dense_k, self.sparse_k, self.fused_k, self.rerank_k) <= 0:
            raise ValueError("retrieval K values must be positive")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if self.window_size <= 0 or not 0 <= self.window_overlap < self.window_size:
            raise ValueError("window overlap must satisfy 0 <= overlap < size")


@dataclass(frozen=True, slots=True)
class SectionSpan:
    parent_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("section span must be non-empty and non-negative")


@dataclass(frozen=True, slots=True)
class CorpusItem:
    item_id: str
    text: str
    parent_ids: tuple[str, ...]
    source_id: str
    version: str
    section_number: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    gold_citations: tuple[str, ...]
    hit_parent_ids: tuple[tuple[str, ...], ...]
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ConfigResult:
    config: AblationConfig
    config_sha256: str
    index_size: int
    recall_at_5: float
    mrr: float
    p95_latency_ms: float
    answer_cases: int
    total_cases: int
    error_count: int
    cases: tuple[CaseResult, ...]


def generate_config_matrix(
    *, window_size: int = 500, window_overlap: int = 100
) -> tuple[AblationConfig, ...]:
    """Return the stable 2 x 2 x 2 matrix exactly once."""
    return tuple(
        AblationConfig(
            chunking,
            use_sparse,
            use_rerank,
            window_size=window_size,
            window_overlap=window_overlap,
        )
        for chunking, use_sparse, use_rerank in product(
            ("fixed_window", "parent_child"), (False, True), (False, True)
        )
    )


def config_hash(config: AblationConfig) -> str:
    """Hash canonical configuration JSON, including every retrieval parameter."""
    payload = json.dumps(
        asdict(config), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_cases_from_bytes(
    frozen: bytes,
) -> tuple[list[EvaluationCase], str]:
    """Parse cases and compute their hash from one immutable byte snapshot."""
    digest = hashlib.sha256(frozen).hexdigest()
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        frozen.decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("row must be a JSON object")
            case = EvaluationCase.from_dict(payload)
            if case.id in seen:
                raise ValueError(f"duplicate id {case.id!r}")
            seen.add(case.id)
            cases.append(case)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid dataset line {line_number}: {exc}") from exc
    return cases, digest


def fixed_items_from_text(
    text: str,
    *,
    doc_id: str,
    sections: Sequence[SectionSpan],
    size: int,
    overlap: int,
    source_id: str,
    version: str,
) -> list[CorpusItem]:
    """Split a document into fixed windows and retain every overlapped section."""
    items: list[CorpusItem] = []
    for index, (start, end) in enumerate(iter_spans(len(text), size, overlap)):
        parent_ids = tuple(
            section.parent_id
            for section in sections
            if start < section.end and end > section.start
        )
        section_number = ""
        if len(parent_ids) == 1 and "#" in parent_ids[0]:
            section_number = parent_ids[0].rsplit("#", 1)[1]
        items.append(
            CorpusItem(
                item_id=f"{doc_id}:fixed:{index}",
                text=text[start:end],
                parent_ids=parent_ids,
                source_id=source_id,
                version=version,
                section_number=section_number,
            )
        )
    return items


def locate_section_spans(text: str, sections: Sequence[object]) -> tuple[SectionSpan, ...]:
    """Locate already-segmented section text sequentially in its source document."""
    cursor = 0
    spans: list[SectionSpan] = []
    for section in sections:
        section_text = str(getattr(section, "text"))
        start = text.find(section_text, cursor)
        if start < 0:
            raise ValueError(f"cannot locate section {getattr(section, 'id')} in document")
        end = start + len(section_text)
        spans.append(SectionSpan(str(getattr(section, "id")), start, end))
        cursor = end
    return tuple(spans)


def parent_child_items_from_parts(
    parents: Sequence[object], children: Sequence[object]
) -> tuple[list[CorpusItem], dict[str, CorpusItem]]:
    """Normalize the production parent/child corpus for the ablation runtime."""
    parent_items = {
        str(parent.id): CorpusItem(
            item_id=str(parent.id),
            text=str(parent.text),
            parent_ids=(str(parent.id),),
            source_id=str(parent.metadata.get("source_id", "")),
            version=str(parent.metadata.get("version", "")),
            section_number=str(parent.number),
        )
        for parent in parents
    }
    items = [
        CorpusItem(
            item_id=str(child.id),
            text=str(child.text),
            parent_ids=(str(child.metadata["parent_id"]),),
            source_id=str(child.metadata.get("source_id", child.metadata.get("source", ""))),
            version=str(child.metadata.get("version", "")),
            section_number=str(child.metadata.get("section_number", "")),
        )
        for child in children
    ]
    return items, parent_items


def build_parent_child_corpus() -> tuple[list[CorpusItem], dict[str, CorpusItem]]:
    from ingest.index import build_corpus

    parents, children = build_corpus()
    return parent_child_items_from_parts(parents, children)


def build_fixed_corpus(
    *, size: int = 500, overlap: int = 100, parsed_dir: Path = PARSED
) -> list[CorpusItem]:
    """Build true document-level windows and map them to overlapped sections."""
    from ingest.index import build_corpus, document_meta_for

    parents, _children = build_corpus()
    items: list[CorpusItem] = []
    for path in sorted(parsed_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        document = document_meta_for(path, text)
        document_parents = [
            parent
            for parent in parents
            if parent.id.startswith(f"{document.document_id}#")
        ]
        spans = locate_section_spans(text, document_parents)
        items.extend(
            fixed_items_from_text(
                text,
                doc_id=document.document_id,
                sections=spans,
                size=size,
                overlap=overlap,
                source_id=document.source_id,
                version=document.version,
            )
        )
    return items


def recall_at_k_for_hits(
    hits: Sequence[CorpusItem], gold_citations: Sequence[str], *, k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    gold = set(gold_citations)
    if not gold:
        return 0.0
    covered = {
        parent_id for hit in hits[:k] for parent_id in hit.parent_ids
    }
    return len(covered & gold) / len(gold)


def reciprocal_rank_for_hits(
    hits: Sequence[CorpusItem], gold_citations: Sequence[str]
) -> float:
    gold = set(gold_citations)
    for rank, hit in enumerate(hits, start=1):
        if gold.intersection(hit.parent_ids):
            return 1.0 / rank
    return 0.0


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def aggregate_config(
    config: AblationConfig,
    *,
    index_size: int,
    cases: Sequence[CaseResult],
) -> ConfigResult:
    answer_cases = [case for case in cases if case.gold_citations]
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for case in answer_cases:
        hits = [
            CorpusItem(str(index), "", parent_ids, "", "", "")
            for index, parent_ids in enumerate(case.hit_parent_ids)
        ]
        recalls.append(recall_at_k_for_hits(hits, case.gold_citations, k=5))
        reciprocal_ranks.append(reciprocal_rank_for_hits(hits, case.gold_citations))
    return ConfigResult(
        config=config,
        config_sha256=config_hash(config),
        index_size=index_size,
        recall_at_5=sum(recalls) / len(recalls) if recalls else 0.0,
        mrr=(
            sum(reciprocal_ranks) / len(reciprocal_ranks)
            if reciprocal_ranks
            else 0.0
        ),
        p95_latency_ms=_nearest_rank_percentile(
            [case.latency_ms for case in cases], 0.95
        ),
        answer_cases=len(answer_cases),
        total_cases=len(cases),
        error_count=sum(case.error is not None for case in cases),
        cases=tuple(cases),
    )


def build_ablation_index(
    items: Sequence[CorpusItem],
    *,
    collection_name: str,
    client,
    vectors: Sequence[Sequence[float]],
) -> int:
    """Replace one dedicated experiment collection and verify its final count."""
    if collection_name not in {FIXED_COLLECTION, PARENT_CHILD_COLLECTION}:
        raise ValueError("ablation index must use a dedicated collection")
    if len(items) != len(vectors):
        raise ValueError("item and vector counts must match")
    if len(items) == 0 or len(vectors) == 0 or len(vectors[0]) == 0:
        raise ValueError("ablation index cannot be empty")

    from qdrant_client import models

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name,
        vectors_config=models.VectorParams(
            size=len(vectors[0]), distance=models.Distance.COSINE
        ),
    )
    points = [
        models.PointStruct(
            id=index,
            vector=list(vectors[index]),
            payload={
                "item_id": item.item_id,
                "text": item.text,
                "parent_ids": list(item.parent_ids),
                "source_id": item.source_id,
                "version": item.version,
                "section_number": item.section_number,
            },
        )
        for index, item in enumerate(items)
    ]
    for start in range(0, len(points), 256):
        client.upsert(collection_name, points=points[start : start + 256])
    count = int(client.count(collection_name).count)
    if count != len(items):
        raise RuntimeError(
            f"Qdrant count mismatch for {collection_name}: {count} != {len(items)}"
        )
    return count


@dataclass(slots=True)
class AblationRuntime:
    """Real retrieval dependencies shared by all eight configurations."""

    client: object
    model: object
    corpora: dict[Chunking, list[CorpusItem]]
    parent_items: dict[str, CorpusItem]
    _item_maps: dict[Chunking, dict[str, CorpusItem]] = field(init=False)
    _sparse_indexes: dict[Chunking, object] = field(init=False)

    def __post_init__(self) -> None:
        from rag.sparse import SparseIndex

        self._item_maps = {
            strategy: {item.item_id: item for item in items}
            for strategy, items in self.corpora.items()
        }
        self._sparse_indexes = {
            strategy: SparseIndex([self._as_search_hit(item) for item in items])
            for strategy, items in self.corpora.items()
        }

    @staticmethod
    def _as_search_hit(item: CorpusItem) -> SearchHit:
        return SearchHit(
            chunk_id=item.item_id,
            parent_id=item.item_id,
            score=0.0,
            text=item.text,
            source_id=item.source_id,
            version=item.version,
            section_number=item.section_number,
        )

    @staticmethod
    def collection_for(chunking: Chunking) -> str:
        return FIXED_COLLECTION if chunking == "fixed_window" else PARENT_CHILD_COLLECTION

    def dense_search(self, query: str, config: AblationConfig) -> list[SearchHit]:
        from ingest.index import embed

        vector = embed(self.model, [query])[0]
        points = self.client.query_points(
            self.collection_for(config.chunking),
            query=vector.tolist(),
            limit=config.dense_k,
            with_payload=True,
        ).points
        return [
            SearchHit(
                chunk_id=str(point.payload["item_id"]),
                parent_id=str(point.payload["item_id"]),
                score=float(point.score),
                text=str(point.payload["text"]),
                source_id=str(point.payload.get("source_id", "")),
                version=str(point.payload.get("version", "")),
                section_number=str(point.payload.get("section_number", "")),
            )
            for point in points
        ]

    def sparse_search(self, query: str, config: AblationConfig) -> list[SearchHit]:
        return self._sparse_indexes[config.chunking].search(query, k=config.sparse_k)

    @staticmethod
    def fuse(
        dense_hits: list[SearchHit],
        sparse_hits: list[SearchHit],
        config: AblationConfig,
    ) -> list[SearchHit]:
        from rag.fusion import reciprocal_rank_fusion

        return reciprocal_rank_fusion(
            dense_hits,
            sparse_hits,
            rrf_k=config.rrf_k,
            limit=config.fused_k,
        )

    def materialize(
        self, hits: Sequence[SearchHit], chunking: Chunking
    ) -> list[CorpusItem]:
        item_map = self._item_maps[chunking]
        if chunking == "fixed_window":
            return [item_map[hit.chunk_id] for hit in hits if hit.chunk_id in item_map]

        expanded: list[CorpusItem] = []
        seen: set[str] = set()
        for hit in hits:
            child = item_map.get(hit.chunk_id)
            if child is None or not child.parent_ids:
                continue
            parent_id = child.parent_ids[0]
            if parent_id in seen or parent_id not in self.parent_items:
                continue
            seen.add(parent_id)
            parent = self.parent_items[parent_id]
            expanded.append(
                CorpusItem(
                    item_id=hit.chunk_id,
                    text=parent.text,
                    parent_ids=parent.parent_ids,
                    source_id=parent.source_id,
                    version=parent.version,
                    section_number=parent.section_number,
                )
            )
        return expanded

    @staticmethod
    def rerank(query: str, items: Sequence[CorpusItem], limit: int) -> list[CorpusItem]:
        from rag.rerank import rerank_hits

        by_id = {item.item_id: item for item in items}
        hits = [AblationRuntime._as_search_hit(item) for item in items]
        reranked = rerank_hits(query, hits, limit=limit, strict=True)
        return [by_id[hit.chunk_id] for hit in reranked if hit.chunk_id in by_id]


def retrieve_for_ablation(
    query: str, config: AblationConfig, runtime: AblationRuntime
) -> list[CorpusItem]:
    """Run one configured retrieval path and normalize output to coverage items."""
    dense_hits = runtime.dense_search(query, config)
    if config.use_sparse:
        sparse_hits = runtime.sparse_search(query, config)
        candidates = runtime.fuse(dense_hits, sparse_hits, config)
    else:
        candidates = dense_hits[: config.fused_k]
    items = runtime.materialize(candidates, config.chunking)
    if config.use_rerank:
        return runtime.rerank(query, items, config.rerank_k)
    return items[: config.fused_k]


def evaluate_config(
    cases: Sequence[object],
    *,
    config: AblationConfig,
    index_size: int,
    runtime: AblationRuntime,
    retrieve_fn: Callable[[str, AblationConfig, AblationRuntime], list[CorpusItem]] = retrieve_for_ablation,
) -> ConfigResult:
    """Evaluate one configuration while retaining each query failure."""
    records: list[CaseResult] = []
    for position, case in enumerate(cases, start=1):
        started = perf_counter()
        try:
            hits = retrieve_fn(str(case.question), config, runtime)
            hit_parent_ids = tuple(hit.parent_ids for hit in hits)
            error = None
        except Exception as exc:
            hit_parent_ids = ()
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            CaseResult(
                case_id=str(case.id),
                gold_citations=tuple(case.gold_citations),
                hit_parent_ids=hit_parent_ids,
                latency_ms=(perf_counter() - started) * 1000,
                error=error,
            )
        )
        print(
            f"[{config_hash(config)[:8]} {position}/{len(cases)}] {case.id}",
            flush=True,
        )
    return aggregate_config(config, index_size=index_size, cases=records)


def run_matrix(
    configs: Iterable[AblationConfig],
    *,
    evaluate: Callable[[AblationConfig], ConfigResult],
) -> tuple[ConfigResult, ...]:
    """Execute each configuration once and reject duplicate identities."""
    results: list[ConfigResult] = []
    seen: set[str] = set()
    for config in configs:
        identity = config_hash(config)
        if identity in seen:
            raise ValueError(f"duplicate configuration: {identity}")
        seen.add(identity)
        results.append(evaluate(config))
    return tuple(results)


def select_default(results: Sequence[ConfigResult]) -> ConfigResult:
    """Choose quality first, using latency only when retrieval quality ties."""
    if not results:
        raise ValueError("cannot select a default from empty results")
    return max(
        results,
        key=lambda result: (
            result.recall_at_5,
            result.mrr,
            -result.p95_latency_ms,
        ),
    )


def _find_reference(
    result: ConfigResult,
    results: Sequence[ConfigResult],
    baseline: ConfigResult,
) -> tuple[ConfigResult, str] | None:
    if result is baseline:
        return None
    config = result.config
    if config.use_sparse:
        wanted = replace(config, use_sparse=False)
        label = "同切片 Dense"
    elif config.use_rerank:
        wanted = replace(config, use_rerank=False)
        label = "同切片无 Rerank"
    elif config.chunking == "parent_child":
        wanted = replace(config, chunking="fixed_window")
        label = "固定窗口同配置"
    else:
        return baseline, "固定窗口 Dense 基准"
    reference = next((candidate for candidate in results if candidate.config == wanted), baseline)
    return reference, label


def _conclusion(
    result: ConfigResult,
    results: Sequence[ConfigResult],
    baseline: ConfigResult,
) -> str:
    reference_info = _find_reference(result, results, baseline)
    if reference_info is None:
        return "基准：固定窗口、Dense、无 Rerank。"
    reference, reference_label = reference_info
    recall_delta = result.recall_at_5 - reference.recall_at_5
    mrr_delta = result.mrr - reference.mrr
    latency_delta = result.p95_latency_ms - reference.p95_latency_ms
    if recall_delta > 1e-12:
        label = "改善"
    elif recall_delta < -1e-12:
        label = "退化"
    elif mrr_delta > 1e-12:
        label = "改善（Recall 不变，MRR 上升）"
    elif mrr_delta < -1e-12:
        label = "退化（Recall 不变，MRR 下降）"
    else:
        label = "质量无变化"
    return (
        f"相对{reference_label} {label}：Recall@5 {recall_delta:+.4f}，"
        f"MRR {mrr_delta:+.4f}，P95 {latency_delta:+.2f} ms。"
    )


def render_report(
    results: Sequence[ConfigResult],
    *,
    dataset_path: str,
    dataset_sha256: str,
    started_at: str,
    finished_at: str,
    device: str,
    embedding_model: str = "BAAI/bge-m3",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    collection_sizes: dict[str, int] | None = None,
) -> str:
    if not results:
        raise ValueError("report needs at least one result")
    baseline = next(
        (
            result
            for result in results
            if result.config.chunking == "fixed_window"
            and not result.config.use_sparse
            and not result.config.use_rerank
        ),
        results[0],
    )
    default = select_default(results)
    lines = [
        "# Task 10：检索消融矩阵",
        "",
        "> 八种组合全部保留；失败和退化不会被删除。",
        "",
        "## 运行元数据",
        "",
        f"- 数据集：`{dataset_path}`",
        f"- 数据集 SHA-256：`{dataset_sha256}`",
        f"- 设备：`{device}`",
        f"- Embedding 模型：`{embedding_model}`",
        f"- Reranker：`{reranker_model}`",
        f"- Collections：`{json.dumps(collection_sizes or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- 开始时间：`{started_at}`",
        f"- 结束时间：`{finished_at}`",
        "",
        "## 完整矩阵",
        "",
        "| 切片 | Sparse | Rerank | 配置 hash | Recall@5 | MRR | P95 ms | 索引数 | 错误 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| `{result.config.chunking}` | {result.config.use_sparse} | "
            f"{result.config.use_rerank} | `{result.config_sha256}` | "
            f"{result.recall_at_5:.4f} | {result.mrr:.4f} | "
            f"{result.p95_latency_ms:.2f} | {result.index_size} | "
            f"{result.error_count} |"
        )
    lines.extend(["", "## 逐行结论", ""])
    for result in results:
        lines.append(
            f"- `{result.config_sha256}` 逐行结论："
            f"{_conclusion(result, results, baseline)}"
        )
    lines.append("")
    lines.extend(["## 配置详情", ""])
    for result in results:
        lines.append(
            f"- `{result.config_sha256}`：`"
            f"{json.dumps(asdict(result.config), ensure_ascii=False, sort_keys=True)}`"
        )
    lines.append("")
    lines.extend(
        [
            "## 默认配置",
            "",
            f"选择 `{default.config_sha256}`：Recall@5={default.recall_at_5:.4f}，"
            f"MRR={default.mrr:.4f}，P95={default.p95_latency_ms:.2f} ms。",
            "选择顺序固定为 Recall@5、MRR、P95；模型名称不参与优先级。",
            "",
            "## 错误与失败记录",
            "",
        ]
    )
    failures = [
        (result, case)
        for result in results
        for case in result.cases
        if case.error
    ]
    if not failures:
        lines.append("本次运行没有基础设施错误；检索质量退化仍保留在矩阵中。")
    else:
        for result, case in failures:
            lines.append(
                f"- `{result.config_sha256}` / `{case.case_id}`：{case.error}"
            )
    lines.extend(
        [
            "",
            "## 完整逐题记录",
            "",
            "| 配置 hash | ID | R@5 | RR | Gold 父章节 | 返回结果覆盖的父章节 | 延迟 ms | 错误 |",
            "|---|---|---:|---:|---|---|---:|---|",
        ]
    )
    for result in results:
        for case in result.cases:
            hits = [
                CorpusItem(str(index), "", parent_ids, "", "", "")
                for index, parent_ids in enumerate(case.hit_parent_ids)
            ]
            if case.gold_citations:
                recall = f"{recall_at_k_for_hits(hits, case.gold_citations, k=5):.4f}"
                reciprocal_rank = f"{reciprocal_rank_for_hits(hits, case.gold_citations):.4f}"
            else:
                recall = reciprocal_rank = "—"
            gold = json.dumps(case.gold_citations, ensure_ascii=False)
            returned = json.dumps(case.hit_parent_ids, ensure_ascii=False)
            error = (case.error or "").replace("|", "\\|")
            lines.append(
                f"| `{result.config_sha256}` | `{case.case_id}` | {recall} | "
                f"{reciprocal_rank} | `{gold}` | `{returned}` | "
                f"{case.latency_ms:.2f} | {error or '—'} |"
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "本报告只比较检索覆盖、首个正确结果位置、检索延迟和索引规模；不调用生成模型，也不证明最终答案文字正确。",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evals/dataset.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/retrieval_ablation.md")
    )
    parser.set_defaults(window_size=500, window_overlap=100)
    return parser


def _device_label() -> str:
    import torch

    if not torch.cuda.is_available():
        return f"cpu; torch={torch.__version__}"
    return (
        f"cuda:0 {torch.cuda.get_device_name(0)}; torch={torch.__version__}; "
        f"cuda={torch.version.cuda}; dtype=float16"
    )


def _execute_experiment(args: argparse.Namespace) -> tuple[str, int]:
    """Build both real indexes, run all eight rows, and return report + error count."""
    from ingest.index import _client, embed, get_model
    from rag.rerank import RERANK_MODEL_NAME
    from ingest.index import MODEL_NAME

    if args.window_size <= 0 or not 0 <= args.window_overlap < args.window_size:
        raise ValueError("window overlap must satisfy 0 <= overlap < size")
    frozen_dataset = args.dataset.read_bytes()
    cases, dataset_sha256 = load_cases_from_bytes(frozen_dataset)
    if len(cases) != 60:
        raise ValueError(f"Task10 requires exactly 60 cases, found {len(cases)}")

    started = datetime.now(timezone.utc)
    print("building fixed-window corpus", flush=True)
    fixed_items = build_fixed_corpus(
        size=args.window_size,
        overlap=args.window_overlap,
    )
    print("building parent-child corpus", flush=True)
    parent_items, parents = build_parent_child_corpus()

    model = get_model()
    client = _client()
    print(f"embedding fixed-window items={len(fixed_items)}", flush=True)
    fixed_vectors = embed(model, [item.text for item in fixed_items])
    fixed_count = build_ablation_index(
        fixed_items,
        collection_name=FIXED_COLLECTION,
        client=client,
        vectors=fixed_vectors,
    )
    print(f"embedding parent-child items={len(parent_items)}", flush=True)
    parent_vectors = embed(model, [item.text for item in parent_items])
    parent_count = build_ablation_index(
        parent_items,
        collection_name=PARENT_CHILD_COLLECTION,
        client=client,
        vectors=parent_vectors,
    )

    corpora: dict[Chunking, list[CorpusItem]] = {
        "fixed_window": fixed_items,
        "parent_child": parent_items,
    }
    runtime = AblationRuntime(
        client=client,
        model=model,
        corpora=corpora,
        parent_items=parents,
    )
    sizes = {
        FIXED_COLLECTION: fixed_count,
        PARENT_CHILD_COLLECTION: parent_count,
    }
    matrix = generate_config_matrix(
        window_size=args.window_size,
        window_overlap=args.window_overlap,
    )
    results = run_matrix(
        matrix,
        evaluate=lambda config: evaluate_config(
            cases,
            config=config,
            index_size=sizes[AblationRuntime.collection_for(config.chunking)],
            runtime=runtime,
        ),
    )
    finished = datetime.now(timezone.utc)
    report = render_report(
        results,
        dataset_path=args.dataset.as_posix(),
        dataset_sha256=dataset_sha256,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        device=_device_label(),
        embedding_model=MODEL_NAME,
        reranker_model=RERANK_MODEL_NAME,
        collection_sizes=sizes,
    )
    return report, sum(result.error_count for result in results)


def main(
    argv: Sequence[str] | None = None,
    *,
    execute: Callable[[argparse.Namespace], tuple[str, int]] | None = None,
) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = build_parser().parse_args(argv)
    try:
        report, error_count = (execute or _execute_experiment)(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
