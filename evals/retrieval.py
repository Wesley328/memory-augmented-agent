from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

from core.config import Settings
from core.embedding import ExternalEmbedder
from memory.retriever import MemoryRetriever, SimpleEmbedder
from memory.schema import Memory
from memory.store import InMemoryStore


@dataclass
class RetrievalEvalCase:
    case_id: str
    query: str
    relevant_memory_ids: List[str]
    top_k: int
    notes: str = ""


@dataclass
class RetrievalEvalResult:
    case_id: str
    query: str
    top_k: int
    query_route: str
    relevant_memory_ids: List[str]
    retrieved_memory_ids: List[str]
    retrieved_types: List[str]
    recall_at_k: float
    reciprocal_rank: float
    hit_at_1: float


def _parse_bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


def _load_dataset(dataset_path: Path) -> tuple[List[Memory], List[RetrievalEvalCase]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    memory_items = payload.get("memory_bank", [])
    case_items = payload.get("cases", [])

    memories: List[Memory] = []
    for item in memory_items:
        memory_id = str(item["id"])
        memory = Memory(
            content=str(item["content"]),
            type=str(item["type"]),
            importance=float(item.get("importance", 0.5)),
            confidence=float(item.get("confidence", 0.7)),
            timestamp=datetime.fromisoformat(str(item["timestamp"])),
            metadata={"id": memory_id, "tags": item.get("tags", [])},
        )
        memories.append(memory)

    cases: List[RetrievalEvalCase] = []
    for item in case_items:
        cases.append(
            RetrievalEvalCase(
                case_id=str(item["id"]),
                query=str(item["query"]),
                relevant_memory_ids=[str(value) for value in item["relevant_memory_ids"]],
                top_k=int(item.get("top_k", 3)),
                notes=str(item.get("notes", "")),
            )
        )

    return memories, cases


def _build_embedder(settings: Settings) -> ExternalEmbedder | SimpleEmbedder:
    if settings.embedding_model and settings.embedding_api_key:
        return ExternalEmbedder(
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            base_url=settings.embedding_base_url,
            fallback_dim=settings.embedding_dim,
        )
    return SimpleEmbedder(dim=settings.embedding_dim)


def _recall_at_k(relevant_ids: Sequence[str], retrieved_ids: Sequence[str]) -> float:
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    hits = sum(1 for memory_id in retrieved_ids if memory_id in relevant_set)
    return hits / len(relevant_set)


def _reciprocal_rank(relevant_ids: Sequence[str], retrieved_ids: Sequence[str]) -> float:
    relevant_set = set(relevant_ids)
    for rank, memory_id in enumerate(retrieved_ids, start=1):
        if memory_id in relevant_set:
            return 1.0 / rank
    return 0.0


def run_retrieval_eval(
    dataset_path: Path,
    *,
    use_type_aware: bool | None = None,
    use_query_aware: bool | None = None,
    use_mmr: bool | None = None,
    top_k: int | None = None,
) -> Dict[str, Any]:
    settings = Settings.from_env()
    memories, cases = _load_dataset(dataset_path)
    embedder = _build_embedder(settings)

    store = InMemoryStore(max_size=max(settings.max_memories, len(memories) + 10))
    for memory in memories:
        memory.embedding = embedder.embed(memory.content)
        store.add(memory)

    retriever = MemoryRetriever(
        store=store,
        embedding_fn=embedder.embed,
        w1=settings.w1,
        w2=settings.w2,
        w3=settings.w3,
        w4=settings.w4,
        type_aware=(
            settings.enable_type_aware_retrieval
            if use_type_aware is None
            else use_type_aware
        ),
        query_aware_routing=(
            settings.enable_query_aware_routing
            if use_query_aware is None
            else use_query_aware
        ),
        use_mmr=settings.enable_mmr_reranking if use_mmr is None else use_mmr,
        mmr_lambda=settings.mmr_lambda,
        mmr_candidate_pool_size=settings.mmr_candidate_pool_size,
    )

    case_results: List[RetrievalEvalResult] = []
    for case in cases:
        effective_top_k = top_k or case.top_k
        retrieved = retriever.retrieve(query=case.query, top_k=effective_top_k)
        retrieved_ids = [
            str(item.memory.metadata.get("id", "")) for item in retrieved if item.memory.metadata
        ]
        retrieved_types = [item.memory.type for item in retrieved]
        recall = _recall_at_k(case.relevant_memory_ids, retrieved_ids)
        reciprocal_rank = _reciprocal_rank(case.relevant_memory_ids, retrieved_ids)
        hit_at_1 = 1.0 if retrieved_ids[:1] and retrieved_ids[0] in case.relevant_memory_ids else 0.0
        case_results.append(
            RetrievalEvalResult(
                case_id=case.case_id,
                query=case.query,
                top_k=effective_top_k,
                query_route=retrieved[0].query_route if retrieved else "general",
                relevant_memory_ids=case.relevant_memory_ids,
                retrieved_memory_ids=retrieved_ids,
                retrieved_types=retrieved_types,
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank,
                hit_at_1=hit_at_1,
            )
        )

    summary = {
        "dataset": str(dataset_path),
        "num_memories": len(memories),
        "num_cases": len(case_results),
        "config": {
            "w1": settings.w1,
            "w2": settings.w2,
            "w3": settings.w3,
            "w4_confidence": settings.w4,
            "type_aware": retriever.type_aware,
            "query_aware_routing": retriever.query_aware_routing,
            "use_mmr": retriever.use_mmr,
            "mmr_lambda": retriever.mmr_lambda,
            "mmr_candidate_pool_size": retriever.mmr_candidate_pool_size,
        },
        "metrics": {
            "Recall@K": round(mean(result.recall_at_k for result in case_results), 4),
            "MRR": round(mean(result.reciprocal_rank for result in case_results), 4),
            "Hit@1": round(mean(result.hit_at_1 for result in case_results), 4),
        },
        "cases": [
            {
                "case_id": result.case_id,
                "query": result.query,
                "top_k": result.top_k,
                "query_route": result.query_route,
                "relevant_memory_ids": result.relevant_memory_ids,
                "retrieved_memory_ids": result.retrieved_memory_ids,
                "retrieved_types": result.retrieved_types,
                "recall_at_k": round(result.recall_at_k, 4),
                "reciprocal_rank": round(result.reciprocal_rank, 4),
                "hit_at_1": round(result.hit_at_1, 4),
            }
            for result in case_results
        ],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a lightweight offline retrieval evaluation on the local memory agent."
    )
    parser.add_argument(
        "--dataset",
        default="evals/data/retrieval_eval_sample.json",
        help="Path to the retrieval evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-k for every evaluation case.",
    )
    parser.add_argument(
        "--type-aware",
        type=_parse_bool_flag,
        default=None,
        help="Override MEMORY_TYPE_AWARE for this evaluation run.",
    )
    parser.add_argument(
        "--mmr",
        type=_parse_bool_flag,
        default=None,
        help="Override MEMORY_ENABLE_MMR for this evaluation run.",
    )
    parser.add_argument(
        "--query-aware",
        type=_parse_bool_flag,
        default=None,
        help="Override MEMORY_QUERY_AWARE_ROUTING for this evaluation run.",
    )
    args = parser.parse_args()

    summary = run_retrieval_eval(
        Path(args.dataset),
        use_type_aware=args.type_aware,
        use_query_aware=args.query_aware,
        use_mmr=args.mmr,
        top_k=args.top_k,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
