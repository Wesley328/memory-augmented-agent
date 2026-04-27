from __future__ import annotations

import os
from dataclasses import dataclass


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass
class Settings:
    openai_api_key: str | None
    openai_base_url: str
    model: str
    temperature: float
    embedding_api_key: str | None
    embedding_base_url: str
    embedding_model: str | None
    top_k: int
    max_memories: int
    embedding_dim: int
    memory_store_backend: str
    memory_sqlite_path: str
    w1: float
    w2: float
    w3: float
    w4: float
    enable_type_aware_retrieval: bool
    enable_query_aware_routing: bool
    enable_mmr_reranking: bool
    mmr_lambda: float
    mmr_candidate_pool_size: int

    @classmethod
    def from_env(cls) -> "Settings":
        w1 = _get_float("MEM_W1", 0.5)
        w2 = _get_float("MEM_W2", 0.2)
        w3 = _get_float("MEM_W3", 0.3)
        w4 = _get_float("MEM_W4", 0.15)
        total = w1 + w2 + w3 + w4
        if total <= 0:
            w1, w2, w3, w4 = 0.45, 0.2, 0.25, 0.1
        else:
            w1, w2, w3, w4 = (
                w1 / total,
                w2 / total,
                w3 / total,
                w4 / total,
            )

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv(
                "OPENAI_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=_get_float("OPENAI_TEMPERATURE", 0.2),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
            embedding_base_url=os.getenv(
                "EMBEDDING_BASE_URL",
                os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL"),
            top_k=_get_int("MEMORY_TOP_K", 3),
            max_memories=_get_int("MEMORY_MAX_SIZE", 200),
            embedding_dim=_get_int("MEMORY_EMBEDDING_DIM", 64),
            memory_store_backend=os.getenv("MEMORY_STORE_BACKEND", "memory"),
            memory_sqlite_path=os.getenv("MEMORY_SQLITE_PATH", "data/memory_agent.db"),
            w1=w1,
            w2=w2,
            w3=w3,
            w4=w4,
            enable_type_aware_retrieval=_get_bool("MEMORY_TYPE_AWARE", True),
            enable_query_aware_routing=_get_bool("MEMORY_QUERY_AWARE_ROUTING", False),
            enable_mmr_reranking=_get_bool("MEMORY_ENABLE_MMR", False),
            mmr_lambda=max(0.0, min(1.0, _get_float("MEMORY_MMR_LAMBDA", 0.7))),
            mmr_candidate_pool_size=max(1, _get_int("MEMORY_MMR_CANDIDATE_POOL", 4)),
        )
