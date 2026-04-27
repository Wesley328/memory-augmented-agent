from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from memory.schema import Memory, utcnow
from memory.store import MemoryStore

EmbeddingFn = Callable[[str], List[float]]
TypeWeightProfile = Tuple[float, float, float]
QueryRouteTypeProfiles = Dict[str, TypeWeightProfile]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SimpleEmbedder:
    """
    Tiny hashing-based embedder for a no-dependency demo.

    It keeps this project runnable without external embedding services.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = _tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            idx = hash(token) % self.dim
            vec[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


DEFAULT_TYPE_WEIGHT_PROFILES: Dict[str, TypeWeightProfile] = {
    # preference: emphasize long-term user taste and stable personalization
    "preference": (1.10, 0.85, 1.25),
    # event/history: emphasize timeliness for recent conversational context
    "event": (0.95, 1.35, 0.90),
    "history": (0.90, 1.25, 0.85),
    # summary: compact high-signal memory synthesized from multiple atomic memories
    "summary": (1.10, 0.95, 1.15),
    # fact/user_profile: prioritize semantic match + durable profile info
    "fact": (1.15, 0.90, 1.00),
    "user_profile": (1.20, 0.85, 1.10),
}


@dataclass(frozen=True)
class QueryRouteProfile:
    weight_multipliers: TypeWeightProfile = (1.0, 1.0, 1.0)
    type_profiles: QueryRouteTypeProfiles = field(default_factory=dict)


DEFAULT_QUERY_ROUTE_PROFILES: Dict[str, QueryRouteProfile] = {
    "general": QueryRouteProfile(),
    "preference": QueryRouteProfile(
        weight_multipliers=(1.05, 0.85, 1.15),
        type_profiles={
            "preference": (1.25, 0.90, 1.20),
            "summary": (1.10, 0.95, 1.10),
            "user_profile": (1.10, 0.90, 1.05),
        },
    ),
    "temporal": QueryRouteProfile(
        weight_multipliers=(0.95, 1.30, 0.90),
        type_profiles={
            "event": (1.05, 1.30, 0.95),
            "history": (1.00, 1.25, 0.90),
            "summary": (0.95, 1.05, 1.00),
        },
    ),
    "profile": QueryRouteProfile(
        weight_multipliers=(1.20, 0.85, 1.00),
        type_profiles={
            "summary": (1.20, 0.95, 1.15),
            "user_profile": (1.30, 0.90, 1.10),
            "fact": (1.15, 0.95, 1.00),
        },
    ),
    "planning": QueryRouteProfile(
        weight_multipliers=(1.05, 1.10, 1.05),
        type_profiles={
            "event": (1.10, 1.20, 1.00),
            "history": (1.00, 1.10, 0.95),
            "summary": (1.15, 1.00, 1.10),
            "user_profile": (1.10, 0.95, 1.05),
        },
    ),
}


@dataclass
class ScoredMemory:
    memory: Memory
    score: float
    base_score: float
    relevance: float
    recency: float
    importance: float
    confidence: float
    weight_relevance: float
    weight_recency: float
    weight_importance: float
    weight_confidence: float
    diversity_penalty: float = 0.0
    query_route: str = "general"


class MemoryRetriever:
    def __init__(
        self,
        store: MemoryStore,
        embedding_fn: Optional[EmbeddingFn] = None,
        w1: float = 0.5,
        w2: float = 0.2,
        w3: float = 0.3,
        w4: float = 0.1,
        type_aware: bool = True,
        type_weight_profiles: Optional[Dict[str, TypeWeightProfile]] = None,
        query_aware_routing: bool = False,
        query_route_profiles: Optional[Dict[str, QueryRouteProfile]] = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
        mmr_candidate_pool_size: int = 10,
    ) -> None:
        self.store = store
        self.embedding_fn = embedding_fn
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.type_aware = type_aware
        self.type_weight_profiles = self._normalize_type_profiles(type_weight_profiles)
        self.query_aware_routing = query_aware_routing
        self.query_route_profiles = self._normalize_query_route_profiles(query_route_profiles)
        self.use_mmr = use_mmr
        self.mmr_lambda = max(0.0, min(1.0, mmr_lambda))
        self.mmr_candidate_pool_size = max(1, mmr_candidate_pool_size)

    def retrieve(
        self, query: str, top_k: int = 5, now: Optional[datetime] = None
    ) -> List[ScoredMemory]:
        now = now or utcnow()
        memories = [
            memory
            for memory in self.store.all()
            if str(memory.metadata.get("status", "active")) == "active"
        ]
        if not memories:
            return []

        query_embedding = self.embedding_fn(query) if self.embedding_fn else None
        query_route = self.detect_query_route(query)
        scored_items: List[ScoredMemory] = []

        for memory in memories:
            relevance = self._compute_relevance(query, query_embedding, memory)
            time_diff_hours = max((now - memory.timestamp).total_seconds() / 3600.0, 0.0)
            recency = 1.0 / (1.0 + time_diff_hours)
            importance = memory.importance
            confidence = memory.confidence
            
            (   score,
                weight_relevance,
                weight_recency,
                weight_importance,
                weight_confidence,
            ) = self._score_with_type_profile(
                memory_type=memory.type,
                query_route=query_route,
                relevance=relevance,
                recency=recency,
                importance=importance,
                confidence=confidence,
            )
            scored_items.append(
                ScoredMemory(
                    memory=memory,
                    score=score,
                    base_score=score,
                    relevance=relevance,
                    recency=recency,
                    importance=importance,
                    confidence=confidence,
                    weight_relevance=weight_relevance,
                    weight_recency=weight_recency,
                    weight_importance=weight_importance,
                    weight_confidence=weight_confidence,
                    query_route=query_route,
                )
            )

        scored_items.sort(key=lambda item: item.base_score, reverse=True)
        if not self.use_mmr or len(scored_items) <= 1:
            return scored_items[:top_k]
        return self._rerank_with_mmr(scored_items, top_k=top_k)

    def detect_query_route(self, query: str) -> str:
        if not self.query_aware_routing:
            return "general"
        return self._detect_query_route(query)

    def _compute_relevance(
        self, query: str, query_embedding: Optional[List[float]], memory: Memory
    ) -> float:
        if query_embedding is not None and memory.embedding is not None:
            similarity = _cosine_similarity(query_embedding, memory.embedding)
            return max(0.0, min(1.0, (similarity + 1.0) / 2.0))

        query_tokens = set(_tokenize(query))
        memory_tokens = set(_tokenize(memory.content))
        if not query_tokens or not memory_tokens:
            return 0.0
        overlap = len(query_tokens & memory_tokens)
        union = len(query_tokens | memory_tokens)
        return overlap / union

    def _normalize_type_profiles(
        self, type_weight_profiles: Optional[Dict[str, TypeWeightProfile]]
    ) -> Dict[str, TypeWeightProfile]:
        profiles = dict(DEFAULT_TYPE_WEIGHT_PROFILES)
        if not type_weight_profiles:
            return profiles

        for memory_type, multipliers in type_weight_profiles.items():
            if len(multipliers) != 3:
                continue
            rel_mult, rec_mult, imp_mult = multipliers
            try:
                profiles[memory_type.strip().lower()] = (
                    max(float(rel_mult), 0.0),
                    max(float(rec_mult), 0.0),
                    max(float(imp_mult), 0.0),
                )
            except (TypeError, ValueError):
                continue
        return profiles

    def _normalize_query_route_profiles(
        self, query_route_profiles: Optional[Dict[str, QueryRouteProfile]]
    ) -> Dict[str, QueryRouteProfile]:
        profiles = dict(DEFAULT_QUERY_ROUTE_PROFILES)
        if not query_route_profiles:
            return profiles

        for route_name, profile in query_route_profiles.items():
            if not isinstance(profile, QueryRouteProfile):
                continue
            normalized_route = route_name.strip().lower()
            weight_multipliers = self._coerce_weight_profile(profile.weight_multipliers)
            type_profiles: QueryRouteTypeProfiles = {}
            for memory_type, multipliers in profile.type_profiles.items():
                coerced = self._coerce_weight_profile(multipliers)
                if coerced is not None:
                    type_profiles[memory_type.strip().lower()] = coerced
            profiles[normalized_route] = QueryRouteProfile(
                weight_multipliers=weight_multipliers or (1.0, 1.0, 1.0),
                type_profiles=type_profiles,
            )
        return profiles

    def _coerce_weight_profile(
        self, multipliers: TypeWeightProfile | Tuple[float, float, float]
    ) -> TypeWeightProfile | None:
        if len(multipliers) != 3:
            return None
        rel_mult, rec_mult, imp_mult = multipliers
        try:
            return (
                max(float(rel_mult), 0.0),
                max(float(rec_mult), 0.0),
                max(float(imp_mult), 0.0),
            )
        except (TypeError, ValueError):
            return None

    def _detect_query_route(self, query: str) -> str:
        normalized = query.strip().lower()
        if not normalized:
            return "general"

        temporal_keywords = {
            "recent",
            "recently",
            "yesterday",
            "today",
            "latest",
            "last",
            "before",
            "when",
            "trip",
            "travel",
            "visited",
            "happened",
            "最近",
            "刚刚",
            "刚才",
            "昨天",
            "今天",
            "上次",
            "上周",
            "上个月",
            "什么时候",
            "何时",
            "发生",
            "提到过",
            "旅行",
            "旅游",
            "出差",
            "去过",
        }
        planning_keywords = {
            "plan",
            "planning",
            "prepare",
            "prep",
            "upcoming",
            "next",
            "schedule",
            "deadline",
            "interview",
            "todo",
            "计划",
            "规划",
            "准备",
            "复习",
            "安排",
            "下一步",
            "接下来",
            "即将",
            "面试",
            "日程",
            "ddl",
            "截止",
            "待办",
        }
        profile_keywords = {
            "background",
            "profile",
            "bio",
            "about me",
            "who am i",
            "my name",
            "major",
            "skill",
            "skills",
            "experience",
            "背景",
            "简介",
            "介绍",
            "我是",
            "我是谁",
            "名字",
            "姓名",
            "专业",
            "技能",
            "经历",
            "经验",
        }
        preference_keywords = {
            "recommend",
            "recommendation",
            "suggest",
            "favorite",
            "favourite",
            "prefer",
            "like",
            "love",
            "enjoy",
            "taste",
            "food",
            "restaurant",
            "dinner",
            "lunch",
            "eat",
            "推荐",
            "建议",
            "喜欢",
            "偏好",
            "爱吃",
            "想吃",
            "口味",
            "餐厅",
            "吃什么",
            "晚饭",
            "午饭",
            "晚餐",
            "午餐",
            "美食",
        }

        if _contains_any(normalized, planning_keywords):
            return "planning"
        if _contains_any(normalized, temporal_keywords):
            return "temporal"
        if _contains_any(normalized, profile_keywords):
            return "profile"
        if _contains_any(normalized, preference_keywords):
            return "preference"
        return "general"

    def _score_with_type_profile(
        self,
        memory_type: str,
        query_route: str,
        relevance: float,
        recency: float,
        importance: float,
        confidence: float,
    ) -> Tuple[float, float, float, float, float]:
        if self.type_aware:
            rel_mult, rec_mult, imp_mult = self.type_weight_profiles.get(
                memory_type, (1.0, 1.0, 1.0)
            )
        else:
            rel_mult, rec_mult, imp_mult = (1.0, 1.0, 1.0)

        route_profile = self.query_route_profiles.get(query_route, DEFAULT_QUERY_ROUTE_PROFILES["general"])
        route_rel_mult, route_rec_mult, route_imp_mult = route_profile.weight_multipliers
        route_type_rel, route_type_rec, route_type_imp = route_profile.type_profiles.get(
            memory_type,
            (1.0, 1.0, 1.0),
        )

        weighted_relevance = self.w1 * rel_mult * route_rel_mult * route_type_rel
        weighted_recency = self.w2 * rec_mult * route_rec_mult * route_type_rec
        weighted_importance = self.w3 * imp_mult * route_imp_mult * route_type_imp
        weighted_confidence = self.w4
        total = (
            weighted_relevance
            + weighted_recency
            + weighted_importance
            + weighted_confidence
        )

        if total <= 0.0:
            weighted_relevance, weighted_recency, weighted_importance, weighted_confidence = (
                self.w1,
                self.w2,
                self.w3,
                self.w4,
            )
            total = (
                weighted_relevance
                + weighted_recency
                + weighted_importance
                + weighted_confidence
            )

        # Normalize to keep weighted sum stable and interpretable.
        weight_relevance = weighted_relevance / total
        weight_recency = weighted_recency / total
        weight_importance = weighted_importance / total
        weight_confidence = weighted_confidence / total

        score = (
            weight_relevance * relevance
            + weight_recency * recency
            + weight_importance * importance
            + weight_confidence * confidence
        )
        return (
            score,
            weight_relevance,
            weight_recency,
            weight_importance,
            weight_confidence,
        )

    def _rerank_with_mmr(
        self, scored_items: List[ScoredMemory], top_k: int
    ) -> List[ScoredMemory]:
        if top_k <= 0:
            return []

        candidate_pool_size = min(
            len(scored_items),
            max(top_k, top_k * self.mmr_candidate_pool_size),
        )
        candidates = list(scored_items[:candidate_pool_size])
        selected: List[ScoredMemory] = []

        while candidates and len(selected) < top_k:
            best_item: Optional[ScoredMemory] = None
            best_score = float("-inf")
            best_penalty = 0.0

            for candidate in candidates:
                novelty_penalty = self._max_similarity(candidate, selected)
                mmr_score = (
                    self.mmr_lambda * candidate.base_score
                    - (1.0 - self.mmr_lambda) * novelty_penalty
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = candidate
                    best_penalty = novelty_penalty

            if best_item is None:
                break

            candidates.remove(best_item)
            selected.append(
                ScoredMemory(
                    memory=best_item.memory,
                    score=best_score,
                    base_score=best_item.base_score,
                    relevance=best_item.relevance,
                    recency=best_item.recency,
                    importance=best_item.importance,
                    confidence=best_item.confidence,
                    weight_relevance=best_item.weight_relevance,
                    weight_recency=best_item.weight_recency,
                    weight_importance=best_item.weight_importance,
                    weight_confidence=best_item.weight_confidence,
                    diversity_penalty=best_penalty,
                    query_route=best_item.query_route,
                )
            )

        return selected

    def _max_similarity(
        self, candidate: ScoredMemory, selected: List[ScoredMemory]
    ) -> float:
        if not selected:
            return 0.0
        return max(
            self._memory_similarity(candidate.memory, chosen.memory) for chosen in selected
        )

    def _memory_similarity(self, left: Memory, right: Memory) -> float:
        if left.embedding is not None and right.embedding is not None:
            similarity = _cosine_similarity(left.embedding, right.embedding)
            return max(0.0, min(1.0, (similarity + 1.0) / 2.0))

        left_tokens = set(_tokenize(left.content))
        right_tokens = set(_tokenize(right.content))
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return overlap / union
