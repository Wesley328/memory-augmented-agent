from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any, List

from dotenv import load_dotenv

from agent.executor import AgentExecutor
from agent.planner import PromptPlanner, TurnPlan
from agent.self_check import ResponseSelfChecker, SelfCheckResult
from core.config import Settings
from core.embedding import ExternalEmbedder
from core.llm import LLMClient, LLMError
from memory.extractor import MemoryExtractor
from memory.retriever import MemoryRetriever, ScoredMemory, SimpleEmbedder
from memory.schema import Memory, utcnow
from memory.store import create_memory_store
from memory.updater import MemoryUpdater, UpdateStats
from observability.schema import TurnTrace
from observability.trace_store import create_turn_trace_store
from tools.search import SearchTool

load_dotenv()


@dataclass(frozen=True)
class MemoryLifecycleResult:
    extracted: int
    added: int
    updated: int
    versioned: int
    removed: int
    extraction_error: str | None = None


@dataclass(frozen=True)
class ChatTurnResult:
    query: str
    answer: str
    query_route: str
    turn_plan: TurnPlan
    retrieved_memories: List[ScoredMemory]
    tool_context: str | None
    self_check: SelfCheckResult | None
    memory_lifecycle: MemoryLifecycleResult | None
    turn_id: int | None


class MemoryAgentService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self._lock = RLock()
        self.llm = LLMClient(self.settings)
        self.store = create_memory_store(
            backend=self.settings.memory_store_backend,
            max_size=self.settings.max_memories,
            sqlite_path=self.settings.memory_sqlite_path,
        )
        self.trace_store = create_turn_trace_store(
            backend=self.settings.memory_store_backend,
            sqlite_path=self.settings.memory_sqlite_path,
        )
        self.embedder = self._build_embedder()
        self.retriever = MemoryRetriever(
            store=self.store,
            embedding_fn=self.embedder.embed,
            w1=self.settings.w1,
            w2=self.settings.w2,
            w3=self.settings.w3,
            w4=self.settings.w4,
            type_aware=self.settings.enable_type_aware_retrieval,
            query_aware_routing=self.settings.enable_query_aware_routing,
            use_mmr=self.settings.enable_mmr_reranking,
            mmr_lambda=self.settings.mmr_lambda,
            mmr_candidate_pool_size=self.settings.mmr_candidate_pool_size,
        )
        self.extractor = MemoryExtractor(self.llm)
        self.updater = MemoryUpdater(max_memories=self.settings.max_memories)
        self.planner = PromptPlanner()
        self.executor = AgentExecutor(self.llm)
        self.self_checker = ResponseSelfChecker()
        self.search_tool = SearchTool()
        self._turn_id = 0

    def chat(self, user_query: str) -> ChatTurnResult:
        normalized_query = user_query.strip()
        if not normalized_query:
            raise ValueError("user_query must not be empty")

        with self._lock:
            query_route = self.retriever.detect_query_route(normalized_query)
            retrieved = self.retriever.retrieve(
                query=normalized_query,
                top_k=self.settings.top_k,
            )
            turn_plan = self.planner.plan_turn(
                user_query=normalized_query,
                memories=retrieved,
                query_route=query_route,
            )
            self._turn_id += 1
            turn_id = self._turn_id

            if turn_plan.action == "clarify_then_wait":
                result = ChatTurnResult(
                    query=normalized_query,
                    answer=turn_plan.clarification_message or "",
                    query_route=query_route,
                    turn_plan=turn_plan,
                    retrieved_memories=retrieved,
                    tool_context=None,
                    self_check=None,
                    memory_lifecycle=None,
                    turn_id=turn_id,
                )
                self._record_turn_trace(result)
                return result

            tool_context = None
            if turn_plan.action == "tool_then_answer" and turn_plan.tool_name == "search":
                tool_context = self.search_tool.search(turn_plan.tool_query or normalized_query)

            prompt = self.planner.build_prompt(
                user_query=normalized_query,
                memories=retrieved,
                query_route=query_route,
                turn_plan=turn_plan,
                tool_context=tool_context,
            )

            answer = self.executor.respond(prompt)
            self_check = self.self_checker.check(
                user_query=normalized_query,
                answer=answer,
                memories=retrieved,
                turn_plan=turn_plan,
                tool_context=tool_context,
            )
            final_answer = self_check.final_answer

            extracted, memory_lifecycle = self._extract_and_update(
                user_query=normalized_query,
                answer=final_answer,
                query_route=query_route,
                turn_id=turn_id,
            )

            result = ChatTurnResult(
                query=normalized_query,
                answer=final_answer,
                query_route=query_route,
                turn_plan=turn_plan,
                retrieved_memories=retrieved,
                tool_context=tool_context,
                self_check=self_check,
                memory_lifecycle=memory_lifecycle,
                turn_id=turn_id,
            )
            self._record_turn_trace(result)
            return result

    def list_memories(
        self,
        *,
        status: str | None = None,
        memory_type: str | None = None,
        lineage_id: str | None = None,
        memory_ids: List[str] | None = None,
        summary_only: bool | None = None,
        limit: int | None = 20,
    ) -> List[Memory]:
        return self.store.query_memories(
            status=status,
            memory_type=memory_type,
            lineage_id=lineage_id,
            memory_ids=memory_ids,
            summary_only=summary_only,
            limit=limit,
        )

    def get_lineage(self, lineage_id: str) -> List[Memory]:
        return self.store.get_lineage(lineage_id)

    def get_memory_stats(self) -> dict[str, int]:
        all_memories = self.store.query_memories(limit=None)
        active_count = sum(
            1
            for memory in all_memories
            if str(memory.metadata.get("status", "active")) == "active"
        )
        superseded_count = sum(
            1
            for memory in all_memories
            if str(memory.metadata.get("status", "active")) == "superseded"
        )
        summary_count = sum(
            1
            for memory in all_memories
            if memory.type == "summary" or bool(memory.metadata.get("is_summary"))
        )
        return {
            "total": len(all_memories),
            "active": active_count,
            "summary": summary_count,
            "superseded": superseded_count,
        }

    def list_turn_traces(
        self,
        *,
        query_route: str | None = None,
        planner_action: str | None = None,
        limit: int | None = 20,
    ) -> list[TurnTrace]:
        return self.trace_store.list(
            query_route=query_route,
            planner_action=planner_action,
            limit=limit,
        )

    def get_turn_trace(self, turn_id: int) -> TurnTrace | None:
        return self.trace_store.get(turn_id)

    def get_status(self) -> dict[str, Any]:
        return {
            "llm": {
                "ready": self.llm.is_ready,
                "configuration_issue": self.llm.configuration_issue,
                "model": self.settings.model,
                "base_url": self.settings.openai_base_url,
            },
            "embedding": {
                "model": self.settings.embedding_model,
                "backend": self.embedder.__class__.__name__,
            },
            "store": {
                "backend": self.settings.memory_store_backend,
                "sqlite_path": self.settings.memory_sqlite_path,
                "memory_count": len(self.store),
            },
            "features": {
                "type_aware_retrieval": self.settings.enable_type_aware_retrieval,
                "query_aware_routing": self.settings.enable_query_aware_routing,
                "mmr_reranking": self.settings.enable_mmr_reranking,
            },
        }

    def get_health(self) -> dict[str, Any]:
        status = self.get_status()
        return {
            "ok": True,
            "llm_ready": status["llm"]["ready"],
            "store_backend": status["store"]["backend"],
            "memory_count": status["store"]["memory_count"],
        }

    def close(self) -> None:
        close_targets = [self.store, self.trace_store]
        closed_ids: set[int] = set()
        for target in close_targets:
            if id(target) in closed_ids:
                continue
            closed_ids.add(id(target))
            close_fn = getattr(target, "close", None)
            if callable(close_fn):
                close_fn()

    def serialize_memory(self, memory: Memory) -> dict[str, Any]:
        return {
            "content": memory.content,
            "type": memory.type,
            "importance": memory.importance,
            "confidence": memory.confidence,
            "timestamp": memory.timestamp.isoformat(),
            "embedding_dim": len(memory.embedding) if memory.embedding else 0,
            "metadata": dict(memory.metadata),
        }

    def serialize_scored_memory(self, item: ScoredMemory) -> dict[str, Any]:
        return {
            "memory": self.serialize_memory(item.memory),
            "score": item.score,
            "base_score": item.base_score,
            "relevance": item.relevance,
            "recency": item.recency,
            "importance": item.importance,
            "confidence": item.confidence,
            "weight_relevance": item.weight_relevance,
            "weight_recency": item.weight_recency,
            "weight_importance": item.weight_importance,
            "weight_confidence": item.weight_confidence,
            "diversity_penalty": item.diversity_penalty,
            "query_route": item.query_route,
        }

    def serialize_turn_plan(self, turn_plan: TurnPlan) -> dict[str, Any]:
        return asdict(turn_plan)

    def serialize_self_check(self, result: SelfCheckResult | None) -> dict[str, Any] | None:
        if result is None:
            return None
        return {
            "passed": result.passed,
            "revised": result.revised,
            "has_blocking_issue": result.has_blocking_issue,
            "summary": result.summary(),
            "issues": [asdict(issue) for issue in result.issues],
            "original_answer": result.original_answer,
            "final_answer": result.final_answer,
        }

    def serialize_chat_turn(self, result: ChatTurnResult) -> dict[str, Any]:
        return {
            "query": result.query,
            "answer": result.answer,
            "query_route": result.query_route,
            "turn_id": result.turn_id,
            "turn_plan": self.serialize_turn_plan(result.turn_plan),
            "tool_context": result.tool_context,
            "self_check": self.serialize_self_check(result.self_check),
            "retrieved_memories": [
                self.serialize_scored_memory(item) for item in result.retrieved_memories
            ],
            "memory_lifecycle": (
                asdict(result.memory_lifecycle) if result.memory_lifecycle is not None else None
            ),
        }

    def serialize_turn_trace(self, trace: TurnTrace) -> dict[str, Any]:
        return trace.to_dict()

    def _record_turn_trace(self, result: ChatTurnResult) -> None:
        if result.turn_id is None:
            return
        self.trace_store.add(self._build_turn_trace(result))

    def _build_turn_trace(self, result: ChatTurnResult) -> TurnTrace:
        return TurnTrace(
            turn_id=result.turn_id or 0,
            created_at=utcnow(),
            query=result.query,
            answer=result.answer,
            query_route=result.query_route,
            turn_plan=self.serialize_turn_plan(result.turn_plan),
            tool_context=result.tool_context,
            self_check=self.serialize_self_check(result.self_check),
            retrieved_memories=[
                self.serialize_scored_memory(item) for item in result.retrieved_memories
            ],
            memory_lifecycle=(
                asdict(result.memory_lifecycle) if result.memory_lifecycle is not None else None
            ),
        )

    def _extract_and_update(
        self,
        *,
        user_query: str,
        answer: str,
        query_route: str,
        turn_id: int,
    ) -> tuple[List[Memory], MemoryLifecycleResult]:
        extraction_error = None
        extracted: List[Memory] = []
        try:
            extracted = self.extractor.extract(
                user_query,
                answer,
                turn_id=turn_id,
                query_route=query_route,
            )
        except LLMError as exc:
            extraction_error = exc.user_message

        for memory in extracted:
            memory.embedding = self.embedder.embed(memory.content)

        stats = self._empty_update_stats()
        if extracted:
            stats = self.updater.update(self.store, extracted)

        return (
            extracted,
            MemoryLifecycleResult(
                extracted=len(extracted),
                added=stats.added,
                updated=stats.updated,
                versioned=stats.versioned,
                removed=stats.removed,
                extraction_error=extraction_error,
            ),
        )

    def _build_embedder(self) -> ExternalEmbedder | SimpleEmbedder:
        if self.settings.embedding_model and self.settings.embedding_api_key:
            return ExternalEmbedder(
                api_key=self.settings.embedding_api_key,
                model=self.settings.embedding_model,
                base_url=self.settings.embedding_base_url,
                fallback_dim=self.settings.embedding_dim,
            )
        return SimpleEmbedder(dim=self.settings.embedding_dim)

    def _empty_update_stats(self) -> UpdateStats:
        return UpdateStats(
            added=0,
            updated=0,
            versioned=0,
            removed=0,
        )
