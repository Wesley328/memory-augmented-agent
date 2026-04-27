from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User message to send to the agent")


class MemoryPayload(BaseModel):
    content: str
    type: str
    importance: float
    confidence: float
    timestamp: str
    embedding_dim: int
    metadata: dict[str, Any]


class RetrievedMemoryPayload(BaseModel):
    memory: MemoryPayload
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
    diversity_penalty: float
    query_route: str


class TurnPlanPayload(BaseModel):
    action: str
    query_route: str
    response_language: str
    reason: str
    tool_name: str | None = None
    tool_query: str | None = None
    clarification_message: str | None = None


class SelfCheckIssuePayload(BaseModel):
    code: str
    severity: str
    message: str


class SelfCheckPayload(BaseModel):
    passed: bool
    revised: bool
    has_blocking_issue: bool
    summary: str
    issues: List[SelfCheckIssuePayload]
    original_answer: str
    final_answer: str


class MemoryLifecyclePayload(BaseModel):
    extracted: int
    added: int
    updated: int
    versioned: int
    removed: int
    extraction_error: str | None = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    query_route: str
    turn_id: int | None = None
    turn_plan: TurnPlanPayload
    tool_context: str | None = None
    self_check: SelfCheckPayload | None = None
    retrieved_memories: List[RetrievedMemoryPayload]
    memory_lifecycle: MemoryLifecyclePayload | None = None


class TurnTracePayload(BaseModel):
    turn_id: int
    created_at: str
    query: str
    answer: str
    query_route: str
    turn_plan: TurnPlanPayload
    tool_context: str | None = None
    self_check: SelfCheckPayload | None = None
    retrieved_memories: List[RetrievedMemoryPayload]
    memory_lifecycle: MemoryLifecyclePayload | None = None


class TurnTracesResponse(BaseModel):
    items: List[TurnTracePayload]
    count: int


class MemoriesResponse(BaseModel):
    items: List[MemoryPayload]
    count: int


class MemoryStatsResponse(BaseModel):
    total: int
    active: int
    summary: int
    superseded: int


class HealthResponse(BaseModel):
    ok: bool
    llm_ready: bool
    store_backend: str
    memory_count: int


class StatusResponse(BaseModel):
    llm: dict[str, Any]
    embedding: dict[str, Any]
    store: dict[str, Any]
    features: dict[str, Any]


class ErrorResponse(BaseModel):
    detail: str
