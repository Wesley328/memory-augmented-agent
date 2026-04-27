from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles

from core.llm import LLMConfigurationError, LLMError, LLMRequestError
from service.agent_service import MemoryAgentService

from .schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    MemoryStatsResponse,
    MemoriesResponse,
    StatusResponse,
    TurnTracePayload,
    TurnTracesResponse,
)

WEB_APP_DIR = Path(__file__).resolve().parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = getattr(app.state, "agent_service", None)
    if service is None:
        service = MemoryAgentService()
        app.state.agent_service = service
    try:
        yield
    finally:
        service.close()


app = FastAPI(
    title="Memory Agent API",
    version="0.1.0",
    description="FastAPI service layer for the memory-augmented agent project.",
    lifespan=lifespan,
)
app.mount("/app", StaticFiles(directory=WEB_APP_DIR, html=True), name="web-app")


def get_service(request: Request) -> MemoryAgentService:
    service = getattr(request.app.state, "agent_service", None)
    if service is None:
        service = MemoryAgentService()
        request.app.state.agent_service = service
    return service


def _raise_from_llm_error(exc: LLMError) -> None:
    if isinstance(exc, LLMConfigurationError):
        raise HTTPException(status_code=503, detail=exc.user_message) from exc
    if isinstance(exc, LLMRequestError):
        raise HTTPException(status_code=502, detail=exc.user_message) from exc
    raise HTTPException(status_code=500, detail=exc.user_message) from exc


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "name": "memory-agent-api",
        "message": "Memory Agent FastAPI service is running. Open /app for the minimal web UI.",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    service = get_service(request)
    return HealthResponse.model_validate(service.get_health())


@app.get("/status", response_model=StatusResponse, tags=["meta"])
def status(request: Request) -> StatusResponse:
    service = get_service(request)
    return StatusResponse.model_validate(service.get_status())


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["chat"],
)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    service = get_service(request)
    try:
        result = service.chat(payload.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        _raise_from_llm_error(exc)

    return ChatResponse.model_validate(service.serialize_chat_turn(result))


@app.get("/memories", response_model=MemoriesResponse, tags=["memory"])
def list_memories(
    request: Request,
    status: str | None = Query(default=None),
    memory_type: str | None = Query(default=None),
    lineage_id: str | None = Query(default=None),
    memory_ids: str | None = Query(default=None, description="Comma-separated memory_id list"),
    summary_only: bool | None = Query(default=None),
    limit: int | None = Query(default=20, ge=1, le=200),
) -> MemoriesResponse:
    service = get_service(request)
    parsed_memory_ids = None
    if memory_ids is not None:
        parsed_memory_ids = [
            item.strip() for item in memory_ids.split(",") if item.strip()
        ]
    memories = service.list_memories(
        status=status,
        memory_type=memory_type,
        lineage_id=lineage_id,
        memory_ids=parsed_memory_ids,
        summary_only=summary_only,
        limit=limit,
    )
    items = [service.serialize_memory(memory) for memory in memories]
    return MemoriesResponse.model_validate(
        {
            "items": items,
            "count": len(items),
        }
    )


@app.get("/memory-stats", response_model=MemoryStatsResponse, tags=["memory"])
def memory_stats(request: Request) -> MemoryStatsResponse:
    service = get_service(request)
    return MemoryStatsResponse.model_validate(service.get_memory_stats())


@app.get("/traces", response_model=TurnTracesResponse, tags=["trace"])
def list_traces(
    request: Request,
    query_route: str | None = Query(default=None),
    planner_action: str | None = Query(default=None),
    limit: int | None = Query(default=20, ge=1, le=200),
) -> TurnTracesResponse:
    service = get_service(request)
    traces = service.list_turn_traces(
        query_route=query_route,
        planner_action=planner_action,
        limit=limit,
    )
    items = [service.serialize_turn_trace(trace) for trace in traces]
    return TurnTracesResponse.model_validate(
        {
            "items": items,
            "count": len(items),
        }
    )


@app.get("/traces/{turn_id}", response_model=TurnTracePayload, tags=["trace"])
def get_trace(turn_id: int, request: Request) -> TurnTracePayload:
    service = get_service(request)
    trace = service.get_turn_trace(turn_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Turn trace {turn_id} not found.")
    return TurnTracePayload.model_validate(service.serialize_turn_trace(trace))


@app.get("/memories/{lineage_id}/lineage", response_model=MemoriesResponse, tags=["memory"])
def get_lineage(lineage_id: str, request: Request) -> MemoriesResponse:
    service = get_service(request)
    memories = service.get_lineage(lineage_id)
    items = [service.serialize_memory(memory) for memory in memories]
    return MemoriesResponse.model_validate(
        {
            "items": items,
            "count": len(items),
        }
    )
