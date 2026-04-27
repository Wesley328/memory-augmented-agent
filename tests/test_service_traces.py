from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app
from service.agent_service import MemoryAgentService


def test_chat_records_turn_trace_and_memory_lifecycle(
    direct_answer_service: MemoryAgentService,
) -> None:
    service = direct_answer_service
    try:
        result = service.chat("帮我梳理后端开发面试的准备计划")

        assert result.turn_id == 1
        assert result.answer == "这是测试回答。"
        assert result.memory_lifecycle is not None
        assert result.memory_lifecycle.extracted == 1
        assert result.memory_lifecycle.added == 1

        traces = service.list_turn_traces()
        assert len(traces) == 1
        trace = traces[0]
        assert trace.turn_id == 1
        assert trace.query == "帮我梳理后端开发面试的准备计划"
        assert trace.turn_plan["action"] == "direct_answer"
        assert trace.memory_lifecycle is not None
        assert trace.memory_lifecycle["added"] == 1

        stored_memories = service.list_memories(limit=10)
        assert len(stored_memories) == 1
        assert stored_memories[0].metadata["source_turn_id"] == 1
    finally:
        service.close()


def test_clarify_branch_still_records_turn_trace(
    clarify_service: MemoryAgentService,
) -> None:
    service = clarify_service
    try:
        result = service.chat("帮我制定一个计划")

        assert result.turn_id == 1
        assert result.turn_plan.action == "clarify_then_wait"
        assert result.memory_lifecycle is None

        trace = service.get_turn_trace(1)
        assert trace is not None
        assert trace.turn_plan["action"] == "clarify_then_wait"
        assert trace.memory_lifecycle is None
        assert trace.answer == "你更想要复习路线图，还是一周日程安排？"
    finally:
        service.close()


def test_traces_api_returns_trace_history(
    direct_answer_service: MemoryAgentService,
) -> None:
    service = direct_answer_service
    app.state.agent_service = service
    client = TestClient(app)

    try:
        chat_response = client.post("/chat", json={"query": "帮我安排后端面试复习顺序"})
        assert chat_response.status_code == 200

        traces_response = client.get("/traces")
        assert traces_response.status_code == 200
        traces_payload = traces_response.json()
        assert traces_payload["count"] == 1
        assert traces_payload["items"][0]["turn_plan"]["action"] == "direct_answer"

        detail_response = client.get("/traces/1")
        assert detail_response.status_code == 200
        detail_payload = detail_response.json()
        assert detail_payload["turn_id"] == 1
        assert detail_payload["query_route"] == "general"
    finally:
        service.close()
