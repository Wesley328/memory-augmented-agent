from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.planner import TurnPlan
from agent.self_check import SelfCheckIssue, SelfCheckResult
from core.config import Settings
from memory.schema import Memory
from service.agent_service import MemoryAgentService


@pytest.fixture
def base_settings() -> Settings:
    settings = Settings.from_env()
    settings.memory_store_backend = "memory"
    settings.memory_sqlite_path = "tmp/test_memory_agent.db"
    settings.embedding_model = None
    settings.embedding_api_key = None
    settings.enable_query_aware_routing = False
    settings.enable_mmr_reranking = False
    settings.enable_type_aware_retrieval = True
    return settings


@pytest.fixture
def direct_answer_service(base_settings: Settings) -> MemoryAgentService:
    service = MemoryAgentService(settings=base_settings)
    service.planner.plan_turn = lambda *args, **kwargs: TurnPlan(
        action="direct_answer",
        query_route="general",
        response_language="zh",
        reason="Test planner direct answer path.",
    )
    service.executor.respond = lambda prompt: "这是测试回答。"
    service.self_checker.check = lambda **kwargs: SelfCheckResult(
        issues=[],
        original_answer="这是测试回答。",
        final_answer="这是测试回答。",
    )

    def fake_extract(user_message: str, assistant_message: str = "", turn_id: int | None = None, query_route: str = "general"):
        timestamp = datetime.now(timezone.utc)
        return [
            Memory(
                content="用户正在准备后端开发面试",
                type="fact",
                importance=0.85,
                confidence=0.92,
                timestamp=timestamp,
                metadata={
                    "status": "active",
                    "version": 1,
                    "memory_id": "memory-direct-1",
                    "lineage_id": "memory-direct-1",
                    "topic_key": "fact:backend_interview",
                    "source_turn_id": turn_id,
                    "query_route_at_extraction": query_route,
                },
            )
        ]

    service.extractor.extract = fake_extract
    return service


@pytest.fixture
def clarify_service(base_settings: Settings) -> MemoryAgentService:
    service = MemoryAgentService(settings=base_settings)
    service.planner.plan_turn = lambda *args, **kwargs: TurnPlan(
        action="clarify_then_wait",
        query_route="planning",
        response_language="zh",
        reason="Need clarification for an underspecified planning request.",
        clarification_message="你更想要复习路线图，还是一周日程安排？",
    )
    return service


def build_self_check_with_issue() -> SelfCheckResult:
    return SelfCheckResult(
        issues=[
            SelfCheckIssue(
                code="planning_answer_without_steps",
                severity="medium",
                message="Planning answer should contain steps.",
            )
        ],
        original_answer="测试回答",
        final_answer="测试回答 你如果愿意，我也可以把它进一步拆成 1、2、3 的可执行步骤。",
    )
