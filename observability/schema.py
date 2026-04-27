from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TurnTrace:
    turn_id: int
    created_at: datetime
    query: str
    answer: str
    query_route: str
    turn_plan: dict[str, Any]
    tool_context: str | None
    self_check: dict[str, Any] | None
    retrieved_memories: list[dict[str, Any]]
    memory_lifecycle: dict[str, Any] | None

    def __post_init__(self) -> None:
        created_at = self.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        object.__setattr__(self, "created_at", created_at)

        object.__setattr__(self, "turn_plan", dict(self.turn_plan))
        object.__setattr__(
            self,
            "retrieved_memories",
            [dict(item) for item in self.retrieved_memories],
        )
        object.__setattr__(
            self,
            "self_check",
            dict(self.self_check) if self.self_check is not None else None,
        )
        object.__setattr__(
            self,
            "memory_lifecycle",
            dict(self.memory_lifecycle) if self.memory_lifecycle is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "created_at": self.created_at.isoformat(),
            "query": self.query,
            "answer": self.answer,
            "query_route": self.query_route,
            "turn_plan": dict(self.turn_plan),
            "tool_context": self.tool_context,
            "self_check": dict(self.self_check) if self.self_check is not None else None,
            "retrieved_memories": [dict(item) for item in self.retrieved_memories],
            "memory_lifecycle": (
                dict(self.memory_lifecycle) if self.memory_lifecycle is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TurnTrace":
        created_at = payload.get("created_at")
        if isinstance(created_at, str):
            parsed_created_at = datetime.fromisoformat(created_at)
        elif isinstance(created_at, datetime):
            parsed_created_at = created_at
        else:
            parsed_created_at = datetime.now(timezone.utc)

        return cls(
            turn_id=int(payload["turn_id"]),
            created_at=parsed_created_at,
            query=str(payload.get("query", "")),
            answer=str(payload.get("answer", "")),
            query_route=str(payload.get("query_route", "general")),
            turn_plan=dict(payload.get("turn_plan") or {}),
            tool_context=(
                None
                if payload.get("tool_context") is None
                else str(payload.get("tool_context"))
            ),
            self_check=(
                dict(payload["self_check"]) if isinstance(payload.get("self_check"), dict) else None
            ),
            retrieved_memories=[
                dict(item)
                for item in payload.get("retrieved_memories", [])
                if isinstance(item, dict)
            ],
            memory_lifecycle=(
                dict(payload["memory_lifecycle"])
                if isinstance(payload.get("memory_lifecycle"), dict)
                else None
            ),
        )
