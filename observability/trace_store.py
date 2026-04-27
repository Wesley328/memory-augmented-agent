from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .schema import TurnTrace


class TurnTraceStore(ABC):
    def __init__(self) -> None:
        self._lock = threading.RLock()

    @abstractmethod
    def add(self, trace: TurnTrace) -> None:
        raise NotImplementedError

    @abstractmethod
    def list(
        self,
        *,
        query_route: str | None = None,
        planner_action: str | None = None,
        limit: int | None = None,
    ) -> list[TurnTrace]:
        raise NotImplementedError

    @abstractmethod
    def get(self, turn_id: int) -> TurnTrace | None:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class InMemoryTurnTraceStore(TurnTraceStore):
    def __init__(self) -> None:
        super().__init__()
        self._items: list[TurnTrace] = []

    def add(self, trace: TurnTrace) -> None:
        with self._lock:
            self._items = [item for item in self._items if item.turn_id != trace.turn_id]
            self._items.append(trace)
            self._items.sort(key=lambda item: item.turn_id)

    def list(
        self,
        *,
        query_route: str | None = None,
        planner_action: str | None = None,
        limit: int | None = None,
    ) -> list[TurnTrace]:
        with self._lock:
            items = list(self._items)
            if query_route is not None:
                items = [item for item in items if item.query_route == query_route]
            if planner_action is not None:
                items = [
                    item
                    for item in items
                    if str(item.turn_plan.get("action", "")) == planner_action
                ]
            items.sort(key=lambda item: item.turn_id, reverse=True)
            if limit is not None:
                return items[:limit]
            return items

    def get(self, turn_id: int) -> TurnTrace | None:
        with self._lock:
            for item in self._items:
                if item.turn_id == turn_id:
                    return item
            return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class SQLiteTurnTraceStore(TurnTraceStore):
    def __init__(self, db_path: str) -> None:
        super().__init__()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turn_traces (
                    turn_id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    query_route TEXT NOT NULL,
                    planner_action TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_turn_traces_created_at ON turn_traces(created_at)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_turn_traces_query_route ON turn_traces(query_route)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_turn_traces_planner_action ON turn_traces(planner_action)"
            )
            self._conn.commit()

    def add(self, trace: TurnTrace) -> None:
        payload = trace.to_dict()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO turn_traces (
                    turn_id,
                    created_at,
                    query_route,
                    planner_action,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    trace.turn_id,
                    trace.created_at.isoformat(),
                    trace.query_route,
                    str(trace.turn_plan.get("action", "unknown")),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            self._conn.commit()

    def list(
        self,
        *,
        query_route: str | None = None,
        planner_action: str | None = None,
        limit: int | None = None,
    ) -> list[TurnTrace]:
        where_clauses: list[str] = []
        params: list[Any] = []
        if query_route is not None:
            where_clauses.append("query_route = ?")
            params.append(query_route)
        if planner_action is not None:
            where_clauses.append("planner_action = ?")
            params.append(planner_action)

        sql = "SELECT payload_json FROM turn_traces"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += " ORDER BY turn_id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
            return [self._row_to_trace(row) for row in rows]

    def get(self, turn_id: int) -> TurnTrace | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM turn_traces WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_trace(row)

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS count FROM turn_traces").fetchone()
            return int(row["count"]) if row is not None else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _row_to_trace(self, row: sqlite3.Row) -> TurnTrace:
        payload = json.loads(str(row["payload_json"]))
        return TurnTrace.from_dict(payload)


def create_turn_trace_store(
    *,
    backend: str = "memory",
    sqlite_path: str = "data/memory_agent.db",
) -> TurnTraceStore:
    normalized_backend = backend.strip().lower()
    if normalized_backend in {"memory", "in_memory", "in-memory"}:
        return InMemoryTurnTraceStore()
    if normalized_backend == "sqlite":
        return SQLiteTurnTraceStore(db_path=sqlite_path)
    raise ValueError(
        f"Unsupported trace store backend: {backend}. Expected 'memory' or 'sqlite'."
    )
