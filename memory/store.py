from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from memory.schema import Memory


class MemoryStore(ABC):
    """
    Abstract store interface for long-term memories.

    The rest of the system depends on this contract so we can swap the
    underlying storage backend without rewriting retriever/updater logic.
    """

    def __init__(self, max_size: int = 500) -> None:
        self.max_size = max_size
        self._lock = threading.RLock()

    @abstractmethod
    def add(self, memory: Memory) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_many(self, memories: List[Memory]) -> None:
        raise NotImplementedError

    @abstractmethod
    def all(self) -> List[Memory]:
        raise NotImplementedError

    @abstractmethod
    def replace_all(self, memories: List[Memory]) -> None:
        raise NotImplementedError

    @abstractmethod
    def recent(self, limit: int = 10) -> List[Memory]:
        raise NotImplementedError

    @abstractmethod
    def set_embedding(self, index: int, embedding: List[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_embedding(self, index: int) -> Optional[List[float]]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def query_memories(
        self,
        *,
        status: str | None = None,
        memory_type: str | None = None,
        lineage_id: str | None = None,
        memory_ids: List[str] | None = None,
        summary_only: bool | None = None,
        limit: int | None = None,
    ) -> List[Memory]:
        memories = self.all()
        if status is not None:
            memories = [
                memory
                for memory in memories
                if str(memory.metadata.get("status", "active")) == status
            ]
        if memory_type is not None:
            normalized_type = memory_type.strip().lower()
            memories = [memory for memory in memories if memory.type == normalized_type]
        if lineage_id is not None:
            memories = [
                memory
                for memory in memories
                if str(memory.metadata.get("lineage_id", "")) == lineage_id
            ]
        if memory_ids is not None:
            wanted_ids = {memory_id for memory_id in memory_ids if memory_id}
            memories = [
                memory
                for memory in memories
                if str(memory.metadata.get("memory_id", "")) in wanted_ids
            ]
        if summary_only is True:
            memories = [
                memory
                for memory in memories
                if memory.type == "summary" or bool(memory.metadata.get("is_summary"))
            ]
        elif summary_only is False:
            memories = [
                memory
                for memory in memories
                if memory.type != "summary" and not bool(memory.metadata.get("is_summary"))
            ]

        memories.sort(key=lambda item: item.timestamp, reverse=True)
        if limit is not None:
            return memories[:limit]
        return memories

    def get_active_memories(self, limit: int | None = None) -> List[Memory]:
        return self.query_memories(status="active", limit=limit)

    def get_lineage(self, lineage_id: str) -> List[Memory]:
        memories = self.query_memories(lineage_id=lineage_id)
        return sorted(
            memories,
            key=lambda item: (
                int(item.metadata.get("version", 1)),
                item.timestamp,
            ),
        )


class InMemoryStore(MemoryStore):
    """
    Minimal in-memory store used for demos, tests, and fast local iteration.
    """

    def __init__(self, max_size: int = 500) -> None:
        super().__init__(max_size=max_size)
        self._memories: List[Memory] = []

    def add(self, memory: Memory) -> None:
        with self._lock:
            self._memories.append(memory)
            if len(self._memories) > self.max_size:
                self._memories.pop(0)

    def add_many(self, memories: List[Memory]) -> None:
        with self._lock:
            for memory in memories:
                self.add(memory)

    def all(self) -> List[Memory]:
        with self._lock:
            return list(self._memories)

    def replace_all(self, memories: List[Memory]) -> None:
        with self._lock:
            self._memories = list(memories)

    def recent(self, limit: int = 10) -> List[Memory]:
        with self._lock:
            return sorted(self._memories, key=lambda item: item.timestamp, reverse=True)[:limit]

    def set_embedding(self, index: int, embedding: List[float]) -> None:
        with self._lock:
            self._memories[index].embedding = embedding

    def get_embedding(self, index: int) -> Optional[List[float]]:
        with self._lock:
            return self._memories[index].embedding

    def __len__(self) -> int:
        with self._lock:
            return len(self._memories)


class SQLiteMemoryStore(MemoryStore):
    """
    Persistent SQLite-backed store.

    This is the first storage upgrade stage: preserve the same store contract
    while moving memory state into a local database file.
    """

    def __init__(self, db_path: str, max_size: int = 500) -> None:
        super().__init__(max_size=max_size)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    memory_id TEXT,
                    lineage_id TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    version INTEGER NOT NULL DEFAULT 1,
                    topic_key TEXT,
                    summary_scope TEXT,
                    is_summary INTEGER NOT NULL DEFAULT 0,
                    embedding_json TEXT,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            self._ensure_column("memory_id", "TEXT")
            self._ensure_column("lineage_id", "TEXT")
            self._ensure_column("status", "TEXT NOT NULL DEFAULT 'active'")
            self._ensure_column("version", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column("topic_key", "TEXT")
            self._ensure_column("summary_scope", "TEXT")
            self._ensure_column("is_summary", "INTEGER NOT NULL DEFAULT 0")
            self._create_indexes()
            self._conn.commit()

    def add(self, memory: Memory) -> None:
        with self._lock:
            self._insert_memory(memory)
            self._trim_to_max_size()

    def add_many(self, memories: List[Memory]) -> None:
        with self._lock:
            with self._conn:
                for memory in memories:
                    self._insert_memory(memory, commit=False)
            self._trim_to_max_size()

    def all(self) -> List[Memory]:
        with self._lock:
            rows = self._conn.execute(self._base_select_sql() + " ORDER BY id ASC").fetchall()
            return [self._row_to_memory(row) for row in rows]

    def replace_all(self, memories: List[Memory]) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM memories")
                for memory in memories:
                    self._insert_memory(memory, commit=False)
            self._trim_to_max_size()

    def recent(self, limit: int = 10) -> List[Memory]:
        with self._lock:
            rows = self._conn.execute(
                self._base_select_sql() + " ORDER BY timestamp DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_memory(row) for row in rows]

    def query_memories(
        self,
        *,
        status: str | None = None,
        memory_type: str | None = None,
        lineage_id: str | None = None,
        memory_ids: List[str] | None = None,
        summary_only: bool | None = None,
        limit: int | None = None,
    ) -> List[Memory]:
        where_clauses: List[str] = []
        params: List[Any] = []

        if status is not None:
            where_clauses.append("status = ?")
            params.append(status)
        if memory_type is not None:
            where_clauses.append("type = ?")
            params.append(memory_type.strip().lower())
        if lineage_id is not None:
            where_clauses.append("lineage_id = ?")
            params.append(lineage_id)
        if memory_ids is not None:
            wanted_ids = [memory_id for memory_id in memory_ids if memory_id]
            if not wanted_ids:
                return []
            placeholders = ", ".join("?" for _ in wanted_ids)
            where_clauses.append(f"memory_id IN ({placeholders})")
            params.extend(wanted_ids)
        if summary_only is True:
            where_clauses.append("is_summary = 1")
        elif summary_only is False:
            where_clauses.append("is_summary = 0")

        sql = self._base_select_sql()
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += " ORDER BY timestamp DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
            return [self._row_to_memory(row) for row in rows]

    def get_active_memories(self, limit: int | None = None) -> List[Memory]:
        with self._lock:
            return self.query_memories(status="active", limit=limit)

    def get_lineage(self, lineage_id: str) -> List[Memory]:
        with self._lock:
            rows = self._conn.execute(
                self._base_select_sql()
                + " WHERE lineage_id = ? ORDER BY version ASC, timestamp ASC, id ASC",
                (lineage_id,),
            ).fetchall()
            return [self._row_to_memory(row) for row in rows]

    def set_embedding(self, index: int, embedding: List[float]) -> None:
        with self._lock:
            row_id = self._row_id_for_index(index)
            if row_id is None:
                raise IndexError("memory index out of range")
            self._conn.execute(
                "UPDATE memories SET embedding_json = ? WHERE id = ?",
                (json.dumps(embedding), row_id),
            )
            self._conn.commit()

    def get_embedding(self, index: int) -> Optional[List[float]]:
        with self._lock:
            row_id = self._row_id_for_index(index)
            if row_id is None:
                raise IndexError("memory index out of range")
            row = self._conn.execute(
                "SELECT embedding_json FROM memories WHERE id = ?",
                (row_id,),
            ).fetchone()
            if row is None or row["embedding_json"] is None:
                return None
            payload = json.loads(row["embedding_json"])
            return [float(value) for value in payload]

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS count FROM memories").fetchone()
            return int(row["count"]) if row is not None else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _insert_memory(self, memory: Memory, commit: bool = True) -> None:
        self._conn.execute(
            """
            INSERT INTO memories (
                content, type, importance, confidence, timestamp,
                memory_id, lineage_id, status, version, topic_key, summary_scope, is_summary,
                embedding_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.content,
                memory.type,
                memory.importance,
                memory.confidence,
                memory.timestamp.isoformat(),
                memory.metadata.get("memory_id"),
                memory.metadata.get("lineage_id"),
                str(memory.metadata.get("status", "active")),
                int(memory.metadata.get("version", 1)),
                memory.metadata.get("topic_key"),
                memory.metadata.get("summary_scope"),
                1 if (memory.type == "summary" or bool(memory.metadata.get("is_summary"))) else 0,
                json.dumps(memory.embedding) if memory.embedding is not None else None,
                json.dumps(memory.metadata, ensure_ascii=False),
            ),
        )
        if commit:
            self._conn.commit()

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        embedding_json = row["embedding_json"]
        metadata_json = row["metadata_json"]
        embedding = None
        if embedding_json:
            payload = json.loads(embedding_json)
            embedding = [float(value) for value in payload]

        metadata = json.loads(metadata_json) if metadata_json else {}
        if row["memory_id"] is not None:
            metadata["memory_id"] = row["memory_id"]
        if row["lineage_id"] is not None:
            metadata["lineage_id"] = row["lineage_id"]
        metadata["status"] = row["status"]
        metadata["version"] = int(row["version"])
        if row["topic_key"] is not None:
            metadata["topic_key"] = row["topic_key"]
        if row["summary_scope"] is not None:
            metadata["summary_scope"] = row["summary_scope"]
        metadata["is_summary"] = bool(row["is_summary"])
        return Memory(
            content=str(row["content"]),
            type=str(row["type"]),
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            timestamp=datetime.fromisoformat(str(row["timestamp"])),
            embedding=embedding,
            metadata=metadata,
        )

    def _row_id_for_index(self, index: int) -> Optional[int]:
        row = self._conn.execute(
            "SELECT id FROM memories ORDER BY id ASC LIMIT 1 OFFSET ?",
            (index,),
        ).fetchone()
        return int(row["id"]) if row is not None else None

    def _trim_to_max_size(self) -> None:
        if self.max_size <= 0:
            return
        overflow = len(self) - self.max_size
        if overflow <= 0:
            return
        self._conn.execute(
            """
            DELETE FROM memories
            WHERE id IN (
                SELECT id FROM memories
                ORDER BY id ASC
                LIMIT ?
            )
            """,
            (overflow,),
        )
        self._conn.commit()

    def _base_select_sql(self) -> str:
        return """
            SELECT
                content,
                type,
                importance,
                confidence,
                timestamp,
                memory_id,
                lineage_id,
                status,
                version,
                topic_key,
                summary_scope,
                is_summary,
                embedding_json,
                metadata_json
            FROM memories
        """

    def _ensure_column(self, column_name: str, ddl: str) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        if column_name in columns:
            return
        self._conn.execute(f"ALTER TABLE memories ADD COLUMN {column_name} {ddl}")

    def _create_indexes(self) -> None:
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_lineage_id ON memories(lineage_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_topic_key ON memories(topic_key)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_summary_scope ON memories(summary_scope)"
        )


def create_memory_store(
    *,
    backend: str = "memory",
    max_size: int = 500,
    sqlite_path: str = "data/memory_agent.db",
) -> MemoryStore:
    normalized_backend = backend.strip().lower()
    if normalized_backend in {"memory", "in_memory", "in-memory"}:
        return InMemoryStore(max_size=max_size)
    if normalized_backend == "sqlite":
        return SQLiteMemoryStore(db_path=sqlite_path, max_size=max_size)
    raise ValueError(
        f"Unsupported memory store backend: {backend}. Expected 'memory' or 'sqlite'."
    )
