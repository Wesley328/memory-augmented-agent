from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memory.schema import Memory
from memory.store import InMemoryStore
from memory.updater import MemoryUpdater


def test_memory_updater_versions_conflicting_preference_memory() -> None:
    now = datetime.now(timezone.utc)
    store = InMemoryStore(max_size=20)
    updater = MemoryUpdater()

    existing = Memory(
        content="User likes sushi for dinner.",
        type="preference",
        importance=0.8,
        confidence=0.9,
        timestamp=now - timedelta(days=1),
        metadata={
            "memory_id": "pref-v1",
            "lineage_id": "pref-lineage",
            "status": "active",
            "version": 1,
            "topic_key": "preference:sushi",
        },
    )
    store.add(existing)

    incoming = Memory(
        content="User does not like sushi anymore.",
        type="preference",
        importance=0.86,
        confidence=0.88,
        timestamp=now,
        metadata={
            "memory_id": "pref-v2",
            "topic_key": "preference:sushi",
        },
    )

    stats = updater.update(store, [incoming])
    memories = store.all()

    assert stats.versioned == 1
    assert len(memories) == 2

    superseded = next(memory for memory in memories if memory.metadata["memory_id"] == "pref-v1")
    active = next(memory for memory in memories if memory.metadata["memory_id"] == "pref-v2")

    assert superseded.metadata["status"] == "superseded"
    assert superseded.metadata["superseded_by"] == "pref-v2"
    assert active.metadata["status"] == "active"
    assert active.metadata["version"] == 2
    assert active.metadata["lineage_id"] == "pref-lineage"
    assert active.metadata["supersedes"] == "pref-v1"


def test_memory_updater_creates_preference_summary_memory() -> None:
    now = datetime.now(timezone.utc)
    store = InMemoryStore(max_size=20)
    updater = MemoryUpdater()

    first = Memory(
        content="User likes Japanese food.",
        type="preference",
        importance=0.82,
        confidence=0.91,
        timestamp=now - timedelta(hours=3),
        metadata={
            "memory_id": "pref-1",
            "lineage_id": "pref-1",
            "status": "active",
            "version": 1,
            "topic_key": "preference:japanese_food",
        },
    )
    second = Memory(
        content="User prefers concise explanations.",
        type="preference",
        importance=0.79,
        confidence=0.87,
        timestamp=now - timedelta(hours=1),
        metadata={
            "memory_id": "pref-2",
            "lineage_id": "pref-2",
            "status": "active",
            "version": 1,
            "topic_key": "preference:concise_explanations",
        },
    )

    stats = updater.update(store, [first, second])
    memories = store.all()
    summary_memories = [memory for memory in memories if memory.type == "summary"]

    assert stats.added == 2
    assert len(summary_memories) == 1

    summary = summary_memories[0]
    assert summary.metadata["summary_scope"] == "preference_profile"
    assert summary.metadata["summary_kind"] == "preference_rollup"
    assert summary.metadata["summary_source_count"] == 2
    assert set(summary.metadata["summary_source_ids"]) == {"pref-1", "pref-2"}
    assert summary.metadata["status"] == "active"
