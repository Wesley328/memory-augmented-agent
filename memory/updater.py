from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from difflib import SequenceMatcher
from uuid import uuid4
from typing import Dict, List, Sequence

from memory.schema import Memory, infer_topic_key, normalize_memory_text, utcnow
from memory.store import MemoryStore


@dataclass
class UpdateStats:
    added: int
    updated: int
    versioned: int
    removed: int


class MemoryUpdater:
    def __init__(
        self,
        min_importance: float = 0.15,
        max_memories: int = 200,
        merge_similarity: float = 0.92,
        summary_window_days: int = 14,
        summary_max_sources: int = 5,
    ) -> None:
        self.min_importance = min_importance
        self.max_memories = max_memories
        self.merge_similarity = merge_similarity
        self.summary_window_days = summary_window_days
        self.summary_max_sources = summary_max_sources

    def update(self, store: MemoryStore, new_memories: List[Memory]) -> UpdateStats:
        existing = store.all()
        added = 0
        updated = 0
        versioned = 0

        for incoming in new_memories:
            self._ensure_identity(incoming)
            target = self._find_duplicate(existing, incoming)
            if target is None:
                conflict_target = self._find_conflict(existing, incoming)
                if conflict_target is not None:
                    self._version_conflict(existing, conflict_target, incoming)
                    versioned += 1
                    continue

                existing.append(incoming)
                added += 1
                continue

            self._merge_duplicate(target, incoming)
            updated += 1

        self._refresh_summaries(existing)

        before_compress = len(existing)
        compressed = self._compress(existing)
        after_compress = len(compressed)

        if len(compressed) > self.max_memories:
            compressed.sort(
                key=lambda item: (item.importance, item.timestamp),
                reverse=True,
            )
            compressed = compressed[: self.max_memories]

        store.replace_all(compressed)
        removed = max(0, before_compress - after_compress)
        return UpdateStats(
            added=added,
            updated=updated,
            versioned=versioned,
            removed=removed,
        )

    def _find_duplicate(self, existing: List[Memory], incoming: Memory) -> Memory | None:
        normalized = incoming.content.strip().lower()
        for memory in existing:
            if not self._is_active(memory):
                continue
            if self._is_summary(memory) or self._is_summary(incoming):
                continue
            if memory.type != incoming.type:
                continue
            candidate = memory.content.strip().lower()
            similarity = SequenceMatcher(a=normalized, b=candidate).ratio()
            if similarity >= self.merge_similarity:
                return memory
        return None

    def _find_conflict(self, existing: List[Memory], incoming: Memory) -> Memory | None:
        incoming_topic_key = self._topic_key(incoming)
        for memory in existing:
            if not self._is_active(memory):
                continue
            if self._is_summary(memory):
                continue
            if memory.type != incoming.type:
                continue
            if self._topic_key(memory) != incoming_topic_key:
                continue
            if self._is_conflicting(memory, incoming):
                return memory
        return None

    def _compress(self, memories: List[Memory]) -> List[Memory]:
        now = utcnow()
        cutoff = now - timedelta(days=7)
        filtered = [
            memory
            for memory in memories
            if not self._should_prune(memory, cutoff)
        ]

        merged: List[Memory] = []
        for memory in filtered:
            if not self._is_active(memory):
                merged.append(memory)
                continue
            if self._is_summary(memory):
                merged.append(memory)
                continue
            target = self._find_duplicate(merged, memory)
            if target is None:
                merged.append(memory)
                continue
            self._merge_duplicate(target, memory)
        return merged

    def _merge_duplicate(self, target: Memory, incoming: Memory) -> None:
        target.importance = max(target.importance, incoming.importance)
        target.confidence = max(target.confidence, incoming.confidence)
        if incoming.timestamp > target.timestamp:
            target.timestamp = incoming.timestamp
        if incoming.embedding:
            target.embedding = incoming.embedding
        target.metadata["source_turn_id"] = incoming.metadata.get(
            "source_turn_id", target.metadata.get("source_turn_id")
        )
        target.metadata["source_user_message"] = incoming.metadata.get(
            "source_user_message", target.metadata.get("source_user_message")
        )
        target.metadata["source_assistant_message"] = incoming.metadata.get(
            "source_assistant_message", target.metadata.get("source_assistant_message")
        )
        target.metadata["last_updated_at"] = utcnow().isoformat()

    def _ensure_identity(self, memory: Memory) -> None:
        memory.metadata.setdefault("memory_id", str(uuid4()))
        memory.metadata.setdefault("lineage_id", memory.metadata["memory_id"])
        memory.metadata.setdefault("version", 1)
        memory.metadata.setdefault("status", "active")
        memory.metadata.setdefault("topic_key", infer_topic_key(memory.type, memory.content))

    def _topic_key(self, memory: Memory) -> str:
        return str(memory.metadata.get("topic_key") or infer_topic_key(memory.type, memory.content))

    def _is_active(self, memory: Memory) -> bool:
        return str(memory.metadata.get("status", "active")) == "active"

    def _is_summary(self, memory: Memory) -> bool:
        return memory.type == "summary" or bool(memory.metadata.get("is_summary"))

    def _should_prune(self, memory: Memory, cutoff) -> bool:
        status = str(memory.metadata.get("status", "active"))
        if status == "superseded":
            old_superseded_cutoff = utcnow() - timedelta(days=30)
            return memory.timestamp < old_superseded_cutoff
        return memory.importance < self.min_importance and memory.timestamp < cutoff

    def _refresh_summaries(self, existing: List[Memory]) -> None:
        desired = self._build_desired_summaries(existing)
        active_summaries = {
            str(memory.metadata.get("summary_scope")): memory
            for memory in existing
            if self._is_summary(memory) and self._is_active(memory)
        }

        for scope, summary_spec in desired.items():
            current = active_summaries.get(scope)
            if current is not None and self._summary_matches(current, summary_spec):
                continue

            latest = self._latest_summary_for_scope(existing, scope)
            version = int(latest.metadata.get("version", 0)) + 1 if latest else 1
            lineage_id = (
                latest.metadata.get("lineage_id")
                if latest is not None
                else str(uuid4())
            )

            new_summary = Memory(
                content=summary_spec["content"],
                type="summary",
                importance=summary_spec["importance"],
                confidence=summary_spec["confidence"],
                timestamp=utcnow(),
                metadata={
                    "status": "active",
                    "version": version,
                    "memory_id": str(uuid4()),
                    "lineage_id": lineage_id,
                    "topic_key": f"summary:{scope}",
                    "summary_scope": scope,
                    "summary_kind": summary_spec["kind"],
                    "summary_source_ids": summary_spec["source_ids"],
                    "summary_source_count": len(summary_spec["source_ids"]),
                    "is_summary": True,
                    "derived_from": "memory_updater",
                },
            )

            if current is not None:
                current.metadata["status"] = "superseded"
                current.metadata["superseded_by"] = new_summary.metadata["memory_id"]
                current.metadata["superseded_at"] = utcnow().isoformat()
                new_summary.metadata["supersedes"] = current.metadata["memory_id"]

            existing.append(new_summary)

        for scope, current in active_summaries.items():
            if scope in desired:
                continue
            current.metadata["status"] = "superseded"
            current.metadata["superseded_at"] = utcnow().isoformat()
            current.metadata["version_reason"] = "summary_stale"

    def _build_desired_summaries(self, memories: Sequence[Memory]) -> Dict[str, Dict[str, object]]:
        now = utcnow()
        recent_cutoff = now - timedelta(days=self.summary_window_days)
        active_base = [
            memory
            for memory in memories
            if self._is_active(memory) and not self._is_summary(memory)
        ]
        desired: Dict[str, Dict[str, object]] = {}

        recent_activity = [
            memory
            for memory in active_base
            if memory.type in {"event", "history"} and memory.timestamp >= recent_cutoff
        ]
        if len(recent_activity) >= 2:
            recent_activity.sort(
                key=lambda item: (item.timestamp, item.importance, item.confidence),
                reverse=True,
            )
            selected = recent_activity[: self.summary_max_sources]
            desired["recent_activity"] = self._make_summary_spec(
                scope="recent_activity",
                kind="activity_rollup",
                prefix="Recent activity summary",
                memories=selected,
            )

        preference_memories = [memory for memory in active_base if memory.type == "preference"]
        if len(preference_memories) >= 2:
            preference_memories.sort(
                key=lambda item: (item.importance, item.confidence, item.timestamp),
                reverse=True,
            )
            selected = self._dedupe_by_topic(preference_memories)[: self.summary_max_sources]
            if len(selected) >= 2:
                desired["preference_profile"] = self._make_summary_spec(
                    scope="preference_profile",
                    kind="preference_rollup",
                    prefix="Preference summary",
                    memories=selected,
                )

        profile_memories = [
            memory for memory in active_base if memory.type in {"user_profile", "fact"}
        ]
        if len(profile_memories) >= 2:
            profile_memories.sort(
                key=lambda item: (item.importance, item.confidence, item.timestamp),
                reverse=True,
            )
            selected = self._dedupe_by_topic(profile_memories)[: self.summary_max_sources]
            if len(selected) >= 2:
                desired["profile_snapshot"] = self._make_summary_spec(
                    scope="profile_snapshot",
                    kind="profile_rollup",
                    prefix="Profile summary",
                    memories=selected,
                )

        return desired

    def _dedupe_by_topic(self, memories: Sequence[Memory]) -> List[Memory]:
        seen_topics = set()
        selected: List[Memory] = []
        for memory in memories:
            topic_key = self._topic_key(memory)
            if topic_key in seen_topics:
                continue
            seen_topics.add(topic_key)
            selected.append(memory)
        return selected

    def _make_summary_spec(
        self,
        *,
        scope: str,
        kind: str,
        prefix: str,
        memories: Sequence[Memory],
    ) -> Dict[str, object]:
        source_ids = [
            str(memory.metadata.get("memory_id"))
            for memory in memories
            if memory.metadata.get("memory_id")
        ]
        snippets = [memory.content.strip() for memory in memories if memory.content.strip()]
        content = f"{prefix}: " + "; ".join(snippets)
        avg_confidence = sum(memory.confidence for memory in memories) / len(memories)
        avg_importance = sum(memory.importance for memory in memories) / len(memories)
        return {
            "scope": scope,
            "kind": kind,
            "content": content,
            "source_ids": source_ids,
            "confidence": max(0.5, min(0.95, avg_confidence)),
            "importance": max(0.5, min(0.98, avg_importance + 0.05)),
        }

    def _summary_matches(self, memory: Memory, summary_spec: Dict[str, object]) -> bool:
        existing_ids = list(memory.metadata.get("summary_source_ids", []))
        desired_ids = list(summary_spec.get("source_ids", []))
        return memory.content == summary_spec.get("content") and existing_ids == desired_ids

    def _latest_summary_for_scope(
        self, memories: Sequence[Memory], scope: str
    ) -> Memory | None:
        candidates = [
            memory
            for memory in memories
            if self._is_summary(memory) and str(memory.metadata.get("summary_scope")) == scope
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (int(item.metadata.get("version", 1)), item.timestamp),
            reverse=True,
        )
        return candidates[0]

    def _version_conflict(
        self, existing: List[Memory], current: Memory, incoming: Memory
    ) -> None:
        self._ensure_identity(current)
        self._ensure_identity(incoming)
        incoming.metadata["lineage_id"] = current.metadata["lineage_id"]
        incoming.metadata["version"] = int(current.metadata.get("version", 1)) + 1
        incoming.metadata["status"] = "active"
        incoming.metadata["supersedes"] = current.metadata["memory_id"]
        incoming.metadata["version_reason"] = "conflict_update"

        current.metadata["status"] = "superseded"
        current.metadata["superseded_by"] = incoming.metadata["memory_id"]
        current.metadata["superseded_at"] = utcnow().isoformat()
        existing.append(incoming)

    def _is_conflicting(self, left: Memory, right: Memory) -> bool:
        if left.type == "preference":
            left_polarity = self._preference_polarity(left.content)
            right_polarity = self._preference_polarity(right.content)
            if left_polarity != 0 and right_polarity != 0 and left_polarity != right_polarity:
                return True

        if left.type in {"fact", "user_profile"}:
            left_norm = normalize_memory_text(left.content)
            right_norm = normalize_memory_text(right.content)
            similarity = SequenceMatcher(a=left_norm, b=right_norm).ratio()
            return similarity < self.merge_similarity

        return False

    def _preference_polarity(self, content: str) -> int:
        normalized = normalize_memory_text(content)
        negative_markers = [
            "does not like",
            "do not like",
            "dislike",
            "dislikes",
            "hate",
            "hates",
            "不喜欢",
            "讨厌",
        ]
        positive_markers = [
            "like",
            "likes",
            "love",
            "loves",
            "prefer",
            "prefers",
            "喜欢",
            "偏好",
            "爱",
        ]
        if any(marker in normalized for marker in negative_markers):
            return -1
        if any(marker in normalized for marker in positive_markers):
            return 1
        return 0
