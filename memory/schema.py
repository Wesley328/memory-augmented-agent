from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

BASE_MEMORY_TYPES = {"fact", "preference", "event"}
EXTENDED_MEMORY_TYPES = {"user_profile", "history", "summary"}
ALLOWED_MEMORY_TYPES = BASE_MEMORY_TYPES | EXTENDED_MEMORY_TYPES


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_memory_text(text: str) -> str:
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    return " ".join(tokens)


def infer_topic_key(memory_type: str, content: str) -> str:
    normalized = normalize_memory_text(content)
    if not normalized:
        return f"{memory_type}:unknown"

    preference_patterns = [
        r"user (?:likes|loves|prefers|dislikes|hates|does not like)\s+(.+)",
        r"用户(?:喜欢|偏好|爱|不喜欢|讨厌)\s*(.+)",
    ]
    for pattern in preference_patterns:
        match = re.search(pattern, normalized)
        if match:
            topic = match.group(1).strip()
            topic = re.sub(
                r"\b(?:not|dont|don't|doesnt|doesn't|dislike|dislikes|hate|hates|不|别)\b",
                "",
                topic,
            ).strip()
            return f"{memory_type}:{topic or 'unknown'}"

    name_patterns = [
        r"user name is\s+(.+)",
        r"用户名字是\s*(.+)",
    ]
    for pattern in name_patterns:
        if re.search(pattern, normalized):
            return "fact:name"

    allergy_patterns = [
        r"user is allergic to\s+(.+)",
        r"用户对\s*(.+)\s*过敏",
    ]
    for pattern in allergy_patterns:
        match = re.search(pattern, normalized)
        if match:
            return f"fact:allergy:{match.group(1).strip()}"

    prefix = " ".join(normalized.split()[:6])
    return f"{memory_type}:{prefix}"


@dataclass
class Memory:
    content: str
    type: str
    importance: float
    timestamp: datetime
    confidence: float = 0.7
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_type = self.type.strip().lower()
        if normalized_type not in ALLOWED_MEMORY_TYPES:
            raise ValueError(
                f"Unsupported memory type: {self.type}. "
                f"Expected one of {sorted(ALLOWED_MEMORY_TYPES)}."
            )
        self.type = normalized_type
        self.importance = max(0.0, min(1.0, float(self.importance)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

        self.metadata = dict(self.metadata)
        self.metadata.setdefault("status", "active")
        self.metadata.setdefault("version", 1)
        self.metadata.setdefault("topic_key", infer_topic_key(self.type, self.content))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "type": self.type,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }
