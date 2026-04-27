from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from core.llm import LLMClient
from memory.schema import ALLOWED_MEMORY_TYPES, Memory, infer_topic_key, utcnow

EXTRACTION_PROMPT = """You extract long-term memories from conversation.
Return strict JSON array only. No markdown and no extra words.

Each item must contain:
- content: string
- type: one of ["fact", "preference", "event", "user_profile", "history"]
- importance: float in [0,1]
- confidence: float in [0,1]

Rules:
- Keep only durable information useful in future turns.
- Ignore transient details unless highly important.
- Keep content concise and atomic.

USER_MESSAGE: {user_message}
ASSISTANT_MESSAGE: {assistant_message}
EXTRACT_MEMORY_JSON
"""


class MemoryExtractor:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract(
        self,
        user_message: str,
        assistant_message: str = "",
        turn_id: int | None = None,
        query_route: str = "general",
    ) -> List[Memory]:
        prompt = EXTRACTION_PROMPT.format(
            user_message=user_message.strip(),
            assistant_message=assistant_message.strip(),
        )
        raw_output = self.llm.generate(prompt)
        payload = self._parse_json_array(raw_output)
        results: List[Memory] = []

        for item in payload:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            memory_type = str(item.get("type", "fact")).strip().lower()
            if memory_type not in ALLOWED_MEMORY_TYPES:
                memory_type = "fact"
            importance = self._safe_importance(item.get("importance", 0.5))
            confidence = self._safe_confidence(item.get("confidence", 0.7))
            metadata = self._build_metadata(
                content=content,
                memory_type=memory_type,
                user_message=user_message,
                assistant_message=assistant_message,
                turn_id=turn_id,
                query_route=query_route,
            )
            try:
                results.append(
                    Memory(
                        content=content,
                        type=memory_type,
                        importance=importance,
                        timestamp=utcnow(),
                        confidence=confidence,
                        metadata=metadata,
                    )
                )
            except ValueError:
                continue
        return results

    def _parse_json_array(self, text: str) -> List[dict]:
        text = text.strip()
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, list) else []
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, list) else []
        except json.JSONDecodeError:
            return []

    def _safe_importance(self, value: object) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.5
        return max(0.0, min(1.0, value))

    def _safe_confidence(self, value: object) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.7
        return max(0.0, min(1.0, value))

    def _build_metadata(
        self,
        *,
        content: str,
        memory_type: str,
        user_message: str,
        assistant_message: str,
        turn_id: int | None,
        query_route: str,
    ) -> Dict[str, Any]:
        now = utcnow().isoformat()
        return {
            "status": "active",
            "version": 1,
            "topic_key": infer_topic_key(memory_type, content),
            "source_turn_id": turn_id,
            "source_user_message": user_message.strip(),
            "source_assistant_message": assistant_message.strip(),
            "extracted_at": now,
            "extraction_method": "llm_extractor",
            "query_route_at_extraction": query_route,
        }
