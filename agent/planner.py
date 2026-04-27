from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from memory.retriever import ScoredMemory


@dataclass(frozen=True)
class MemoryBuckets:
    summary: List[ScoredMemory]
    reliable_atomic: List[ScoredMemory]
    tentative_atomic: List[ScoredMemory]

    @property
    def total(self) -> int:
        return len(self.summary) + len(self.reliable_atomic) + len(self.tentative_atomic)

    @property
    def strong_evidence_count(self) -> int:
        return len(self.summary) + len(self.reliable_atomic)


@dataclass(frozen=True)
class TurnPlan:
    action: str
    query_route: str
    response_language: str
    reason: str
    tool_name: str | None = None
    tool_query: str | None = None
    clarification_message: str | None = None


class PromptPlanner:
    def __init__(self, reliable_confidence_threshold: float = 0.75) -> None:
        self.reliable_confidence_threshold = reliable_confidence_threshold

    def plan_turn(
        self,
        user_query: str,
        memories: List[ScoredMemory],
        query_route: str = "general",
    ) -> TurnPlan:
        effective_route = memories[0].query_route if memories else query_route
        response_language = self._detect_response_language(user_query)
        buckets = self._bucket_memories(memories, effective_route)

        search_query = self._infer_search_query(user_query)
        if search_query is not None:
            return TurnPlan(
                action="tool_then_answer",
                query_route=effective_route,
                response_language=response_language,
                reason="The query appears time-sensitive or explicitly requests external lookup.",
                tool_name="search",
                tool_query=search_query,
            )

        clarification_message = self._clarification_message(
            user_query=user_query,
            query_route=effective_route,
            response_language=response_language,
            buckets=buckets,
        )
        if clarification_message is not None:
            return TurnPlan(
                action="clarify_then_wait",
                query_route=effective_route,
                response_language=response_language,
                reason="The query is ambiguous or under-specified for a high-quality answer.",
                clarification_message=clarification_message,
            )

        return TurnPlan(
            action="direct_answer",
            query_route=effective_route,
            response_language=response_language,
            reason="The current query is specific enough to answer with available memory and local reasoning.",
        )

    def build_prompt(
        self,
        user_query: str,
        memories: List[ScoredMemory],
        query_route: str = "general",
        turn_plan: TurnPlan | None = None,
        tool_context: str | None = None,
    ) -> str:
        effective_route = memories[0].query_route if memories else query_route
        response_language = self._detect_response_language(user_query)
        buckets = self._bucket_memories(memories, effective_route)
        memory_block = self._format_memory_context(buckets, effective_route)
        route_guidance = self._route_guidance(effective_route)
        usage_policy = self._memory_usage_policy(buckets, effective_route)
        response_policy = self._response_policy(response_language, buckets)
        evidence_summary = self._evidence_summary(buckets)
        plan = turn_plan or TurnPlan(
            action="direct_answer",
            query_route=effective_route,
            response_language=response_language,
            reason="No explicit execution plan was provided.",
        )
        turn_plan_block = self._format_turn_plan(plan)
        tool_context_block = self._format_tool_context(plan, tool_context)

        return f"""You are a helpful assistant with persistent long-term memory.
Use memory as reasoning evidence, not as a block of text to quote mechanically.

DETECTED_QUERY_ROUTE: {effective_route}
RESPONSE_LANGUAGE: {response_language}

Difference from static RAG:
- RAG retrieves text snippets.
- This memory system models the user and updates over time.
- Summary memories are for synthesis, atomic memories are for precise details.

TURN_PLAN_START
{turn_plan_block}
TURN_PLAN_END

MEMORY_EVIDENCE_SUMMARY_START
{evidence_summary}
MEMORY_EVIDENCE_SUMMARY_END

MEMORY_USAGE_POLICY_START
{usage_policy}
MEMORY_USAGE_POLICY_END

MEMORY_CONTEXT_START
{memory_block}
MEMORY_CONTEXT_END

TOOL_CONTEXT_START
{tool_context_block}
TOOL_CONTEXT_END

USER_QUERY: {user_query}

RESPONSE_POLICY_START
{response_policy}
RESPONSE_POLICY_END

ROUTE_GUIDANCE_START
{route_guidance}
ROUTE_GUIDANCE_END
"""

    def _bucket_memories(
        self, memories: Sequence[ScoredMemory], query_route: str
    ) -> MemoryBuckets:
        summary: List[ScoredMemory] = []
        reliable_atomic: List[ScoredMemory] = []
        tentative_atomic: List[ScoredMemory] = []

        for item in memories:
            if self._is_summary(item):
                summary.append(item)
                continue
            if item.confidence >= self.reliable_confidence_threshold:
                reliable_atomic.append(item)
            else:
                tentative_atomic.append(item)

        ordered_summary = self._sort_for_prompt(summary, query_route)
        ordered_reliable = self._sort_for_prompt(reliable_atomic, query_route)
        ordered_tentative = self._sort_for_prompt(tentative_atomic, query_route)
        return MemoryBuckets(
            summary=ordered_summary,
            reliable_atomic=ordered_reliable,
            tentative_atomic=ordered_tentative,
        )

    def _sort_for_prompt(
        self, items: Sequence[ScoredMemory], query_route: str
    ) -> List[ScoredMemory]:
        def sort_key(item: ScoredMemory) -> tuple[float, float, float]:
            if query_route == "temporal":
                return (item.recency, item.confidence, item.score)
            if query_route == "planning":
                return (item.importance, item.confidence, item.recency)
            if query_route == "profile":
                return (item.confidence, item.importance, item.score)
            return (item.score, item.confidence, item.importance)

        return sorted(items, key=sort_key, reverse=True)

    def _format_memory_context(self, buckets: MemoryBuckets, query_route: str) -> str:
        sections = []
        preferred_order = self._section_order(query_route)
        section_map = {
            "summary": (
                "SUMMARY_MEMORY",
                "Use these for high-level synthesis and stable pattern description.",
                buckets.summary,
                "summary",
            ),
            "reliable_atomic": (
                "RELIABLE_ATOMIC_MEMORY",
                "Use these for specific facts, user preferences, or concrete timeline details.",
                buckets.reliable_atomic,
                "reliable",
            ),
            "tentative_atomic": (
                "TENTATIVE_MEMORY",
                "Use these only with hedging language or a clarifying follow-up.",
                buckets.tentative_atomic,
                "tentative",
            ),
        }

        for section_key in preferred_order:
            section_title, description, items, evidence_tag = section_map[section_key]
            sections.append(f"[{section_title}]")
            sections.append(f"- note: {description}")
            if not items:
                sections.append("- (empty)")
                continue
            for item in items:
                sections.append(self._format_memory_line(item, evidence_tag))
        return "\n".join(sections)

    def _format_memory_line(self, item: ScoredMemory, evidence_tag: str) -> str:
        memory = item.memory
        metadata = memory.metadata
        status = str(metadata.get("status", "active"))
        topic_key = str(metadata.get("topic_key", ""))
        version = int(metadata.get("version", 1))
        lineage_id = str(metadata.get("lineage_id", ""))[:8]
        detail_bits = []

        if metadata.get("summary_scope"):
            detail_bits.append(f"summary_scope={metadata['summary_scope']}")
        if metadata.get("summary_source_count") is not None:
            detail_bits.append(f"source_count={metadata['summary_source_count']}")
        if metadata.get("supersedes"):
            detail_bits.append("latest_version_of_lineage")
        if metadata.get("query_route_at_extraction"):
            detail_bits.append(
                f"extracted_route={metadata['query_route_at_extraction']}"
            )
        detail_note = ", ".join(detail_bits) if detail_bits else "none"

        return (
            "- content: {content} | type: {type} | evidence: {evidence} "
            "| importance: {importance:.2f} | confidence: {confidence:.2f} "
            "| relevance: {relevance:.2f} | recency: {recency:.2f} "
            "| route: {route} | topic: {topic} | status: {status} "
            "| version: v{version} | lineage: {lineage} | details: {details}"
        ).format(
            content=memory.content,
            type=memory.type,
            evidence=evidence_tag,
            importance=item.importance,
            confidence=item.confidence,
            relevance=item.relevance,
            recency=item.recency,
            route=item.query_route,
            topic=topic_key or "unknown",
            status=status,
            version=version,
            lineage=lineage_id or "n/a",
            details=detail_note,
        )

    def _format_turn_plan(self, turn_plan: TurnPlan) -> str:
        lines = [
            f"- action: {turn_plan.action}",
            f"- reason: {turn_plan.reason}",
            f"- query_route: {turn_plan.query_route}",
            f"- response_language: {turn_plan.response_language}",
        ]
        if turn_plan.tool_name:
            lines.append(f"- tool_name: {turn_plan.tool_name}")
        if turn_plan.tool_query:
            lines.append(f"- tool_query: {turn_plan.tool_query}")
        if turn_plan.action == "tool_then_answer":
            lines.append(
                "- instruction: First interpret the tool result, then combine it with relevant memory."
            )
        else:
            lines.append("- instruction: Answer directly with memory-aware reasoning.")
        return "\n".join(lines)

    def _format_tool_context(
        self, turn_plan: TurnPlan, tool_context: str | None
    ) -> str:
        if turn_plan.action != "tool_then_answer":
            return "- no tool call for this turn"
        if tool_context:
            return (
                f"- tool_name: {turn_plan.tool_name}\n"
                f"- tool_query: {turn_plan.tool_query or ''}\n"
                f"- tool_result: {tool_context}"
            )
        return (
            f"- tool_name: {turn_plan.tool_name}\n"
            f"- tool_query: {turn_plan.tool_query or ''}\n"
            "- tool_result: tool was planned but no result is available"
        )

    def _section_order(self, query_route: str) -> List[str]:
        if query_route in {"profile", "planning"}:
            return ["summary", "reliable_atomic", "tentative_atomic"]
        if query_route == "temporal":
            return ["reliable_atomic", "summary", "tentative_atomic"]
        return ["reliable_atomic", "summary", "tentative_atomic"]

    def _evidence_summary(self, buckets: MemoryBuckets) -> str:
        return (
            f"- total_retrieved: {buckets.total}\n"
            f"- summary_memory_count: {len(buckets.summary)}\n"
            f"- reliable_atomic_count: {len(buckets.reliable_atomic)}\n"
            f"- tentative_memory_count: {len(buckets.tentative_atomic)}"
        )

    def _memory_usage_policy(
        self, buckets: MemoryBuckets, query_route: str
    ) -> str:
        policy_lines = [
            "1) Reliable atomic memories have the highest priority for precise claims.",
            "2) Summary memories are for overview, synthesis, and pattern compression.",
            "3) Tentative memories must never be stated as certain facts.",
            "4) If summary and atomic memory both apply, use summary for framing and atomic memory for detail.",
            "5) If evidence seems incomplete, answer helpfully and ask one clarifying follow-up only when it adds real value.",
        ]

        if buckets.tentative_atomic:
            policy_lines.append(
                "6) There are tentative memories in context, so hedge uncertain statements with language like maybe, it seems, or 我记得/似乎."
            )
        if not buckets.reliable_atomic and not buckets.summary:
            policy_lines.append(
                "6) No strong memory evidence is available; rely on general reasoning instead of inventing personalization."
            )
        if query_route == "temporal":
            policy_lines.append(
                "7) For timeline questions, prefer the most recent and highest-confidence atomic memory."
            )
        if query_route == "profile":
            policy_lines.append(
                "7) For profile questions, compress scattered facts into a concise profile snapshot."
            )
        return "\n".join(f"- {line}" for line in policy_lines)

    def _response_policy(
        self, response_language: str, buckets: MemoryBuckets
    ) -> str:
        language_instruction = {
            "zh": "Reply in Chinese unless the user explicitly asks for another language.",
            "en": "Reply in English unless the user explicitly asks for another language.",
            "mixed": "Mirror the user's dominant language, and keep terminology bilingual only when helpful.",
        }.get(response_language, "Reply in the same language as the user.")

        policy_lines = [
            language_instruction,
            "Keep the answer concise, practical, and naturally personalized.",
            "Do not mention raw memory scores unless the user asks for system details.",
        ]
        if buckets.tentative_atomic:
            policy_lines.append(
                "If you use tentative memory, explicitly signal uncertainty rather than presenting it as settled truth."
            )
        return "\n".join(f"- {line}" for line in policy_lines)

    def _route_guidance(self, query_route: str) -> str:
        guidance_map = {
            "preference": (
                "- Focus on stable user preferences, taste, and personalization.\n"
                "- When making suggestions, connect them explicitly to remembered likes/dislikes.\n"
                "- If preference memory is weak, ask a targeted follow-up instead of pretending strong personalization."
            ),
            "temporal": (
                "- Prioritize time-sensitive memories and present them in a clear timeline.\n"
                "- Be careful with recency language such as recently, yesterday, or last time.\n"
                "- If timing is uncertain, say so briefly instead of inventing chronology."
            ),
            "profile": (
                "- Summarize durable identity facts such as background, role, skills, and profile.\n"
                "- Prefer concise synthesis over listing many scattered details.\n"
                "- Highlight the most decision-relevant profile details first."
            ),
            "planning": (
                "- Use memory to support planning, preparation, and next-step recommendations.\n"
                "- Combine recent events, active constraints, and user profile when suggesting actions.\n"
                "- If context is still missing, ask a minimal clarifying question before overcommitting."
            ),
            "general": (
                "- Use memory when relevant, but do not force it into the answer.\n"
                "- Keep the response practical, natural, and grounded in the current query."
            ),
        }
        return guidance_map.get(query_route, guidance_map["general"])

    def _infer_search_query(self, user_query: str) -> str | None:
        normalized = user_query.strip()
        lowered = normalized.lower()
        explicit_prefixes = ("search ", "lookup ", "look up ", "查一下", "查一查", "搜索", "帮我查")
        if lowered.startswith("search "):
            return normalized[7:].strip() or normalized
        if lowered.startswith("lookup "):
            return normalized[7:].strip() or normalized
        if lowered.startswith("look up "):
            return normalized[8:].strip() or normalized
        for prefix in explicit_prefixes[3:]:
            if normalized.startswith(prefix):
                return normalized[len(prefix) :].strip() or normalized

        search_keywords = {
            "latest",
            "current",
            "today",
            "news",
            "weather",
            "price",
            "stock",
            "real-time",
            "breaking",
            "官网",
            "最新",
            "现在",
            "今天",
            "新闻",
            "天气",
            "价格",
            "股价",
            "实时",
            "汇率",
        }
        if any(keyword in lowered for keyword in search_keywords):
            return normalized
        if any(keyword in normalized for keyword in search_keywords):
            return normalized
        return None

    def _clarification_message(
        self,
        *,
        user_query: str,
        query_route: str,
        response_language: str,
        buckets: MemoryBuckets,
    ) -> str | None:
        normalized = user_query.strip()
        lowered = normalized.lower()
        prefer_zh = response_language == "zh" or (
            response_language == "mixed" and bool(re.search(r"[\u4e00-\u9fff]", normalized))
        )
        unit_count = len(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", normalized))

        if unit_count <= 2:
            if prefer_zh:
                return "你这次的问题还比较短，我还没法准确判断你的目标。你想让我帮你回答什么、推荐什么，还是规划下一步？"
            return (
                "Your query is still too short for a high-quality answer. "
                "Do you want a direct answer, a recommendation, or a planning suggestion?"
            )

        ambiguous_patterns = [
            r"^(这个|那个|它|this|that|it)$",
            r"(这个|那个)(怎么样|如何|咋样)",
            r"^(this|that|it)\s*(one)?\s*(how is it|what about it)?$",
        ]
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in ambiguous_patterns):
            if prefer_zh:
                return "我还不确定你指的是哪一项。你可以把对象说得更具体一点，我再继续帮你判断。"
            return (
                "I am not sure what specific item you mean yet. "
                "Please name the thing more explicitly and I can help from there."
            )

        recommendation_keywords = {
            "recommend",
            "suggest",
            "choose",
            "推荐",
            "建议",
            "选",
            "吃什么",
        }
        if (
            query_route in {"preference", "general"}
            and buckets.strong_evidence_count == 0
            and any(keyword in lowered for keyword in recommendation_keywords)
        ) or (
            query_route in {"preference", "general"}
            and buckets.strong_evidence_count == 0
            and any(keyword in normalized for keyword in recommendation_keywords)
        ):
            if prefer_zh:
                return "这次是偏好类问题，但我手上还没有足够强的偏好记忆。你更在意口味、预算、还是场景？"
            return (
                "This looks like a preference question, but I do not have enough strong preference memory yet. "
                "What matters more here: taste, budget, or context?"
            )

        planning_keywords = {
            "plan",
            "planning",
            "next step",
            "schedule",
            "study plan",
            "计划",
            "准备",
            "规划",
            "安排",
            "下一步",
        }
        if (
            query_route in {"planning", "general"}
            and buckets.strong_evidence_count == 0
            and any(keyword in lowered for keyword in planning_keywords)
        ) or (
            query_route in {"planning", "general"}
            and buckets.strong_evidence_count == 0
            and any(keyword in normalized for keyword in planning_keywords)
        ):
            if prefer_zh:
                return "这是规划类问题，但我还缺少关键上下文。你想规划的目标、时间范围或截止时间是什么？"
            return (
                "This looks like a planning question, but I am missing key context. "
                "What goal, timeframe, or deadline should I plan around?"
            )
        return None

    def _detect_response_language(self, user_query: str) -> str:
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", user_query))
        has_ascii_word = bool(re.search(r"[A-Za-z]{2,}", user_query))
        if has_cjk and has_ascii_word:
            return "mixed"
        if has_cjk:
            return "zh"
        return "en"

    def _is_summary(self, item: ScoredMemory) -> bool:
        return item.memory.type == "summary" or bool(item.memory.metadata.get("is_summary"))
