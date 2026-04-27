from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from agent.planner import TurnPlan
from memory.retriever import ScoredMemory


UNCERTAINTY_MARKERS = {
    "en": {
        "maybe",
        "might",
        "may",
        "perhaps",
        "it seems",
        "i think",
        "i remember",
        "not entirely sure",
        "possibly",
    },
    "zh": {
        "可能",
        "也许",
        "大概",
        "似乎",
        "我记得",
        "如果我没记错",
        "不太确定",
        "也可能",
    },
}


@dataclass(frozen=True)
class SelfCheckIssue:
    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class SelfCheckResult:
    issues: List[SelfCheckIssue]
    original_answer: str
    final_answer: str

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def revised(self) -> bool:
        return self.final_answer != self.original_answer

    @property
    def has_blocking_issue(self) -> bool:
        return any(issue.severity == "high" for issue in self.issues)

    def summary(self) -> str:
        if not self.issues:
            return "passed"
        issue_text = "; ".join(
            f"{issue.severity}:{issue.code}" for issue in self.issues
        )
        revised_text = "yes" if self.revised else "no"
        return f"issues={issue_text} revised={revised_text}"


class ResponseSelfChecker:
    def __init__(self, reliable_confidence_threshold: float = 0.75) -> None:
        self.reliable_confidence_threshold = reliable_confidence_threshold

    def check(
        self,
        *,
        user_query: str,
        answer: str,
        memories: Sequence[ScoredMemory],
        turn_plan: TurnPlan,
        tool_context: str | None = None,
    ) -> SelfCheckResult:
        issues: List[SelfCheckIssue] = []
        normalized_answer = answer.strip()

        issues.extend(
            self._check_empty_answer(
                answer=normalized_answer,
                response_language=turn_plan.response_language,
            )
        )
        issues.extend(
            self._check_tentative_memory_overclaim(
                answer=normalized_answer,
                memories=memories,
                response_language=turn_plan.response_language,
            )
        )
        issues.extend(
            self._check_tool_alignment(
                answer=normalized_answer,
                turn_plan=turn_plan,
                tool_context=tool_context,
            )
        )
        issues.extend(
            self._check_route_alignment(
                answer=normalized_answer,
                turn_plan=turn_plan,
            )
        )
        issues.extend(
            self._check_language_alignment(
                answer=normalized_answer,
                response_language=turn_plan.response_language,
            )
        )

        final_answer = self._revise_answer(
            answer=normalized_answer,
            user_query=user_query,
            issues=issues,
            turn_plan=turn_plan,
            tool_context=tool_context,
        )

        return SelfCheckResult(
            issues=issues,
            original_answer=normalized_answer,
            final_answer=final_answer,
        )

    def _check_empty_answer(
        self, *, answer: str, response_language: str
    ) -> List[SelfCheckIssue]:
        if answer:
            return []
        return [
            SelfCheckIssue(
                code="empty_answer",
                severity="high",
                message=(
                    "The answer is empty and should be replaced with a safe fallback."
                    if response_language == "en"
                    else "当前回答为空，需要替换成安全兜底回复。"
                ),
            )
        ]

    def _check_tentative_memory_overclaim(
        self,
        *,
        answer: str,
        memories: Sequence[ScoredMemory],
        response_language: str,
    ) -> List[SelfCheckIssue]:
        if not answer:
            return []

        tentative_contents = [
            item.memory.content
            for item in memories
            if item.confidence < self.reliable_confidence_threshold
            and item.memory.type != "summary"
        ]
        if not tentative_contents:
            return []

        if self._contains_uncertainty_marker(answer, response_language):
            return []

        for content in tentative_contents:
            if self._has_meaningful_overlap(answer, content):
                return [
                    SelfCheckIssue(
                        code="tentative_memory_overclaim",
                        severity="high",
                        message=(
                            "The answer appears to use low-confidence memory as a certain fact."
                            if response_language == "en"
                            else "回答看起来把低置信度记忆当成了确定事实。"
                        ),
                    )
                ]
        return []

    def _check_tool_alignment(
        self,
        *,
        answer: str,
        turn_plan: TurnPlan,
        tool_context: str | None,
    ) -> List[SelfCheckIssue]:
        if turn_plan.action != "tool_then_answer":
            return []

        if not tool_context:
            return [
                SelfCheckIssue(
                    code="missing_tool_result",
                    severity="high",
                    message="Tool execution was planned, but no tool result is available.",
                )
            ]

        lowered_tool = tool_context.lower()
        if "not integrated yet" in lowered_tool or "empty" in lowered_tool:
            return [
                SelfCheckIssue(
                    code="tool_backend_unavailable",
                    severity="high",
                    message=(
                        "The tool path was selected, but the search backend is not integrated."
                    ),
                )
            ]

        if answer and self._has_meaningful_overlap(answer, tool_context):
            return []

        return [
            SelfCheckIssue(
                code="tool_result_not_reflected",
                severity="medium",
                message="The answer does not seem to reflect the available tool result.",
            )
        ]

    def _check_route_alignment(
        self,
        *,
        answer: str,
        turn_plan: TurnPlan,
    ) -> List[SelfCheckIssue]:
        if not answer:
            return []

        lowered = answer.lower()
        route = turn_plan.query_route
        if route == "planning":
            planning_markers = {
                "step",
                "steps",
                "next",
                "first",
                "then",
                "plan",
                "1.",
                "2.",
                "首先",
                "然后",
                "接着",
                "下一步",
                "计划",
                "步骤",
            }
            if not any(marker in answer or marker in lowered for marker in planning_markers):
                return [
                    SelfCheckIssue(
                        code="planning_answer_without_steps",
                        severity="medium",
                        message="A planning answer should ideally contain concrete steps.",
                    )
                ]
        if route == "profile":
            sentence_count = len(re.findall(r"[.!?。！？]", answer))
            if sentence_count >= 6:
                return [
                    SelfCheckIssue(
                        code="profile_answer_too_scattered",
                        severity="low",
                        message="A profile answer looks too scattered instead of concise.",
                    )
                ]
        return []

    def _check_language_alignment(
        self,
        *,
        answer: str,
        response_language: str,
    ) -> List[SelfCheckIssue]:
        if not answer:
            return []

        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", answer))
        has_ascii_word = bool(re.search(r"[A-Za-z]{2,}", answer))

        if response_language == "zh" and not has_cjk:
            return [
                SelfCheckIssue(
                    code="language_mismatch",
                    severity="medium",
                    message="The answer should have been in Chinese but appears not to be.",
                )
            ]
        if response_language == "en" and not has_ascii_word:
            return [
                SelfCheckIssue(
                    code="language_mismatch",
                    severity="medium",
                    message="The answer should have been in English but appears not to be.",
                )
            ]
        return []

    def _revise_answer(
        self,
        *,
        answer: str,
        user_query: str,
        issues: Sequence[SelfCheckIssue],
        turn_plan: TurnPlan,
        tool_context: str | None,
    ) -> str:
        if not issues:
            return answer

        if self._has_issue(issues, "empty_answer"):
            return self._safe_fallback_answer(turn_plan.response_language)

        if self._has_issue(issues, "tool_backend_unavailable"):
            return self._tool_unavailable_answer(
                user_query=user_query,
                response_language=turn_plan.response_language,
            )

        revised = answer
        if self._has_issue(issues, "tentative_memory_overclaim"):
            revised = self._hedge_answer(
                answer=revised,
                response_language=turn_plan.response_language,
            )

        if self._has_issue(issues, "tool_result_not_reflected") and tool_context:
            revised = self._append_tool_context(
                answer=revised,
                tool_context=tool_context,
                response_language=turn_plan.response_language,
            )

        if self._has_issue(issues, "planning_answer_without_steps"):
            revised = self._append_planning_suffix(
                answer=revised,
                response_language=turn_plan.response_language,
            )

        return revised

    def _safe_fallback_answer(self, response_language: str) -> str:
        if response_language == "zh":
            return "我暂时没能稳定生成回答。你可以换一种说法再问我一次，我会继续帮你。"
        return (
            "I could not produce a stable answer just now. "
            "Please rephrase the question and I will try again."
        )

    def _tool_unavailable_answer(
        self, *, user_query: str, response_language: str
    ) -> str:
        if response_language == "zh":
            return (
                "这个问题更适合先查外部最新信息，但当前项目里的搜索后端还没有真正接入，"
                "所以我现在不能可靠地给出实时结论。"
                f" 你这次的问题是：{user_query}"
            )
        return (
            "This question would be better answered with up-to-date external information, "
            "but the search backend has not been integrated yet, so I cannot give a reliable live answer right now. "
            f" Your query was: {user_query}"
        )

    def _hedge_answer(self, *, answer: str, response_language: str) -> str:
        if response_language == "zh":
            prefix = "这部分我只能做保守判断，如果我没记错的话，"
        else:
            prefix = "I should phrase this more cautiously: "
        if answer.startswith(prefix):
            return answer
        return prefix + answer[:1].lower() + answer[1:] if answer else prefix

    def _append_tool_context(
        self, *, answer: str, tool_context: str, response_language: str
    ) -> str:
        if response_language == "zh":
            suffix = f" 另外，工具侧返回的信息是：{tool_context}"
        else:
            suffix = f" Additionally, the tool result says: {tool_context}"
        if tool_context in answer:
            return answer
        return answer + suffix

    def _append_planning_suffix(
        self, *, answer: str, response_language: str
    ) -> str:
        if response_language == "zh":
            suffix = " 你如果愿意，我也可以把它进一步拆成 1、2、3 的可执行步骤。"
        else:
            suffix = " If useful, I can also break this down into a more explicit step-by-step plan."
        if suffix.strip() in answer:
            return answer
        return answer + suffix

    def _contains_uncertainty_marker(
        self, answer: str, response_language: str
    ) -> bool:
        lowered = answer.lower()
        markers = set(UNCERTAINTY_MARKERS["en"])
        if response_language in {"zh", "mixed"}:
            markers |= UNCERTAINTY_MARKERS["zh"]
        return any(marker in lowered or marker in answer for marker in markers)

    def _has_issue(self, issues: Sequence[SelfCheckIssue], code: str) -> bool:
        return any(issue.code == code for issue in issues)

    def _has_meaningful_overlap(self, left: str, right: str) -> bool:
        left_terms = set(self._extract_terms(left))
        right_terms = set(self._extract_terms(right))
        if not left_terms or not right_terms:
            return False
        overlap = left_terms & right_terms
        return len(overlap) >= 2 or any(len(term) >= 4 for term in overlap)

    def _extract_terms(self, text: str) -> List[str]:
        terms = re.findall(r"[A-Za-z0-9]{3,}|[\u4e00-\u9fff]{2,}", text.lower())
        stop_terms = {
            "assistant",
            "memory",
            "query",
            "用户",
            "问题",
            "回答",
            "信息",
        }
        return [term for term in terms if term not in stop_terms]
