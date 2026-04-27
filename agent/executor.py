from __future__ import annotations

from core.llm import LLMClient


class AgentExecutor:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def respond(self, prompt: str) -> str:
        return self.llm.generate(prompt).strip()

