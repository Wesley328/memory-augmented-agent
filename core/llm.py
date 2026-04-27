from __future__ import annotations

from typing import List

from core.config import Settings


class LLMError(RuntimeError):
    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class LLMConfigurationError(LLMError):
    pass


class LLMRequestError(LLMError):
    pass


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        self._configuration_issue = self._detect_configuration_issue()
        if self._configuration_issue is not None:
            return

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            self._configuration_issue = (
                "LLM 依赖未安装，无法初始化模型客户端。"
                "请先安装 `openai` 依赖，并检查 `.env` 中的 "
                "`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。"
                f" 底层错误: {exc}"
            )
            return

        try:
            self._client = OpenAI(
                base_url=self.settings.openai_base_url,
                api_key=self.settings.openai_api_key,
            )
        except Exception as exc:
            self._configuration_issue = (
                "LLM 客户端初始化失败。请检查 `.env` 中的 "
                "`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` 是否有效。"
                f" 底层错误: {exc}"
            )

    @property
    def is_ready(self) -> bool:
        return self._client is not None and self._configuration_issue is None

    @property
    def configuration_issue(self) -> str | None:
        return self._configuration_issue

    def generate(self, prompt: str) -> str:
        self._ensure_ready()

        try:
            chat_text = self._generate_with_chat(prompt)
        except Exception as exc:
            raise LLMRequestError(
                "LLM 调用失败。请检查 `.env` 中的 "
                "`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` 是否正确，"
                "并确认模型服务当前可用。"
                f" 底层错误: {exc}"
            ) from exc

        if chat_text:
            return chat_text

        raise LLMRequestError(
            "LLM 返回为空，无法继续生成回答。请检查模型服务是否可用，"
            "以及 `.env` 中的模型配置是否正确。"
        )

    def _ensure_ready(self) -> None:
        if self._configuration_issue is not None:
            raise LLMConfigurationError(self._configuration_issue)
        if self._client is None:
            raise LLMConfigurationError(
                "LLM 尚未成功初始化。请检查 `.env` 中的 "
                "`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。"
            )

    def _detect_configuration_issue(self) -> str | None:
        missing_fields: List[str] = []
        if not (self.settings.openai_api_key or "").strip():
            missing_fields.append("OPENAI_API_KEY")
        if not self.settings.openai_base_url.strip():
            missing_fields.append("OPENAI_BASE_URL")
        if not self.settings.model.strip():
            missing_fields.append("OPENAI_MODEL")

        if not missing_fields:
            return None

        missing_text = ", ".join(missing_fields)
        return (
            "LLM 尚未正确配置，当前缺少以下配置项: "
            f"{missing_text}。请在 `.env` 中补全后重新运行。"
        )

    def _generate_with_chat(self, prompt: str) -> str | None:
        completion = self._client.chat.completions.create(
            model=self.settings.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.settings.temperature,
        )
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return None

        message = getattr(choices[0], "message", None)
        if message is None:
            return None

        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            merged = "\n".join(parts).strip()
            return merged or None
        return None
