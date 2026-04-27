from __future__ import annotations

import math
import re
from typing import List, Optional


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class ExternalEmbedder:
    """
    API-based embedder with a deterministic local fallback.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        fallback_dim: int = 64,
    ) -> None:
        self.model = model
        self.fallback_dim = fallback_dim
        self._client = None
        try:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(base_url=base_url, api_key=api_key)
        except Exception:
            self._client = None

    def embed(self, text: str) -> List[float]:
        cleaned = text.strip()
        if not cleaned:
            return [0.0] * self.fallback_dim

        if self._client is None:
            return self._fallback_embed(cleaned)

        try:
            response = self._client.embeddings.create(model=self.model, input=cleaned)
            data = getattr(response, "data", None) or []
            if data:
                vector = getattr(data[0], "embedding", None)
                if isinstance(vector, list) and vector:
                    return [float(x) for x in vector]
        except Exception:
            pass
        return self._fallback_embed(cleaned)

    def _fallback_embed(self, text: str) -> List[float]:
        vec = [0.0] * self.fallback_dim
        tokens = _tokenize(text)
        if not tokens:
            return vec

        for token in tokens:
            idx = hash(token) % self.fallback_dim
            vec[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]
