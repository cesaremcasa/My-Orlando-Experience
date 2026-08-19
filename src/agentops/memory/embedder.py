from __future__ import annotations

import hashlib
import math
from typing import Any, Protocol

EMBED_DIM = 32
PRODUCTION_MODEL_NAME = "all-MiniLM-L6-v2"
PRODUCTION_EMBED_DIM = 384


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


class FakeEmbedder:
    """Deterministic hash embedder. No model download, no FAISS."""

    dim = EMBED_DIM

    def embed(self, text: str) -> list[float]:
        tokens = [part for part in text.lower().split() if part]
        vector = [0.0] * self.dim
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = digest[0] % self.dim
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            vector[index] += sign * (1.0 + digest[2] / 255.0)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class SentenceTransformerEmbedder:
    """Production embedder. Imports and model load happen on first embed()."""

    dim = PRODUCTION_EMBED_DIM

    def __init__(self, model_name: str = PRODUCTION_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Any = None

    def embed(self, text: str) -> list[float]:
        vector = self._load().encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return [float(value) for value in vector.tolist()]

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                from src.agentops.errors import ConfigurationError

                raise ConfigurationError() from exc
            self._model = SentenceTransformer(self.model_name)
            dimension = getattr(self._model, "get_sentence_embedding_dimension", lambda: self.dim)()
            if dimension:
                self.dim = int(dimension)
        return self._model


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))
