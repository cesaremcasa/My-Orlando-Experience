from __future__ import annotations

import hashlib
import math
from typing import Protocol

EMBED_DIM = 32


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


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))
