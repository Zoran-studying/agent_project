from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class HashEmbeddings(Embeddings):
    """Small deterministic embedding used only by automated tests."""

    dimensions = 48

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token in text:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dimensions
            values[index] += 1.0
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture()
def hash_embeddings() -> HashEmbeddings:
    return HashEmbeddings()

