from __future__ import annotations

import os
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """SentenceTransformer wrapper for Vietnamese/multilingual semantic search."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")
