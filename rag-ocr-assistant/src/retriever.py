from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np


class VectorStore:
    """Small FAISS vector store for local demos."""

    def __init__(self):
        self.index = None
        self.chunks: List[Dict] = []
        self.dim = None

    def build(self, vectors: np.ndarray, chunks: List[Dict]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")
        if len(chunks) == 0:
            raise ValueError("Cannot build index from empty chunks")

        self.dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self.chunks = chunks

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 6,
        min_score: float = 0.10,
    ) -> List[Dict]:
        if self.index is None:
            raise RuntimeError("Index has not been built yet")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores, ids = self.index.search(query_vector.astype("float32"), top_k)
        results: List[Dict] = []

        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            if float(score) < min_score:
                continue
            item = dict(self.chunks[int(idx)])
            item["score"] = float(score)
            results.append(item)

        return results

    def save(self, folder: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Index has not been built yet")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "index.faiss"))
        with open(folder / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, folder: str | Path) -> None:
        folder = Path(folder)
        self.index = faiss.read_index(str(folder / "index.faiss"))
        with open(folder / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        self.dim = self.index.d
