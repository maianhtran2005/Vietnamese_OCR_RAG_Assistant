from __future__ import annotations

import pickle
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np


STOPWORDS = {
    "ai", "gì", "nào", "ở", "đâu", "khi", "bao", "nhiêu", "là", "có", "không",
    "hãy", "cho", "biết", "nêu", "trình", "bày", "về", "của", "trong", "theo",
    "tài", "liệu", "phần", "mục", "nội", "dung", "được", "và", "các", "những",
    "một", "này", "đó", "với", "đến", "từ"
}


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def normalize(text: str) -> str:
    text = strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def keywords(text: str) -> List[str]:
    words = normalize(text).split()
    return [w for w in words if len(w) >= 2 and w not in STOPWORDS]


def keyword_score(query: str, text: str) -> float:
    q_words = keywords(query)
    if not q_words:
        return 0.0

    text_norm = normalize(text)
    matched = sum(1 for w in q_words if w in text_norm)
    return matched / max(len(q_words), 1)


class VectorStore:
    """Small FAISS vector store with hybrid semantic + keyword reranking."""

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
        self.index = faiss.IndexFlatIP(self.dim)  # cosine similarity because embeddings are normalized
        self.index.add(vectors.astype("float32"))
        self.chunks = chunks

    def search(
        self,
        query_vector: np.ndarray,
        query_text: str = "",
        top_k: int = 6,
        min_score: float = 0.0,
    ) -> List[Dict]:
        if self.index is None:
            raise RuntimeError("Index has not been built yet")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Retrieve more candidates first, then rerank with keyword score.
        candidate_k = min(len(self.chunks), max(top_k * 5, 20))
        scores, ids = self.index.search(query_vector.astype("float32"), candidate_k)

        results: List[Dict] = []
        seen = set()

        for dense_score, idx in zip(scores[0], ids[0]):
            if idx == -1 or int(idx) in seen:
                continue
            seen.add(int(idx))

            chunk = self.chunks[int(idx)]
            k_score = keyword_score(query_text, chunk.get("text", "")) if query_text else 0.0
            final_score = 0.75 * float(dense_score) + 0.25 * k_score

            if final_score < min_score:
                continue

            item = dict(chunk)
            item["score"] = final_score
            item["semantic_score"] = float(dense_score)
            item["keyword_score"] = float(k_score)
            results.append(item)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

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
