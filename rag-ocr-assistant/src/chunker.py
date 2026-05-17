from __future__ import annotations

import re
from typing import Dict, List


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_documents(
    docs: List[Dict],
    chunk_size: int = 450,
    overlap: int = 80,
) -> List[Dict]:
    """Split documents into overlapping word chunks.

    Word-based chunking is simple and works well enough for a first RAG project.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Dict] = []
    chunk_id = 0

    for doc in docs:
        words = normalize_text(doc["text"]).split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            chunks.append(
                {
                    "id": f"chunk_{chunk_id}",
                    "text": chunk_text,
                    "source": doc.get("source", "unknown"),
                    "page": doc.get("page"),
                    "method": doc.get("method"),
                    "word_start": start,
                    "word_end": end,
                }
            )
            chunk_id += 1

            if end == len(words):
                break
            start = end - overlap

    return chunks
