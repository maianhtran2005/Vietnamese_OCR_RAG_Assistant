from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from src.chunker import chunk_documents
from src.embeddings import EmbeddingModel
from src.generator import generate_answer
from src.loader import load_document
from src.retriever import VectorStore


def load_uploaded_document(
    file_path: str | Path,
    use_ocr_if_needed: bool,
    ocr_lang: str,
) -> List[Dict]:
    return load_document(
        file_path,
        use_ocr_if_needed=use_ocr_if_needed,
        ocr_lang=ocr_lang,
    )


def build_index_from_documents(
    docs: List[Dict],
    chunk_size: int,
    overlap: int,
) -> Tuple[List[Dict], EmbeddingModel, VectorStore]:
    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        raise ValueError("Không tạo được chunk từ tài liệu.")

    embedder = EmbeddingModel()
    vectors = embedder.encode([chunk["text"] for chunk in chunks])

    store = VectorStore()
    store.build(vectors, chunks)

    return chunks, embedder, store


def retrieve_contexts(
    store: VectorStore,
    embedder: EmbeddingModel,
    question: str,
    top_k: int,
    min_score: float,
) -> List[Dict]:
    query_vector = embedder.encode([question])
    return store.search(
        query_vector,
        query_text=question,
        top_k=top_k,
        min_score=min_score,
    )


def run_answer_generation(
    question: str,
    contexts: List[Dict],
    answer_mode: str,
) -> str:
    return generate_answer(question, contexts, answer_mode=answer_mode)
