from __future__ import annotations

import re
from typing import Dict, List


def clean_text_keep_lines(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(text.split())


def split_into_segments(text: str) -> List[str]:
    text = clean_text_keep_lines(text)
    raw_parts: List[str] = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if count_words(line) > 80:
            pieces = re.split(r"(?<=[.!?])\s+|;\s+", line)
            raw_parts.extend([p.strip() for p in pieces if p.strip()])
        else:
            raw_parts.append(line)

    segments = []
    for part in raw_parts:
        if count_words(part) >= 3 or re.search(r"\d", part):
            segments.append(part)

    return segments


def chunk_documents(
    docs: List[Dict],
    chunk_size: int = 250,
    overlap: int = 40,
) -> List[Dict]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Dict] = []
    chunk_id = 0

    for doc in docs:
        segments = split_into_segments(doc.get("text", ""))
        if not segments:
            continue

        current: List[str] = []
        current_words = 0

        for seg in segments:
            seg_words = count_words(seg)

            if current and current_words + seg_words > chunk_size:
                chunk_text = "\n".join(current).strip()
                chunks.append(
                    {
                        "id": f"chunk_{chunk_id}",
                        "text": chunk_text,
                        "source": doc.get("source", "unknown"),
                        "page": doc.get("page"),
                        "method": doc.get("method"),
                    }
                )
                chunk_id += 1

                overlap_segments: List[str] = []
                overlap_words = 0
                for old_seg in reversed(current):
                    w = count_words(old_seg)
                    if overlap_words + w > overlap:
                        break
                    overlap_segments.insert(0, old_seg)
                    overlap_words += w

                current = overlap_segments[:]
                current_words = overlap_words

            current.append(seg)
            current_words += seg_words

        if current:
            chunk_text = "\n".join(current).strip()
            chunks.append(
                {
                    "id": f"chunk_{chunk_id}",
                    "text": chunk_text,
                    "source": doc.get("source", "unknown"),
                    "page": doc.get("page"),
                    "method": doc.get("method"),
                }
            )
            chunk_id += 1

    return chunks