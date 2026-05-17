from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def log_qa(question: str, answer: str, contexts: List[Dict], log_path: str = "logs/qa_log.csv") -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    exists = path.exists()
    sources = "; ".join(
        [f"{c.get('source')}|page={c.get('page')}|score={c.get('score', 0):.3f}" for c in contexts]
    )

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "question", "answer", "sources"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "answer": answer,
                "sources": sources,
            }
        )
