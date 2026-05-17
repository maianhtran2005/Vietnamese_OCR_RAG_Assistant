from __future__ import annotations

import os
import re
from typing import Dict, List


VI_STOPWORDS = {
    "là", "và", "của", "có", "cho", "trong", "với", "các", "những",
    "một", "được", "về", "đến", "từ", "theo", "này", "đó", "ai",
    "gì", "nào", "hãy", "cho", "biết", "nêu", "trình", "bày"
}


def format_sources(contexts: List[Dict]) -> str:
    lines = []
    for i, ctx in enumerate(contexts, start=1):
        lines.append(
            f"[Nguồn {i}] File: {ctx.get('source')} | Trang: {ctx.get('page')} | "
            f"Phương thức: {ctx.get('method')} | Điểm liên quan: {ctx.get('score', 0):.3f}\n"
            f"{ctx.get('text')}"
        )
    return "\n\n".join(lines)


def build_prompt(question: str, contexts: List[Dict]) -> str:
    return f"""
Bạn là trợ lý hỏi đáp tài liệu tiếng Việt cho chương trình, kế hoạch, báo cáo và phân công nhân sự sinh viên.

Nhiệm vụ của bạn:
- Chỉ trả lời đúng nội dung được hỏi.
- Không chép lại toàn bộ đoạn nguồn.
- Không liệt kê lan man các nội dung không liên quan.
- Mỗi ý trả lời phải có trích dẫn nguồn dạng [Nguồn 1], [Nguồn 2].
- Nếu ngữ cảnh không đủ căn cứ, hãy trả lời đúng câu:
"Tôi chưa tìm thấy dữ liệu đủ căn cứ trong tài liệu đã tải lên."

CÂU HỎI:
{question}

NGỮ CẢNH:
{format_sources(contexts)}

YÊU CẦU TRẢ LỜI:
- Trả lời ngắn gọn, trực tiếp vào câu hỏi.
- Ưu tiên trích xuất chính xác tên người, thời gian, nhiệm vụ, địa điểm, số liệu nếu được hỏi.
- Không đưa nguyên văn toàn bộ chunk.
- Không bịa thông tin ngoài ngữ cảnh.
""".strip()


def generate_answer(question: str, contexts: List[Dict]) -> str:
    if not contexts:
        return "Tôi chưa tìm thấy dữ liệu đủ căn cứ trong tài liệu đã tải lên."

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=model,
                input=build_prompt(question, contexts),
                temperature=0.2,
            )
            return response.output_text.strip()
        except Exception as exc:
            return (
                "Không gọi được LLM qua API. Hệ thống tạm trích xuất các câu liên quan nhất từ tài liệu.\n\n"
                f"Chi tiết lỗi: {exc}\n\n"
                + extractive_answer(question, contexts)
            )

    return extractive_answer(question, contexts)


def normalize_for_match(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    words = text.split()
    return [w for w in words if w not in VI_STOPWORDS and len(w) > 1]


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)

    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) >= 4:
            clean_sentences.append(sentence)

    return clean_sentences


def score_sentence(question: str, sentence: str) -> float:
    q_words = set(normalize_for_match(question))
    s_words = set(normalize_for_match(sentence))

    if not q_words or not s_words:
        return 0.0

    overlap = q_words.intersection(s_words)
    score = len(overlap) / len(q_words)

    # Ưu tiên câu có thông tin thường gặp trong tài liệu sự kiện
    bonus_patterns = [
        r"\d{1,2}/\d{1,2}/\d{4}",
        r"\d{1,2}h\d{0,2}",
        r"phụ trách",
        r"thời gian",
        r"địa điểm",
        r"hậu cần",
        r"truyền thông",
        r"nhân sự",
        r"ban tổ chức",
    ]

    for pattern in bonus_patterns:
        if re.search(pattern, sentence.lower()):
            score += 0.08

    return score


def extractive_answer(question: str, contexts: List[Dict], max_sentences: int = 3) -> str:
    """
    Fallback khi chưa có OPENAI_API_KEY.
    Thay vì in toàn bộ chunk, hàm này chỉ chọn các câu liên quan nhất với câu hỏi.
    """
    candidates = []

    for source_index, ctx in enumerate(contexts, start=1):
        text = ctx.get("text", "")
        sentences = split_sentences(text)

        for sentence in sentences:
            score = score_sentence(question, sentence)
            if score > 0:
                candidates.append(
                    {
                        "score": score,
                        "sentence": sentence,
                        "source_index": source_index,
                        "source": ctx.get("source"),
                        "page": ctx.get("page"),
                    }
                )

    if not candidates:
        return "Tôi chưa tìm thấy dữ liệu đủ căn cứ trong tài liệu đã tải lên."

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    selected = []
    seen = set()

    for item in candidates:
        sentence = item["sentence"]

        if sentence in seen:
            continue

        seen.add(sentence)
        selected.append(item)

        if len(selected) >= max_sentences:
            break

    answer_lines = ["Dựa trên tài liệu đã tải lên, có thể xác định như sau:"]

    for item in selected:
        answer_lines.append(
            f"- {item['sentence']} [Nguồn {item['source_index']}]"
        )

    return "\n".join(answer_lines)