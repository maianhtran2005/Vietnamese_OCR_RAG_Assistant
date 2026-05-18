from __future__ import annotations

import os
import re
from typing import Dict, List


def format_sources(contexts: List[Dict]) -> str:
    lines = []

    for i, ctx in enumerate(contexts, start=1):
        lines.append(
            f"[Nguồn {i}] "
            f"File: {ctx.get('source')} | "
            f"Trang: {ctx.get('page')} | "
            f"Phương thức: {ctx.get('method')} | "
            f"Điểm liên quan: {ctx.get('score', 0):.3f}\n"
            f"{ctx.get('text')}"
        )

    return "\n\n".join(lines)


def build_prompt(question: str, contexts: List[Dict]) -> str:
    return f"""
Bạn là trợ lý hỏi đáp tài liệu tiếng Việt.

Chỉ sử dụng thông tin trong NGỮ CẢNH để trả lời.
Chỉ trả lời đúng nội dung được hỏi.
Không chép lại toàn bộ đoạn nguồn.
Mỗi ý trả lời cần có trích dẫn nguồn dạng [Nguồn 1], [Nguồn 2].
Nếu không đủ căn cứ, trả lời: "Tôi chưa tìm thấy dữ liệu đủ căn cứ trong tài liệu đã tải lên."

CÂU HỎI:
{question}

NGỮ CẢNH:
{format_sources(contexts)}

YÊU CẦU:
- Trả lời ngắn gọn.
- Ưu tiên tên người, nhiệm vụ, thời gian, địa điểm, số liệu nếu có.
- Không bịa thông tin ngoài tài liệu.
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
                "Không gọi được LLM qua API. "
                "Hệ thống tạm trích xuất nội dung liên quan nhất từ tài liệu.\n\n"
                f"Chi tiết lỗi: {exc}\n\n"
                + extractive_answer(question, contexts)
            )

    return extractive_answer(question, contexts)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def keywords_from_question(question: str) -> List[str]:
    stopwords = {
        "ai", "gì", "nào", "ở", "đâu", "khi", "là", "có", "không",
        "hãy", "cho", "biết", "nêu", "trình", "bày", "về", "của",
        "trong", "theo", "tài", "liệu", "phần", "mục", "nội", "dung",
        "người", "được", "đảm", "nhận", "phụ", "trách"
    }

    words = normalize(question).split()

    return [w for w in words if w not in stopwords and len(w) >= 2]


def split_units(text: str) -> List[str]:
    """
    Tách đoạn nguồn thành các đơn vị nhỏ.
    Ưu tiên giữ từng dòng phân công riêng biệt.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    units = []

    for line in text.split("\n"):
        line = line.strip(" -–—\t")

        if not line:
            continue

        small_parts = re.split(
            r"(?<=[.!?])\s+|;\s+|\|\s+|•\s+",
            line
        )

        for part in small_parts:
            part = part.strip(" -–—\t")
            if len(part.split()) >= 3:
                units.append(part)

    if units:
        return units

    text = re.sub(r"\s+", " ", text).strip()
    fallback_units = re.split(r"(?<=[.!?])\s+|;\s+|\|\s+|•\s+", text)

    return [
        unit.strip(" -–—\t")
        for unit in fallback_units
        if len(unit.split()) >= 3
    ]


def get_focus_terms(question: str) -> List[str]:
    """
    Lấy trọng tâm cụ thể trong câu hỏi.
    Không đưa các từ chung như 'phụ trách' vào focus,
    vì nếu không hệ thống sẽ kéo cả các ban khác.
    """
    q = normalize(question)

    important_terms = [
        "hậu cần",
        "truyền thông",
        "kỹ thuật",
        "nội dung",
        "nhân sự",
        "đối ngoại",
        "tài chính",
        "văn nghệ",
        "lễ tân",
        "an ninh",
        "y tế",
        "checkin",
        "check in",
        "địa điểm",
        "thời gian",
        "kinh phí",
        "khách mời",
        "ban tổ chức",
    ]

    return [term for term in important_terms if term in q]


def should_return_multiple(question: str) -> bool:
    q = normalize(question)

    multiple_signals = [
        "liệt kê",
        "danh sách",
        "những ai",
        "các ban",
        "các nhiệm vụ",
        "toàn bộ",
        "tất cả",
        "gồm những",
        "bao gồm",
    ]

    return any(signal in q for signal in multiple_signals)


def score_unit(question: str, unit: str) -> float:
    q_norm = normalize(question)
    u_norm = normalize(unit)

    q_keywords = keywords_from_question(question)

    if not q_keywords and not get_focus_terms(question):
        return 0.0

    score = 0.0

    for kw in q_keywords:
        if kw in u_norm:
            score += 1.0

    focus_terms = get_focus_terms(question)

    for term in focus_terms:
        if term in u_norm:
            score += 8.0

    question_task_terms = [
        "phụ trách",
        "phân công",
        "nhiệm vụ",
        "đảm nhận",
        "thực hiện",
    ]

    for term in question_task_terms:
        if term in q_norm and term in u_norm:
            score += 2.0

    if "ai" in q_norm:
        if re.search(r"\b[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){1,4}\b", unit):
            score += 1.0

    if "thời gian" in q_norm or "khi nào" in q_norm or "ngày nào" in q_norm:
        if re.search(r"\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}h\d{0,2}", unit):
            score += 3.0

    if "địa điểm" in q_norm or "ở đâu" in q_norm:
        location_terms = ["tại", "địa điểm", "hội trường", "phòng", "trường", "nhà"]
        if any(term in u_norm for term in location_terms):
            score += 3.0

    return score


def extractive_answer(question: str, contexts: List[Dict], max_items: int = 4) -> str:
    candidates = []
    focus_terms = get_focus_terms(question)

    for source_index, ctx in enumerate(contexts, start=1):
        text = ctx.get("text", "")
        units = split_units(text)

        for unit in units:
            unit_norm = normalize(unit)

            if focus_terms:
                has_focus = any(term in unit_norm for term in focus_terms)

                if not has_focus:
                    continue

            score = score_unit(question, unit)

            if score > 0:
                candidates.append(
                    {
                        "score": score,
                        "text": unit,
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

    limit = max_items if should_return_multiple(question) else 1

    for item in candidates:
        text = item["text"]

        if text in seen:
            continue

        seen.add(text)
        selected.append(item)

        if len(selected) >= limit:
            break

    lines = ["Dựa trên tài liệu đã tải lên, nội dung liên quan trực tiếp là:"]

    for item in selected:
        lines.append(f"- {item['text']} [Nguồn {item['source_index']}]")

    return "\n".join(lines)