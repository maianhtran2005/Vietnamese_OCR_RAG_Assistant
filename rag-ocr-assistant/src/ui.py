from __future__ import annotations

import csv
import io
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.encoding import configure_utf8_io
from src.logger import log_qa
from src.model_runner import (
    build_index_from_documents,
    load_uploaded_document,
    retrieve_contexts,
    run_answer_generation,
)

configure_utf8_io()
load_dotenv()
def reset_document_state() -> None:
    """
    Reset toàn bộ tài liệu, chunks, vector store và widget upload file.
    Không xóa log hỏi đáp.
    """
    keys_to_clear = [
    "store",
    "embedder",
    "docs",
    "documents",
    "chunks",
    "uploaded_file_name",
    "current_file",
    "last_answer",
    "last_contexts",
    "messages",
    "last_uploaded_signature",
    "indexed_signature",
    "qa_history",
]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Đổi key của file_uploader để Streamlit xóa file upload cũ khỏi giao diện
    st.session_state.uploader_reset_counter = (
        st.session_state.get("uploader_reset_counter", 0) + 1
    )

    # Xóa file index nếu có lưu trong storage
    storage_dir = Path("storage")
    if storage_dir.exists():
        for file_path in storage_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()

    st.cache_data.clear()
    st.cache_resource.clear()
def clear_index_state_only() -> None:
    """
    Chỉ xóa index và dữ liệu đã xử lý.
    Không xóa file đang upload trên giao diện.
    Dùng khi phát hiện người dùng upload file mới.
    """
    keys_to_clear = [
        "store",
        "embedder",
        "docs",
        "documents",
        "chunks",
        "last_answer",
        "last_contexts",
        "messages",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    storage_dir = Path("storage")
    if storage_dir.exists():
        for file_path in storage_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()

    st.cache_data.clear()
    st.cache_resource.clear()


def get_uploaded_files_signature(uploaded_files) -> tuple | None:
    """
    Tạo dấu hiệu nhận biết bộ file đang upload.
    Nếu người dùng đổi file, tên file hoặc dung lượng sẽ khác.
    """
    if not uploaded_files:
        return None

    signature = []

    for file in uploaded_files:
        file_name = getattr(file, "name", "")
        file_size = getattr(file, "size", 0)
        signature.append((file_name, file_size))

    return tuple(sorted(signature))
def render_answer(answer: str) -> None:
    st.subheader("Câu trả lời")

    if not answer or not answer.strip():
        st.warning("Chưa tạo được câu trả lời.")
        return

    if "Tôi chưa tìm thấy dữ liệu đủ căn cứ" in answer:
        st.warning(answer)
    else:
        st.markdown(answer)


def render_sources(contexts) -> None:
    st.subheader("Nguồn trích xuất")

    if not contexts:
        st.info("Không có nguồn trích xuất phù hợp.")
        return

    for index, ctx in enumerate(contexts, start=1):
        source = ctx.get("source", "Không rõ file")
        page = ctx.get("page", "Không rõ")
        method = ctx.get("method", "Không rõ")
        score = ctx.get("score", 0)
        text = ctx.get("text", "")

        with st.expander(f"Nguồn {index} | File: {source} | Trang: {page} | Điểm: {score:.3f}"):
            st.caption(f"Phương thức đọc tài liệu: {method}")
            st.markdown("**Đoạn tài liệu được truy xuất:**")
            st.text_area(
                label=f"Nội dung nguồn {index}",
                value=text,
                height=180,
                disabled=True,
                key=f"source_text_{index}",
            )
def get_confidence_level(contexts) -> tuple[str, str]:
    """
    Đánh giá độ tin cậy dựa trên nguồn truy xuất.
    Đây không phải độ đúng tuyệt đối, mà là mức độ hệ thống tìm được nguồn phù hợp.
    """
    if not contexts:
        return "Thấp", "Không tìm thấy nguồn liên quan trong tài liệu."

    best_score = max(float(ctx.get("score", 0)) for ctx in contexts)
    best_semantic = max(float(ctx.get("semantic_score", 0)) for ctx in contexts)
    best_keyword = max(float(ctx.get("keyword_score", 0)) for ctx in contexts)

    if best_score >= 0.65 or best_keyword >= 0.50 or best_semantic >= 0.55:
        return "Cao", (
            f"Nguồn truy xuất có mức liên quan tốt "
            f"(score={best_score:.3f}, semantic={best_semantic:.3f}, keyword={best_keyword:.3f})."
        )

    if best_score >= 0.30 or best_keyword >= 0.20 or best_semantic >= 0.30:
        return "Trung bình", (
            f"Hệ thống tìm được nguồn có liên quan nhưng chưa thật mạnh "
            f"(score={best_score:.3f}, semantic={best_semantic:.3f}, keyword={best_keyword:.3f})."
        )

    return "Thấp", (
        f"Nguồn truy xuất yếu, nên kiểm tra lại đoạn trích trước khi sử dụng "
        f"(score={best_score:.3f}, semantic={best_semantic:.3f}, keyword={best_keyword:.3f})."
    )


def render_confidence(contexts) -> None:
    level, reason = get_confidence_level(contexts)

    if level == "Cao":
        st.success(f"Độ tin cậy: {level}")
    elif level == "Trung bình":
        st.warning(f"Độ tin cậy: {level}")
    else:
        st.error(f"Độ tin cậy: {level}")

    st.caption(reason)
def normalize_for_refuse(text: str) -> str:
    text = text.lower()
    text = re.sub(
        r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
        r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_broad_document_question(question: str) -> bool:
    """
    Các câu hỏi tổng quát được phép trả lời nếu có tài liệu,
    ví dụ: tóm tắt tài liệu, nội dung chính là gì.
    """
    q = normalize_for_refuse(question)

    broad_signals = [
        "tóm tắt",
        "nội dung chính",
        "tài liệu nói gì",
        "văn bản nói gì",
        "khái quát",
        "tổng quan",
        "cho biết nội dung",
    ]

    return any(signal in q for signal in broad_signals)


def extract_question_terms(question: str) -> list[str]:
    """
    Lấy từ/cụm từ trọng tâm của câu hỏi.
    Mục tiêu: câu hỏi ngoài tài liệu như 'hiệu trưởng là ai'
    phải bắt buộc tìm thấy 'hiệu trưởng' trong nguồn, nếu không thì từ chối.
    """
    q = normalize_for_refuse(question)

    phrase_terms = [
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
        "hiệu trưởng",
        "thời tiết",
        "ngày sinh",
        "điểm thi",
        "kết quả học tập",
    ]

    terms = [term for term in phrase_terms if term in q]

    stopwords = {
        "ai", "gì", "nào", "ở", "đâu", "khi", "là", "có", "không",
        "hãy", "cho", "biết", "nêu", "trình", "bày", "về", "của",
        "trong", "theo", "tài", "liệu", "văn", "bản", "phần", "mục",
        "nội", "dung", "người", "được", "đảm", "nhận", "phụ", "trách",
        "gồm", "những", "các", "bao", "nhiêu", "trường", "đại", "học",
        "khoa", "tự", "nhiên", "chương", "trình", "này", "đó", "của",
    }

    words = q.split()

    for word in words:
        if word not in stopwords and len(word) >= 3:
            terms.append(word)

    unique_terms = []
    for term in terms:
        if term not in unique_terms:
            unique_terms.append(term)

    return unique_terms


def should_refuse_answer(question: str, contexts) -> tuple[bool, str]:
    """
    Từ chối trả lời nếu:
    - Không có nguồn;
    - Câu hỏi có từ khóa trọng tâm nhưng nguồn không chứa từ khóa đó;
    - Điểm truy xuất quá yếu.
    """
    if not contexts:
        return True, "Không tìm thấy nguồn liên quan trong tài liệu."

    best_score = max(float(ctx.get("score", 0)) for ctx in contexts)
    best_semantic = max(float(ctx.get("semantic_score", 0)) for ctx in contexts)
    best_keyword = max(float(ctx.get("keyword_score", 0)) for ctx in contexts)

    context_text = " ".join(str(ctx.get("text", "")) for ctx in contexts)
    context_norm = normalize_for_refuse(context_text)

    # Cho phép câu hỏi tổng quát nếu có nguồn tương đối ổn
    if is_broad_document_question(question):
        if best_score < 0.10 and best_semantic < 0.15 and best_keyword < 0.05:
            return True, (
                f"Nguồn truy xuất quá yếu cho câu hỏi tổng quát "
                f"(score={best_score:.3f}, semantic={best_semantic:.3f}, keyword={best_keyword:.3f})."
            )
        return False, "Câu hỏi tổng quát về tài liệu, cho phép trả lời."

    question_terms = extract_question_terms(question)
    matched_terms = [term for term in question_terms if term in context_norm]

    # Chặn mạnh nhất: câu hỏi có từ khóa trọng tâm nhưng nguồn không chứa từ khóa đó
    if question_terms and not matched_terms:
        return True, (
            "Không tìm thấy từ khóa trọng tâm của câu hỏi trong nguồn truy xuất. "
            f"Từ khóa câu hỏi: {', '.join(question_terms[:8])}."
        )

    # Chặn trường hợp retriever lấy bừa nguồn gần nhất
    if best_score < 0.25 and best_semantic < 0.35 and best_keyword < 0.15:
        return True, (
            f"Nguồn truy xuất yếu "
            f"(score={best_score:.3f}, semantic={best_semantic:.3f}, keyword={best_keyword:.3f})."
        )

    return False, (
        f"Nguồn truy xuất đủ căn cứ sơ bộ "
        f"(matched_terms={matched_terms[:8]}, score={best_score:.3f}, "
        f"semantic={best_semantic:.3f}, keyword={best_keyword:.3f})."
    )
def get_available_source_options() -> list[str]:
    docs = st.session_state.get("docs", [])

    sources = sorted(
        {
            str(doc.get("source", "")).strip()
            for doc in docs
            if str(doc.get("source", "")).strip()
        }
    )

    return ["Tất cả tài liệu"] + sources


def parse_page_filter(page_filter: str) -> set[str] | None:
    """
    Nhận chuỗi lọc trang dạng:
    1
    1,2,3
    1-3
    1,3,5-7
    """
    page_filter = page_filter.strip()

    if not page_filter:
        return None

    pages = set()

    parts = [part.strip() for part in page_filter.split(",") if part.strip()]

    for part in parts:
        if "-" in part:
            start_text, end_text = part.split("-", 1)

            try:
                start = int(start_text.strip())
                end = int(end_text.strip())

                if start <= end:
                    for page in range(start, end + 1):
                        pages.add(str(page))
            except ValueError:
                continue
        else:
            pages.add(part)

    return pages if pages else None


def filter_contexts_by_scope(contexts, selected_source: str, page_filter: str):
    allowed_pages = parse_page_filter(page_filter)
    filtered = []

    for ctx in contexts:
        ctx_source = str(ctx.get("source", "")).strip()
        ctx_page = str(ctx.get("page", "")).strip()

        if selected_source != "Tất cả tài liệu" and ctx_source != selected_source:
            continue

        if allowed_pages is not None and ctx_page not in allowed_pages:
            continue

        filtered.append(ctx)

    return filtered


def get_scope_description(selected_source: str, page_filter: str) -> str:
    source_part = selected_source

    if page_filter.strip():
        return f"{source_part}, trang: {page_filter.strip()}"

    return source_part
def init_qa_history() -> None:
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []


def add_to_qa_history(question: str, answer: str, contexts) -> None:
    best_score = 0.0
    confidence_level = "Thấp"

    if contexts:
        best_score = max(float(ctx.get("score", 0)) for ctx in contexts)
        confidence_level, _ = get_confidence_level(contexts)

    source_details = []

    for index, ctx in enumerate(contexts, start=1):
        source_details.append(
            {
                "source_index": index,
                "source": ctx.get("source", ""),
                "page": ctx.get("page", ""),
                "method": ctx.get("method", ""),
                "score": round(float(ctx.get("score", 0)), 3),
                "semantic_score": round(float(ctx.get("semantic_score", 0)), 3),
                "keyword_score": round(float(ctx.get("keyword_score", 0)), 3),
                "text": ctx.get("text", ""),
            }
        )

    st.session_state.qa_history.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer,
            "num_sources": len(contexts),
            "best_score": round(best_score, 3),
            "confidence": confidence_level,
            "sources": source_details,
        }
    )
def convert_qa_history_to_csv() -> str:
    if "qa_history" not in st.session_state or not st.session_state.qa_history:
        return ""

    output = io.StringIO()

    fieldnames = [
        "time",
        "question",
        "answer",
        "num_sources",
        "best_score",
        "confidence",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for item in st.session_state.qa_history:
        writer.writerow(
            {
                "time": item.get("time", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "num_sources": item.get("num_sources", 0),
                "best_score": item.get("best_score", 0),
                "confidence": item.get("confidence", ""),
            }
        )

    return output.getvalue()
def convert_session_report_to_txt(uploaded_files) -> str:
    lines = []

    lines.append("VIETNAMESE OCR + RAG ASSISTANT")
    lines.append("BÁO CÁO DEMO PHIÊN HỎI ĐÁP")
    lines.append("=" * 70)
    lines.append(f"Thời gian xuất báo cáo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("1. THÔNG TIN TÀI LIỆU")
    lines.append("-" * 70)

    if uploaded_files:
        lines.append("File đã tải lên:")
        for file in uploaded_files:
            file_name = getattr(file, "name", "")
            file_size = getattr(file, "size", 0)
            lines.append(f"- {file_name} ({file_size} bytes)")
    else:
        lines.append("Chưa có file đang được tải lên.")

    lines.append("")
    lines.append(f"Số documents/pages đã xử lý: {len(st.session_state.get('docs', []))}")
    lines.append(f"Số chunks đã tạo: {len(st.session_state.get('chunks', []))}")
    lines.append("")

    lines.append("2. LỊCH SỬ HỎI ĐÁP")
    lines.append("-" * 70)

    history = st.session_state.get("qa_history", [])

    if not history:
        lines.append("Chưa có câu hỏi nào trong phiên hiện tại.")
        return "\n".join(lines)

    for idx, item in enumerate(history, start=1):
        lines.append("")
        lines.append(f"Câu hỏi {idx}")
        lines.append("-" * 40)
        lines.append(f"Thời gian: {item.get('time', '')}")
        lines.append(f"Câu hỏi: {item.get('question', '')}")
        lines.append(f"Độ tin cậy: {item.get('confidence', '')}")
        lines.append(f"Số nguồn: {item.get('num_sources', 0)}")
        lines.append(f"Best score: {item.get('best_score', 0)}")
        lines.append("")
        lines.append("Câu trả lời:")
        lines.append(str(item.get("answer", "")))
        lines.append("")

        sources = item.get("sources", [])

        if sources:
            lines.append("Nguồn trích xuất:")
            for src in sources:
                lines.append("")
                lines.append(
                    f"[Nguồn {src.get('source_index')}] "
                    f"File: {src.get('source')} | "
                    f"Trang: {src.get('page')} | "
                    f"Method: {src.get('method')} | "
                    f"Score: {src.get('score')} | "
                    f"Semantic: {src.get('semantic_score')} | "
                    f"Keyword: {src.get('keyword_score')}"
                )
                lines.append("Đoạn trích:")
                lines.append(str(src.get("text", ""))[:1200])
        else:
            lines.append("Không có nguồn trích xuất.")

        lines.append("")
        lines.append("=" * 70)

    return "\n".join(lines)
def convert_extracted_text_to_txt() -> str:
    docs = st.session_state.get("docs", [])

    lines = []
    lines.append("VĂN BẢN ĐÃ TRÍCH XUẤT TỪ TÀI LIỆU")
    lines.append("=" * 70)
    lines.append(f"Thời gian xuất: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Số đơn vị văn bản/pages: {len(docs)}")
    lines.append("")

    if not docs:
        lines.append("Chưa có văn bản được trích xuất.")
        return "\n".join(lines)

    for index, doc in enumerate(docs, start=1):
        lines.append("")
        lines.append(f"ĐƠN VỊ VĂN BẢN {index}")
        lines.append("-" * 70)
        lines.append(f"File: {doc.get('source', '')}")
        lines.append(f"Trang: {doc.get('page', '')}")
        lines.append(f"Phương thức đọc: {doc.get('method', '')}")
        lines.append("")
        lines.append(str(doc.get("text", "")))
        lines.append("")
        lines.append("=" * 70)

    return "\n".join(lines)
def render_extracted_text_preview() -> None:
    docs = st.session_state.get("docs", [])

    st.subheader("Xem trước văn bản đã trích xuất")

    if not docs:
        st.info("Chưa có văn bản trích xuất. Hãy upload file và bấm “Xây dựng chỉ mục”.")
        return

    extracted_txt = convert_extracted_text_to_txt()

    st.download_button(
        label="Tải toàn bộ văn bản trích xuất TXT",
        data=extracted_txt.encode("utf-8-sig"),
        file_name="extracted_text.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.caption(
        "Dùng phần này để kiểm tra PDF scan/OCR có đọc đúng tiếng Việt, tên người, thời gian, địa điểm hay không."
    )

    max_pages = min(len(docs), 10)

    for index, doc in enumerate(docs[:max_pages], start=1):
        source = doc.get("source", "Không rõ file")
        page = doc.get("page", "Không rõ")
        method = doc.get("method", "Không rõ")
        text = doc.get("text", "")

        with st.expander(
            f"Văn bản {index} | File: {source} | Trang: {page} | Method: {method}",
            expanded=False,
        ):
            st.text_area(
                label=f"Nội dung trích xuất {index}",
                value=text,
                height=260,
                disabled=True,
                key=f"extracted_text_preview_{index}",
            )

    if len(docs) > max_pages:
        st.info(f"Đang hiển thị {max_pages}/{len(docs)} đơn vị văn bản đầu tiên. Tải TXT để xem toàn bộ.")
def estimate_extraction_quality(docs) -> dict:
    """
    Đánh giá sơ bộ chất lượng văn bản đã trích xuất/OCR.
    Đây là kiểm tra heuristic, không phải đánh giá tuyệt đối.
    """
    if not docs:
        return {
            "score": 0,
            "label": "Chưa có dữ liệu",
            "total_docs": 0,
            "total_chars": 0,
            "empty_docs": 0,
            "short_docs": 0,
            "suspicious_chars": 0,
            "avg_chars": 0,
            "low_quality_items": [],
        }

    total_docs = len(docs)
    total_chars = 0
    empty_docs = 0
    short_docs = 0
    suspicious_chars = 0
    low_quality_items = []

    for index, doc in enumerate(docs, start=1):
        text = str(doc.get("text", "")).strip()
        char_count = len(text)
        total_chars += char_count

        current_suspicious = text.count("�") + text.count("□") + text.count("???")
        suspicious_chars += current_suspicious

        reasons = []

        if char_count == 0:
            empty_docs += 1
            reasons.append("Không có văn bản")
        elif char_count < 80:
            short_docs += 1
            reasons.append("Văn bản quá ngắn")

        if current_suspicious > 0:
            reasons.append("Có ký tự lỗi OCR")

        if reasons:
            low_quality_items.append(
                {
                    "index": index,
                    "source": doc.get("source", ""),
                    "page": doc.get("page", ""),
                    "method": doc.get("method", ""),
                    "chars": char_count,
                    "issue": "; ".join(reasons),
                }
            )

    avg_chars = total_chars / total_docs if total_docs else 0

    empty_ratio = empty_docs / total_docs
    short_ratio = short_docs / total_docs
    suspicious_ratio = suspicious_chars / max(total_chars, 1)

    score = 100
    score -= empty_ratio * 45
    score -= short_ratio * 25
    score -= min(suspicious_ratio * 1000, 30)

    score = max(0, min(100, round(score, 1)))

    if score >= 80:
        label = "Tốt"
    elif score >= 55:
        label = "Trung bình"
    else:
        label = "Cần kiểm tra"

    return {
        "score": score,
        "label": label,
        "total_docs": total_docs,
        "total_chars": total_chars,
        "empty_docs": empty_docs,
        "short_docs": short_docs,
        "suspicious_chars": suspicious_chars,
        "avg_chars": round(avg_chars, 1),
        "low_quality_items": low_quality_items,
    }


def render_extraction_quality() -> None:
    docs = st.session_state.get("docs", [])
    quality = estimate_extraction_quality(docs)

    st.subheader("Đánh giá chất lượng OCR/trích xuất")

    if not docs:
        st.info("Chưa có dữ liệu để đánh giá. Hãy upload file và bấm “Xây dựng chỉ mục”.")
        return

    label = quality["label"]
    score = quality["score"]

    if label == "Tốt":
        st.success(f"Chất lượng trích xuất: {label} ({score}/100)")
    elif label == "Trung bình":
        st.warning(f"Chất lượng trích xuất: {label} ({score}/100)")
    else:
        st.error(f"Chất lượng trích xuất: {label} ({score}/100)")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Số đơn vị/trang", quality["total_docs"])
    col2.metric("Tổng ký tự", quality["total_chars"])
    col3.metric("Trang rỗng", quality["empty_docs"])
    col4.metric("Trang quá ngắn", quality["short_docs"])

    col5, col6 = st.columns(2)

    col5.metric("Ký tự lỗi OCR", quality["suspicious_chars"])
    col6.metric("Trung bình ký tự/trang", quality["avg_chars"])

    if quality["low_quality_items"]:
        st.markdown("**Các trang/đơn vị cần kiểm tra:**")
        st.table(quality["low_quality_items"][:20])
    else:
        st.info("Không phát hiện trang rỗng, trang quá ngắn hoặc ký tự lỗi rõ ràng.")

    st.caption(
        "Gợi ý: nếu chất lượng thấp, hãy kiểm tra lại file scan, độ phân giải ảnh, "
        "ngôn ngữ OCR `vie+eng`, hoặc xem trước văn bản OCR để biết trang nào bị đọc sai."
    )
def render_qa_history(uploaded_files=None) -> None:
    st.subheader("Lịch sử hỏi đáp trong phiên")

    if not st.session_state.qa_history:
        st.info("Chưa có câu hỏi nào trong phiên hiện tại.")
        return
    csv_data = convert_qa_history_to_csv()

    st.download_button(
        label="Tải lịch sử hỏi đáp CSV",
        data=csv_data.encode("utf-8-sig"),
        file_name="qa_history.csv",
        mime="text/csv",
        use_container_width=True,
    )
    report_txt = convert_session_report_to_txt(uploaded_files)

    st.download_button(
    label="Tải báo cáo demo TXT",
    data=report_txt.encode("utf-8-sig"),
    file_name="rag_demo_report.txt",
    mime="text/plain",
    use_container_width=True,
    )

    st.divider()
    for index, item in enumerate(reversed(st.session_state.qa_history), start=1):
        with st.expander(
            f"Câu hỏi {index}: {item['question']} | "
            f"{item['num_sources']} nguồn | score={item['best_score']:.3f}",
            expanded=False,
        ):
            st.markdown("**Câu hỏi:**")
            st.write(item["question"])

            st.markdown("**Câu trả lời:**")
            st.markdown(item["answer"])

def run_app() -> None:
    st.set_page_config(page_title="Vietnamese OCR + RAG Assistant", page_icon="📄", layout="wide")

    st.markdown(
        """
        <style>
        .block-container { max-width: 980px; padding-top: 1.4rem; padding-bottom: 6rem; }
        [data-testid="stSidebar"] { min-width: 330px; }
        .app-title { font-size: 1.35rem; font-weight: 700; margin-bottom: 0.15rem; }
        .app-caption { color: #667085; font-size: 0.92rem; margin-bottom: 1rem; }
        .status-pill {
            border: 1px solid #e4e7ec;
            border-radius: 999px;
            color: #475467;
            display: inline-block;
            font-size: 0.82rem;
            margin: 0 0.35rem 0.45rem 0;
            padding: 0.25rem 0.7rem;
        }
        .empty-chat {
            border: 1px solid #e4e7ec;
            border-radius: 8px;
            margin-top: 1.5rem;
            padding: 1.2rem;
        }
        .empty-chat h3 { font-size: 1.05rem; margin-bottom: 0.35rem; }
        .empty-chat p { color: #667085; margin-bottom: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_qa_history()

    defaults = {
        "uploader_reset_counter": 0,
        "store": None,
        "embedder": None,
        "chunks": [],
        "docs": [],
        "messages": [],
        "last_contexts": [],
        "last_refuse_reason": "",
        "last_uploaded_signature": None,
        "indexed_signature": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    with st.sidebar:
        st.header("Tài liệu")

        uploaded_files = st.file_uploader(
            "Tải PDF hoặc TXT",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key=f"uploaded_files_{st.session_state.uploader_reset_counter}",
        )

        current_uploaded_signature = get_uploaded_files_signature(uploaded_files)

        if current_uploaded_signature != st.session_state.last_uploaded_signature:
            if st.session_state.last_uploaded_signature is not None:
                clear_index_state_only()
                st.info("File đã đổi. Chỉ mục cũ đã được xóa.")

            st.session_state.last_uploaded_signature = current_uploaded_signature

            if current_uploaded_signature is None:
                st.session_state.indexed_signature = None

        indexed_ready = (
            st.session_state.store is not None
            and st.session_state.embedder is not None
            and current_uploaded_signature == st.session_state.indexed_signature
        )

        if indexed_ready:
            st.success(
                f"Sẵn sàng: {len(st.session_state.docs)} trang/đơn vị, "
                f"{len(st.session_state.chunks)} chunks."
            )
        elif uploaded_files:
            st.warning("Đã có file. Bấm xử lý để bắt đầu hỏi đáp.")
        else:
            st.info("Tải tài liệu để bắt đầu.")

        with st.expander("Cài đặt", expanded=False):
            use_ocr = st.toggle("OCR cho PDF scan", value=True)
            ocr_lang = st.text_input("Ngôn ngữ OCR", value=os.getenv("OCR_LANG", "vie+eng"))
            chunk_size = st.slider("Chunk size", min_value=100, max_value=700, value=250, step=50)
            overlap = st.slider("Overlap", min_value=20, max_value=150, value=40, step=10)
            top_k = st.slider("Số nguồn", min_value=1, max_value=10, value=6)
            min_score = st.slider("Ngưỡng liên quan", min_value=0.00, max_value=0.80, value=0.00, step=0.05)
            answer_mode_label = st.selectbox(
                "Chế độ trả lời",
                options=[
                    "Trả lời tự nhiên bằng LLM nếu có API key",
                    "Trích xuất an toàn",
                ],
                index=0,
            )

        answer_mode = "llm" if answer_mode_label.startswith("Trả lời tự nhiên") else "extractive"

        with st.expander("Phạm vi", expanded=False):
            source_options = get_available_source_options()
            selected_source = st.selectbox("Tài liệu", options=source_options, index=0)
            page_filter = st.text_input("Trang", placeholder="Ví dụ: 1 hoặc 1,2,5-7")

        build_index = st.button(
            "Xử lý tài liệu",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        )
        clear_index = st.button("Reset", use_container_width=True)

        if clear_index:
            reset_document_state()
            st.rerun()

        if build_index:
            all_docs = []
            failed_files = []

            with st.status("Đang xử lý tài liệu...", expanded=False) as status:
                progress = st.progress(0)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_dir = Path(tmp_dir)

                    for i, file in enumerate(uploaded_files, start=1):
                        original_name = file.name
                        status.update(label=f"Đang đọc {i}/{len(uploaded_files)}: {original_name}")
                        file_path = tmp_dir / f"{i}_{original_name}"

                        try:
                            file_path.write_bytes(file.getbuffer())
                            docs = load_uploaded_document(
                                file_path,
                                use_ocr_if_needed=use_ocr,
                                ocr_lang=ocr_lang,
                            )

                            if docs:
                                for doc in docs:
                                    doc["source"] = original_name
                                all_docs.extend(docs)
                            else:
                                failed_files.append(
                                    {"file": original_name, "reason": "Không trích xuất được văn bản."}
                                )

                        except Exception as exc:
                            failed_files.append({"file": original_name, "reason": str(exc)})

                        progress.progress(i / len(uploaded_files))

                if not all_docs:
                    status.update(label="Không đọc được tài liệu.", state="error", expanded=True)
                    st.error("Không trích xuất được văn bản từ file đã tải lên.")
                else:
                    try:
                        status.update(label="Đang tạo embeddings và FAISS index...")
                        chunks, embedder, store = build_index_from_documents(
                            all_docs,
                            chunk_size=chunk_size,
                            overlap=overlap,
                        )

                        st.session_state.embedder = embedder
                        st.session_state.store = store
                        st.session_state.chunks = chunks
                        st.session_state.docs = all_docs
                        st.session_state.indexed_signature = current_uploaded_signature
                        st.session_state.messages = []
                        st.session_state.last_contexts = []
                        st.session_state.last_refuse_reason = ""

                        status.update(
                            label=f"Đã sẵn sàng: {len(all_docs)} trang/đơn vị, {len(chunks)} chunks.",
                            state="complete",
                        )
                    except Exception as exc:
                        status.update(label="Lỗi khi tạo chỉ mục.", state="error", expanded=True)
                        st.error(str(exc))

            if failed_files:
                with st.expander("File bị bỏ qua", expanded=True):
                    st.table(failed_files)

        st.divider()

        with st.expander("Nguồn gần nhất", expanded=False):
            render_sources(st.session_state.get("last_contexts", []))

        with st.expander("Chất lượng OCR", expanded=False):
            render_extraction_quality()

        with st.expander("Lịch sử và báo cáo", expanded=False):
            render_qa_history(uploaded_files)

        with st.expander("Thống kê", expanded=False):
            st.write(
                {
                    "documents/pages": len(st.session_state.docs),
                    "chunks": len(st.session_state.chunks),
                    "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                }
            )

    st.markdown('<div class="app-title">Vietnamese OCR + RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-caption">Hỏi đáp tài liệu tiếng Việt với OCR, RAG và trích dẫn nguồn.</div>',
        unsafe_allow_html=True,
    )

    scope_text = get_scope_description(selected_source, page_filter)
    ready_text = "Sẵn sàng" if indexed_ready else "Chưa xử lý tài liệu"
    mode_text = "LLM" if answer_mode == "llm" else "Trích xuất"

    st.markdown(
        f"""
        <span class="status-pill">{ready_text}</span>
        <span class="status-pill">{len(st.session_state.docs)} trang/đơn vị</span>
        <span class="status-pill">{len(st.session_state.chunks)} chunks</span>
        <span class="status-pill">{mode_text}</span>
        <span class="status-pill">{scope_text}</span>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-chat">
                <h3>Tải tài liệu ở sidebar, rồi bắt đầu hỏi.</h3>
                <p>Các phần cấu hình, nguồn trích xuất, lịch sử và kiểm tra OCR được thu gọn để màn hình chính chỉ tập trung vào hội thoại.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("caption"):
                st.caption(message["caption"])

    question = st.chat_input("Hỏi tài liệu...", disabled=not indexed_ready)

    if not indexed_ready:
        st.caption("Tải file và bấm “Xử lý tài liệu” trong sidebar để mở chat.")

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    is_scope_filtered = selected_source != "Tất cả tài liệu" or bool(page_filter.strip())
    available_chunks = len(st.session_state.get("chunks", []))
    search_top_k = top_k

    if is_scope_filtered:
        search_top_k = min(max(top_k, 30), max(available_chunks, top_k))

    contexts = retrieve_contexts(
        st.session_state.store,
        st.session_state.embedder,
        question,
        top_k=search_top_k,
        min_score=min_score,
    )
    contexts = filter_contexts_by_scope(contexts, selected_source, page_filter)[:top_k]

    refuse, refuse_reason = should_refuse_answer(question, contexts)

    if refuse:
        answer = "Tôi chưa tìm thấy dữ liệu đủ căn cứ trong tài liệu đã tải lên."
    else:
        answer = run_answer_generation(question, contexts, answer_mode=answer_mode)

    log_qa(question, answer, contexts)
    add_to_qa_history(question, answer, contexts)
    st.session_state.last_contexts = contexts
    st.session_state.last_refuse_reason = refuse_reason if refuse else ""

    confidence_level, confidence_reason = get_confidence_level(contexts)
    caption = (
        f"Độ tin cậy: {confidence_level}. "
        f"Chế độ: {answer_mode_label}. "
        f"Phạm vi: {scope_text}."
    )

    if refuse:
        caption = f"{caption} Lý do từ chối: {refuse_reason}"
    else:
        caption = f"{caption} {confidence_reason}"

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "caption": caption}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(caption)
