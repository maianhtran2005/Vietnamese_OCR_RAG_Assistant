from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.chunker import chunk_documents
from src.embeddings import EmbeddingModel
from src.generator import generate_answer
from src.loader import load_document
from src.logger import log_qa
from src.retriever import VectorStore

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
st.set_page_config(page_title="Vietnamese OCR + RAG Assistant", page_icon="📄", layout="wide")

st.title("📄 Vietnamese OCR + RAG Assistant")
st.caption("Chatbot hỏi đáp tài liệu tiếng Việt cho kế hoạch, báo cáo, chương trình và phân công nhân sự.")
if "uploader_reset_counter" not in st.session_state:
    st.session_state.uploader_reset_counter = 0
with st.sidebar:
    st.header("Cấu hình")
    use_ocr = st.toggle("Dùng OCR nếu PDF là bản scan", value=True)
    ocr_lang = st.text_input("Ngôn ngữ OCR", value=os.getenv("OCR_LANG", "vie+eng"))
    chunk_size = st.slider("Chunk size / số từ", min_value=100, max_value=700, value=250, step=50)
    overlap = st.slider("Overlap / số từ", min_value=20, max_value=150, value=40, step=10)
    top_k = st.slider("Số đoạn truy xuất", min_value=1, max_value=10, value=6)
    min_score = st.slider("Ngưỡng liên quan", min_value=0.00, max_value=0.80, value=0.00, step=0.05)

    st.divider()

    if st.button("Reset toàn bộ tài liệu", use_container_width=True):
        reset_document_state()
        st.success("Đã reset toàn bộ tài liệu, chỉ mục và file upload cũ.")
        st.rerun()

uploaded_files = st.file_uploader(
    "Tải tài liệu .txt hoặc .pdf",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    key=f"uploaded_files_{st.session_state.uploader_reset_counter}",
)
current_uploaded_signature = get_uploaded_files_signature(uploaded_files)

if "last_uploaded_signature" not in st.session_state:
    st.session_state.last_uploaded_signature = None

if "indexed_signature" not in st.session_state:
    st.session_state.indexed_signature = None

if current_uploaded_signature != st.session_state.last_uploaded_signature:
    # Nếu không phải lần đầu mở app, tức là người dùng đã thay đổi file upload
    if st.session_state.last_uploaded_signature is not None:
        clear_index_state_only()
        st.info("Đã phát hiện file upload thay đổi. Chỉ mục cũ đã được xóa, hãy bấm “Xây dựng chỉ mục” để xử lý file mới.")

    st.session_state.last_uploaded_signature = current_uploaded_signature

    # Nếu người dùng xóa hết file upload thì index hiện tại cũng không còn hợp lệ
    if current_uploaded_signature is None:
        st.session_state.indexed_signature = None
if "store" not in st.session_state:
    st.session_state.store = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "docs" not in st.session_state:
    st.session_state.docs = []

col1, col2 = st.columns([1, 1])

with col1:
    build_index = st.button("Xây dựng chỉ mục", type="primary", use_container_width=True)

with col2:
    clear_index = st.button("Xóa chỉ mục hiện tại", use_container_width=True)

if clear_index:
    reset_document_state()
    st.success("Đã xóa chỉ mục, tài liệu và file upload hiện tại.")
    st.rerun()

if build_index:
    if not uploaded_files:
        st.warning("Hãy tải lên ít nhất một file .txt hoặc .pdf.")
    else:
        all_docs = []
        progress = st.progress(0)
        status = st.empty()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            for i, file in enumerate(uploaded_files, start=1):
                status.write(f"Đang xử lý: {file.name}")
                file_path = tmp_dir / file.name
                file_path.write_bytes(file.getbuffer())

                try:
                    docs = load_document(file_path, use_ocr_if_needed=use_ocr, ocr_lang=ocr_lang)
                    all_docs.extend(docs)
                except Exception as exc:
                    st.error(f"Lỗi khi đọc file {file.name}: {exc}")

                progress.progress(i / len(uploaded_files))

        if not all_docs:
            st.error("Không trích xuất được văn bản từ tài liệu. Hãy kiểm tra file hoặc cấu hình OCR.")
        else:
            chunks = chunk_documents(all_docs, chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                st.error("Không tạo được chunk từ tài liệu.")
            else:
                status.write("Đang tạo embeddings và xây dựng FAISS index...")
                embedder = EmbeddingModel()
                vectors = embedder.encode([c["text"] for c in chunks])

                store = VectorStore()
                store.build(vectors, chunks)

                st.session_state.embedder = embedder
                st.session_state.store = store
                st.session_state.chunks = chunks
                st.session_state.docs = all_docs
                st.session_state.indexed_signature = current_uploaded_signature

                st.success(f"Đã tạo chỉ mục: {len(all_docs)} trang/đơn vị văn bản, {len(chunks)} chunks.")

st.divider()

question = st.text_input(
    "Nhập câu hỏi",
    placeholder="Ví dụ: Ai phụ trách hậu cần? Chương trình diễn ra vào thời gian nào? Kế hoạch có những nhiệm vụ nào?",
)

ask = st.button("Hỏi tài liệu", type="primary")

if ask:
    if not question.strip():
        st.warning("Hãy nhập câu hỏi.")
    elif current_uploaded_signature != st.session_state.get("indexed_signature"):
        st.warning("File upload đã thay đổi hoặc chưa được xây dựng chỉ mục. Hãy bấm “Xây dựng chỉ mục” trước khi hỏi.")
    elif st.session_state.store is None or st.session_state.embedder is None:
        st.warning("Bạn cần xây dựng chỉ mục trước khi hỏi.")
    else:
        query_vector = st.session_state.embedder.encode([question])
        contexts = st.session_state.store.search(
            query_vector,
            query_text=question,
            top_k=top_k,
            min_score=min_score,
        )

        refuse, refuse_reason = should_refuse_answer(question, contexts)

        if refuse:
            answer = "Tôi chưa tìm thấy dữ liệu đủ căn cứ trong tài liệu đã tải lên."
        else:
            answer = generate_answer(question, contexts)

        log_qa(question, answer, contexts)

        render_answer(answer)

        render_confidence(contexts)

        if refuse:
            st.caption(f"Lý do từ chối: {refuse_reason}")
        st.divider()

        with st.expander("Xem nguồn trích xuất từ tài liệu", expanded=False):
            render_sources(contexts)

with st.expander("Xem thống kê phiên hiện tại"):
    st.write({"documents/pages": len(st.session_state.docs), "chunks": len(st.session_state.chunks)})

    if st.session_state.docs:
        st.markdown("**Văn bản trích xuất mẫu:**")
        st.text(st.session_state.docs[0].get("text", "")[:1000])

    if st.session_state.chunks:
        st.markdown("**3 chunks đầu tiên:**")
        for c in st.session_state.chunks[:3]:
            st.code(c.get("text", "")[:1000])
