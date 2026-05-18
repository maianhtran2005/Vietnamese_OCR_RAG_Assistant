from __future__ import annotations

import os
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

        answer = generate_answer(question, contexts)
        log_qa(question, answer, contexts)

        render_answer(answer)
        render_confidence(contexts)
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
