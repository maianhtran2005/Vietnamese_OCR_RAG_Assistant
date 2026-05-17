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

st.set_page_config(page_title="Vietnamese OCR + RAG Assistant", page_icon="📄", layout="wide")

st.title("📄 Vietnamese OCR + RAG Assistant")
st.caption("Chatbot hỏi đáp tài liệu tiếng Việt cho kế hoạch, báo cáo, chương trình và phân công nhân sự.")

with st.sidebar:
    st.header("Cấu hình")
    use_ocr = st.toggle("Dùng OCR nếu PDF là bản scan", value=True)
    ocr_lang = st.text_input("Ngôn ngữ OCR", value=os.getenv("OCR_LANG", "vie+eng"))
    chunk_size = st.slider("Chunk size / số từ", min_value=150, max_value=900, value=450, step=50)
    overlap = st.slider("Overlap / số từ", min_value=20, max_value=200, value=80, step=10)
    top_k = st.slider("Số đoạn truy xuất", min_value=1, max_value=8, value=4)
    min_score = st.slider("Ngưỡng liên quan", min_value=0.00, max_value=0.80, value=0.10, step=0.05)

    st.divider()
    st.write("**Lưu ý:** Nếu chưa cấu hình `OPENAI_API_KEY`, app vẫn trả về các đoạn liên quan nhất thay vì gọi LLM.")

uploaded_files = st.file_uploader(
    "Tải tài liệu .txt hoặc .pdf",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)

if "store" not in st.session_state:
    st.session_state.store = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

col1, col2 = st.columns([1, 1])

with col1:
    build_index = st.button("Xây dựng chỉ mục", type="primary", use_container_width=True)

with col2:
    clear_index = st.button("Xóa chỉ mục hiện tại", use_container_width=True)

if clear_index:
    st.session_state.store = None
    st.session_state.chunks = []
    st.success("Đã xóa chỉ mục trong phiên hiện tại.")

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
                docs = load_document(file_path, use_ocr_if_needed=use_ocr, ocr_lang=ocr_lang)
                all_docs.extend(docs)
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
        contexts = st.session_state.store.search(query_vector, top_k=top_k, min_score=min_score)

        answer = generate_answer(question, contexts)
        log_qa(question, answer, contexts)

        st.subheader("Câu trả lời")
        st.markdown(answer)

        with st.expander("Xem nguồn truy xuất"):
            if not contexts:
                st.info("Không có đoạn nào vượt ngưỡng liên quan.")
            for i, ctx in enumerate(contexts, start=1):
                st.markdown(
                    f"**[Nguồn {i}]** `{ctx.get('source')}` — trang `{ctx.get('page')}` — "
                    f"method `{ctx.get('method')}` — score `{ctx.get('score', 0):.3f}`"
                )
                st.write(ctx.get("text"))
                st.divider()

with st.expander("Xem thống kê phiên hiện tại"):
    st.write({"chunks": len(st.session_state.chunks)})
    if st.session_state.chunks:
        st.write(st.session_state.chunks[:3])
