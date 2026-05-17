# Vietnamese OCR + RAG Assistant for Student Event Documents

Dự án demo chatbot hỏi đáp tài liệu tiếng Việt phục vụ quản lý chương trình, báo cáo, kế hoạch và phân công nhân sự. Hệ thống hỗ trợ PDF thường, PDF scan thông qua OCR, file `.txt`, sau đó chia tài liệu thành chunks, tạo embeddings, tìm kiếm đoạn liên quan bằng FAISS và sinh câu trả lời có nguồn trích.

## 1. Tính năng

| Tính năng | Trạng thái |
|---|---|
| Upload file `.txt` hoặc `.pdf` | Có |
| Trích xuất text từ PDF thường | Có |
| OCR PDF scan tiếng Việt | Có, dùng Tesseract OCR |
| Chia tài liệu thành chunks | Có |
| Tạo embedding multilingual | Có, dùng SentenceTransformers |
| Tìm đoạn liên quan | Có, dùng FAISS |
| Trả lời có nguồn trích | Có |
| Không trả lời nếu không có dữ liệu | Có, thông qua ngưỡng `min_score` |
| Giao diện Streamlit | Có |
| Log câu hỏi/câu trả lời | Có, lưu tại `logs/qa_log.csv` |

## 2. Kiến trúc xử lý

```text
File .txt/.pdf
   ↓
Loader
   ├── PDF thường: PyMuPDF extract text
   └── PDF scan: OCR bằng Tesseract
   ↓
Chunker
   ↓
Embedding model multilingual
   ↓
FAISS vector index
   ↓
Retriever lấy top-k đoạn liên quan
   ↓
Generator trả lời dựa trên ngữ cảnh + nguồn trích
```

## 3. Cấu trúc thư mục

```text
rag-ocr-assistant/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── sample_document.txt
│   └── uploads/
├── logs/
│   └── qa_log.csv
├── storage/
└── src/
    ├── __init__.py
    ├── loader.py
    ├── ocr.py
    ├── chunker.py
    ├── embeddings.py
    ├── retriever.py
    ├── generator.py
    └── logger.py
```

## 4. Cài đặt

### Bước 1: Tạo môi trường ảo

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Bước 2: Cài thư viện Python

```bash
pip install -r requirements.txt
```

### Bước 3: Cài Tesseract OCR

Vì OCR dùng `pytesseract`, bạn cần cài thêm phần mềm Tesseract OCR trên máy.

Windows: cài Tesseract OCR, sau đó kiểm tra file thường nằm tại:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Ubuntu/Debian:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-vie
```

macOS:

```bash
brew install tesseract tesseract-lang
```

### Bước 4: Tạo file `.env`

```bash
copy .env.example .env
```

Nếu dùng Windows, sửa trong `.env`:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
OCR_LANG=vie+eng
```

Nếu muốn dùng LLM để sinh câu trả lời tự nhiên, thêm:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Nếu chưa có API key, app vẫn chạy được nhưng sẽ trả về các đoạn liên quan nhất thay vì sinh câu trả lời tự nhiên.

## 5. Chạy ứng dụng

```bash
streamlit run app.py
```

Sau khi giao diện mở:

1. Upload file `.txt` hoặc `.pdf`.
2. Bật OCR nếu tài liệu là PDF scan.
3. Bấm **Xây dựng chỉ mục**.
4. Nhập câu hỏi.
5. Xem câu trả lời và phần nguồn truy xuất.

## 6. Ví dụ câu hỏi

Với file mẫu trong `data/sample_document.txt`, có thể hỏi:

- Chương trình diễn ra vào thời gian nào?
- Ai phụ trách hậu cần?
- Ban Kỹ thuật cần chuẩn bị những gì?
- Các bộ phận phải hoàn thành công việc trước thời hạn nào?

## 7. Giải thích kỹ thuật để đưa vào CV/phỏng vấn

Dự án xây dựng hệ thống OCR + Retrieval-Augmented Generation cho tài liệu tiếng Việt. Với PDF thông thường, hệ thống trích xuất văn bản trực tiếp bằng PyMuPDF. Với PDF scan, hệ thống chuyển từng trang thành ảnh và dùng Tesseract OCR để nhận dạng tiếng Việt. Văn bản sau đó được chuẩn hóa, chia thành các chunk có overlap để tránh mất ngữ cảnh ở ranh giới đoạn. Mỗi chunk được mã hóa thành vector bằng mô hình embedding multilingual của SentenceTransformers. Các vector được lưu trong FAISS để truy xuất top-k đoạn có độ tương đồng cao nhất với câu hỏi. Phần generator chỉ trả lời dựa trên các đoạn truy xuất và kèm nguồn trích, đồng thời từ chối trả lời khi điểm liên quan thấp nhằm hạn chế hallucination.

## 8. Hướng phát triển tiếp theo

- Thêm xử lý file `.docx`, `.xlsx` cho kế hoạch và bảng phân công.
- Thêm metadata: loại văn bản, ngày ban hành, đơn vị phụ trách, người ký.
- Thêm reranker để sắp xếp lại các chunk trước khi đưa vào LLM.
- Thay Tesseract bằng PaddleOCR nếu cần OCR tốt hơn với ảnh chụp, bảng biểu hoặc bố cục phức tạp.
- Lưu index bền vững xuống `storage/` và cho phép nạp lại ở lần chạy sau.
- Thêm chức năng xuất câu trả lời ra Word/PDF cho báo cáo hành chính.

## 9. Lỗi thường gặp

### Không OCR được tiếng Việt

Kiểm tra đã cài language pack tiếng Việt chưa. Với Ubuntu/Debian, cần có `tesseract-ocr-vie`. Với Windows, kiểm tra thư mục `tessdata` có file tiếng Việt không.

### Streamlit báo chưa có chỉ mục

Bạn cần upload file và bấm **Xây dựng chỉ mục** trước khi hỏi.

### Câu trả lời nói chưa tìm thấy dữ liệu

Có thể do câu hỏi không liên quan đến tài liệu hoặc ngưỡng `min_score` đang quá cao. Hãy giảm ngưỡng trong sidebar từ `0.35` xuống `0.25` để thử lại.
