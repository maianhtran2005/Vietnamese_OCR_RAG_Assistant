from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
from PIL import Image

from src.ocr import ocr_image


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_txt(file_path: Path) -> List[Dict]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return [
        {
            "text": clean_text(text),
            "source": file_path.name,
            "page": 1,
            "method": "txt",
        }
    ]


def _page_to_image(page: fitz.Page, dpi: int = 250) -> Image.Image:
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def load_pdf(
    file_path: Path,
    use_ocr_if_needed: bool = True,
    min_native_chars: int = 50,
    ocr_lang: str = "vie+eng",
) -> List[Dict]:
    """Load PDF pages.

    First tries native text extraction. If a page has too little text, it falls back
    to OCR. This handles both normal PDF and scanned PDF.
    """
    docs: List[Dict] = []
    pdf = fitz.open(file_path)

    for page_index, page in enumerate(pdf, start=1):
        native_text = clean_text(page.get_text("text", sort=True))
        method = "pdf_text"
        final_text = native_text

        if use_ocr_if_needed and len(native_text) < min_native_chars:
            image = _page_to_image(page)
            final_text = clean_text(ocr_image(image, lang=ocr_lang))
            method = "ocr"

        if final_text:
            docs.append(
                {
                    "text": final_text,
                    "source": file_path.name,
                    "page": page_index,
                    "method": method,
                }
            )

    pdf.close()
    return docs


def load_document(
    file_path: str | Path,
    use_ocr_if_needed: bool = True,
    ocr_lang: str = "vie+eng",
) -> List[Dict]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_txt(path)
    if suffix == ".pdf":
        return load_pdf(path, use_ocr_if_needed=use_ocr_if_needed, ocr_lang=ocr_lang)

    raise ValueError(f"Unsupported file type: {suffix}. Use .txt or .pdf")
