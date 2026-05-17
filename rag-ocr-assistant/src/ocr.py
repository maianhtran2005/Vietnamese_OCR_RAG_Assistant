from __future__ import annotations

import os
from typing import Optional

from PIL import Image


def configure_tesseract() -> None:
    """Configure tesseract executable path from environment when needed.

    On Windows, set TESSERACT_CMD in .env, for example:
    TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
    """
    tesseract_cmd = os.getenv("TESSERACT_CMD")
    if not tesseract_cmd:
        return

    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    except Exception:
        # Let the real OCR call raise a clear error later.
        pass


def ocr_image(image: Image.Image, lang: str = "vie+eng", psm: Optional[int] = 6) -> str:
    """Run OCR for a PIL image.

    Args:
        image: PIL image object.
        lang: Tesseract language code. Use "vie+eng" for Vietnamese + English.
        psm: Page segmentation mode. 6 works well for mostly uniform text blocks.

    Returns:
        Extracted text.
    """
    configure_tesseract()

    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is not installed. Run: pip install pytesseract"
        ) from exc

    config = f"--psm {psm}" if psm is not None else ""
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    return text.strip()
