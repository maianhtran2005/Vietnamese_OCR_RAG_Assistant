from dotenv import load_dotenv
from google import genai
import os

from src.encoding import configure_utf8_io

configure_utf8_io()
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not api_key:
    raise ValueError("Chưa tìm thấy GEMINI_API_KEY trong file .env")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model=model,
    contents="Trả lời một câu ngắn bằng tiếng Việt: RAG là gì?"
)

print(response.text)
