# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

# Quan trọng: Import file chatbot để các mô hình được tải khi khởi động
import chatbot 

# --- Định nghĩa cấu trúc dữ liệu cho API ---
class ChatMessage(BaseModel):
    role: str # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    user_message: str
    history: List[ChatMessage]

# --- Khởi tạo ứng dụng FastAPI ---
app = FastAPI(
    title="Athena AI Therapist API",
    description="API for the virtual psychologist chatbot.",
    version="1.0.0"
)

@app.get("/", tags=["Status"])
def read_root():
    """Endpoint để kiểm tra API có hoạt động không."""
    return {"status": "Athena AI Therapist API is running"}

@app.post("/chat", tags=["Chat"])
def handle_chat(request: ChatRequest):
    """
    Endpoint chính để trò chuyện với Athena.
    Nhận tin nhắn mới của người dùng và lịch sử cuộc trò chuyện,
    trả về phản hồi của AI.
    """
    response_text = chatbot.generate_response(
        user_input=request.user_message,
        history=[msg.dict() for msg in request.history]
    )
    
    return {"response": response_text}