# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import chatbot
from models import load_all_models # <-- Thêm dòng này

# --- Định nghĩa cấu trúc dữ liệu cho API ---
class ChatMessage(BaseModel):
    role: str
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

# --- Sự kiện khởi động ---
@app.on_event("startup")
async def startup_event():
    """
    Tải tất cả các mô hình AI khi ứng dụng bắt đầu.
    Việc này đảm bảo mọi thứ sẵn sàng trước khi nhận request đầu tiên.
    """
    print("🚀 Server is starting up, loading AI models...")
    load_all_models()
    print("✅ All AI models loaded successfully. Server is ready.")


@app.get("/", tags=["Status"])
def read_root():
    """Endpoint để kiểm tra API có hoạt động không."""
    return {"status": "Athena AI Therapist API is running"}

@app.post("/chat", tags=["Chat"])
def handle_chat(request: ChatRequest):
    """
    Endpoint chính để trò chuyện với Athena.
    """
    response_text = chatbot.generate_response(
        user_input=request.user_message,
        history=[msg.dict() for msg in request.history]
    )
    
    return {"response": response_text}
