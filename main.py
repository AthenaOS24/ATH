# main.py
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict

# Nhập các hàm chính từ các file của bạn
from chatbot import generate_response
from models import load_all_models
from processing import combined_sentiment_analysis

# === PART 1: FASTAPI APP SETUP ===

app = FastAPI(
    title="Athena AI Therapist API (Local Inference)",
    description="An API running a local instance of the Athena virtual psychologist model.",
    version="1.0.0"
)

# Load tất cả các model khi ứng dụng khởi động
@app.on_event("startup")
def on_startup():
    print("--- Server is starting up, loading all AI models... ---")
    try:
        load_all_models()
        print("--- All models loaded successfully. Server is ready. ---")
    except Exception as e:
        print(f"--- FATAL ERROR: Could not load models on startup. ---")
        traceback.print_exc()


# === PART 2: PYDANTIC MODELS FOR REQUEST/RESPONSE ===

class HistoryItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_input: str
    history: List[HistoryItem] = Field(default_factory=list)


# === PART 3: API ENDPOINTS ===

@app.get("/", tags=["Status"])
def read_root():
    """Check if the API server is running."""
    return {"status": "Athena AI API is running with local models"}

@app.post("/chat", tags=["Chat"])
async def handle_chat(request: ChatRequest):
    """
    Handles the main chat interaction.
    Processes user input, generates a response using the local LLM,
    and returns the response along with sentiment analysis.
    """
    if not request.user_input or not request.user_input.strip():
        raise HTTPException(status_code=400, detail="User input cannot be empty.")

    try:
        # 1. Lấy lịch sử chat từ request
        # Pydantic đã chuyển đổi JSON thành đối tượng Python, nhưng hàm của bạn
        # có thể mong đợi một list of dicts. Chúng ta cần đảm bảo định dạng đúng.
        history_list_of_dicts = [item.dict() for item in request.history]

        # 2. Phân tích cảm xúc của tin nhắn người dùng
        # Chúng ta chạy lại phân tích ở đây để trả về output giống như bạn mong muốn
        sentiment, score, emotions = combined_sentiment_analysis(request.user_input)
        sentiment_result = {
            "label": sentiment,
            "score": score
        }
        # Định dạng lại emotion_result để tương thích với output cũ
        emotion_result = [[{"label": e[0], "score": e[1]} for e in emotions]]


        # 3. Tạo phản hồi AI bằng cách sử dụng logic từ chatbot.py
        # Đây là bước quan trọng nhất, gọi đến hệ thống "xịn" của bạn
        ai_response = generate_response(
            user_input=request.user_input,
            history=history_list_of_dicts
        )

        # 4. Trả về kết quả
        return {
            "response": ai_response,
            "sentiment_analysis": sentiment_result,
            "emotion_analysis": emotion_result
        }

    except Exception as e:
        print("--- [ERROR] An error occurred in the /chat endpoint! ---")
        traceback.print_exc()
        # Trả về lỗi 500 với thông tin chi tiết
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
