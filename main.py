# main.py
import os
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai

# --- Cấu hình ---
load_dotenv()
try:
    # Thư viện sẽ tự động đọc key từ biến môi trường GOOGLE_API_KEY
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
except Exception as e:
    print(f"Lỗi khởi tạo Google Gemini client: {e}")
    model = None

# --- FastAPI App Setup ---
app = FastAPI(
    title="Athena AI Therapist API (via Google Gemini)",
    version="4.0.0"
)

# --- Pydantic Models ---
class HistoryItem(BaseModel):
    role: str # 'user' hoặc 'model'
    content: str

class ChatRequest(BaseModel):
    user_input: str
    history: List[HistoryItem] = Field(default_factory=list)

# --- API Endpoints ---
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "Athena AI API (Gemini mode) is running"}

@app.post("/chat", tags=["Chat"])
async def handle_chat(request: ChatRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Google Gemini client not initialized. Check GOOGLE_API_KEY.")
    if not request.user_input or not request.user_input.strip():
        raise HTTPException(status_code=400, detail="User input cannot be empty.")

    # 1. Định dạng lại lịch sử cho đúng chuẩn của Gemini
    # vai trò của assistant trong Gemini được gọi là 'model'
    gemini_history = []
    for message in request.history:
        # Đảm bảo vai trò là 'user' hoặc 'model'
        role = 'model' if message.role == 'assistant' else 'user'
        gemini_history.append({'role': role, 'parts': [{'text': message.content}]})

    # 2. Bắt đầu cuộc trò chuyện và gửi tin nhắn mới
    try:
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(request.user_input)
        
        ai_response = response.text.strip()

        return {"response": ai_response}

    except Exception as e:
        # Đây là cách xử lý lỗi an toàn của Gemini
        if "response was blocked" in str(e):
            print(f"Safety filter blocked the response: {e}")
            raise HTTPException(status_code=400, detail="The response was blocked by Google's safety filters. Please rephrase your input.")
        
        print("--- [ERROR] An error occurred in the /chat endpoint! ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
