# main.py
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import google.generativeai as genai

# Import các thành phần từ các file khác
from config import GOOGLE_API_KEY
from models import load_local_models
from processing import sanitize_input, moderate_text, combined_sentiment_analysis

# --- Cấu hình và Khởi động ---
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
except Exception as e:
    print(f"Lỗi khởi tạo Google Gemini client: {e}")
    gemini_model = None

app = FastAPI(
    title="Athena AI Therapist API (Hybrid Architecture)",
    version="5.0.0"
)

@app.on_event("startup")
def startup_event():
    # Khi server khởi động, tải tất cả các model cục bộ
    load_local_models()

# --- Pydantic Models ---
class HistoryItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_input: str
    history: List[HistoryItem] = Field(default_factory=list)

# --- API Endpoints ---
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "Athena AI API (Hybrid mode) is running"}

@app.post("/chat", tags=["Chat"])
async def handle_chat(request: ChatRequest):
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini client not initialized.")
    
    # === BƯỚC 1: TIỀN XỬ LÝ & PHÂN TÍCH CỤC BỘ (trên Railway) ===
    sanitized_input = sanitize_input(request.user_input)
    
    if moderate_text(sanitized_input)['is_harmful']:
        raise HTTPException(status_code=400, detail="Input contains harmful content.")
        
    sentiment_label, sentiment_score, emotions = combined_sentiment_analysis(sanitized_input)

    # === BƯỚC 2: GỬI YÊU CẦU ĐÃ ĐƯỢC LÀM GIÀU TỚI GEMINI ===
    system_prompt = f"""
You are Athena, a virtual psychologist using Cognitive Behavioral Therapy (CBT).
You are empathetic and supportive.
The user's current sentiment has been analyzed as: {sentiment_label} (confidence: {sentiment_score:.2f}).
Adjust your tone accordingly. For negative sentiment, be extra supportive.
"""
    
    gemini_history = [{'role': 'user', 'parts': [{'text': system_prompt}]},
                      {'role': 'model', 'parts': [{'text': "I understand. I am Athena, ready to listen."}]}]
    
    for message in request.history:
        role = 'model' if message.role == 'assistant' else 'user'
        gemini_history.append({'role': role, 'parts': [{'text': message.content}]})

    try:
        chat_session = gemini_model.start_chat(history=gemini_history)
        response = chat_session.send_message(sanitized_input)
        ai_response = response.text.strip()

        # === BƯỚC 3: TRẢ VỀ KẾT QUẢ KẾT HỢP ===
        return {
            "response": ai_response,
            "sentiment_analysis": {"label": sentiment_label, "score": sentiment_score},
            "emotion_analysis": [{"label": e['label'], "score": e['score']} for e in emotions]
        }

    except Exception as e:
        if "response was blocked" in str(e):
            raise HTTPException(status_code=400, detail="Response blocked by safety filters.")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
