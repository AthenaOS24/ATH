# === PHẦN 1: IMPORT VÀ CẤU HÌNH ===
import os
import traceback
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# Lấy API Keys từ biến môi trường
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") # Token của Hugging Face

# Cấu hình client Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# URL cho các model phân tích trên Hugging Face Inference API
SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"


# === PHẦN 2: CÁC HÀM GỌI API ===

# Hàm gọi API của Hugging Face để phân tích
async def analyze_sentiment_emotion_api(text: str, client: httpx.AsyncClient):
    if not HF_TOKEN:
        return {"error": "Hugging Face API token is not configured."}

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    json_payload = {"inputs": text}

    try:
        # Gọi cả 2 API cùng lúc để tiết kiệm thời gian
        sentiment_task = client.post(SENTIMENT_API_URL, headers=headers, json=json_payload)
        emotion_task = client.post(EMOTION_API_URL, headers=headers, json=json_payload)
        
        responses = await asyncio.gather(sentiment_task, emotion_task)
        
        sentiment_res = responses[0].json()
        emotion_res = responses[1].json()

        return {"sentiment": sentiment_res, "emotions": emotion_res}
    except Exception as e:
        print(f"Lỗi khi gọi Hugging Face API: {e}")
        return {"sentiment": "unknown", "emotions": "unknown"}

# Hàm gọi API của Gemini để chat
async def generate_response_from_gemini(user_input: str, history: list):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        gemini_history = []
        for message in history:
            role = "user" if message.get("role") == "user" else "model"
            gemini_history.append({"role": role, "parts": [{"text": message.get("content", "")}]})

        chat = model.start_chat(history=gemini_history)
        response = await chat.send_message_async(user_input)
        return response.text
    except Exception as e:
        print(f"Lỗi khi gọi API Gemini: {e}")
        raise HTTPException(status_code=503, detail="Error communicating with Gemini API.")

# === PHẦN 3: TẠO API VỚI FASTAPI ===
app = FastAPI(title="Athena AI Therapist API (API-Only)")

class ChatRequest(BaseModel):
    user_input: str
    history: list = []

@app.get("/")
def read_root():
    return {"status": "Athena AI API is running"}

@app.post("/chat")
async def handle_chat(request: ChatRequest):
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="User input is empty.")

    try:
        async with httpx.AsyncClient() as client:
            # Gọi cả Gemini và Hugging Face cùng một lúc
            chat_task = generate_response_from_gemini(request.user_input, request.history)
            analysis_task = analyze_sentiment_emotion_api(request.user_input, client)
            
            results = await asyncio.gather(chat_task, analysis_task)
            
            ai_response = results[0]
            sentiment_data = results[1]

        return {
            "response": ai_response,
            "sentiment_analysis": sentiment_data
        }
    except Exception as e:
        print("--- [LỖI] Đã có lỗi xảy ra trong endpoint /chat! ---")
        traceback.print_exc()
        raise e
