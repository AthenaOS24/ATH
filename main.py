# === PHẦN 1: IMPORT VÀ CẤU HÌNH ===
import os
import traceback
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Lấy API Key của Hugging Face từ biến môi trường
HF_TOKEN = os.getenv("HF_TOKEN")

# URL cho các model trên Hugging Face Inference API
CHAT_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"


# === PHẦN 2: CÁC HÀM GỌI API ===

# Helper function to query the HF API
async def query_hf_api(api_url: str, payload: dict, client: httpx.AsyncClient):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN is not configured on the server.")
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        response = await client.post(api_url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        print(f"Lỗi API từ Hugging Face ({api_url}): {e.response.text}")
        # Return a structured error to avoid crashing the whole request
        return {"error": f"API Error: {e.response.status_code}", "detail": e.response.text}
    except Exception as e:
        print(f"Lỗi không xác định khi gọi {api_url}: {e}")
        return {"error": "Unknown error during API call."}


# === PHẦN 3: TẠO API VỚI FASTAPI ===
app = FastAPI(title="Athena AI Therapist API (Hugging Face-Only)")

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
        # Build the prompt for the chat model
        prompt = ""
        for message in request.history:
            role = "User" if message.get("role") == "user" else "Assistant"
            prompt += f"{role}: {message.get('content', '')}\n"
        prompt += f"User: {request.user_input}\nAssistant:"

        # Define payloads for all API calls
        chat_payload = {"inputs": prompt, "parameters": {"max_new_tokens": 250}}
        analysis_payload = {"inputs": request.user_input}

        async with httpx.AsyncClient() as client:
            # Run all three API calls concurrently
            chat_task = query_hf_api(CHAT_API_URL, chat_payload, client)
            sentiment_task = query_hf_api(SENTIMENT_API_URL, analysis_payload, client)
            emotion_task = query_hf_api(EMOTION_API_URL, analysis_payload, client)
            
            results = await asyncio.gather(chat_task, sentiment_task, emotion_task)
            
            chat_result = results[0]
            sentiment_result = results[1]
            emotion_result = results[2]

        # Process chat response to get only the new text
        ai_response = chat_result[0].get('generated_text', '').replace(prompt, '').strip()

        return {
            "response": ai_response,
            "sentiment_analysis": sentiment_result,
            "emotion_analysis": emotion_result
        }
    except Exception as e:
        print("--- [LỖI] Đã có lỗi xảy ra trong endpoint /chat! ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
