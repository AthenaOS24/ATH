# main.py
import os
import httpx
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv

# Load biến môi trường (HF_TOKEN)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# API URL của model bạn đã chọn
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# === FASTAPI APP SETUP ===
app = FastAPI(
    title="Athena AI Therapist API (via Inference API)",
    description="An API that connects to a powerful, dedicated Hugging Face Inference API.",
    version="2.0.0"
)

# === PYDANTIC MODELS ===
class HistoryItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_input: str
    history: List[HistoryItem] = Field(default_factory=list)

# === API HELPER FUNCTION ===
async def query_hf_api(payload: dict):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN is not configured on the server.")
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(API_URL, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status() # Sẽ báo lỗi nếu status code là 4xx hoặc 5xx
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"API Error from Hugging Face: {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Hugging Face API Error: {e.response.text}")
        except httpx.RequestError as e:
            print(f"Request failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to connect to Hugging Face API: {e}")

# === API ENDPOINTS ===
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "Athena AI API (Inference API mode) is running"}

@app.post("/chat", tags=["Chat"])
async def handle_chat(request: ChatRequest):
    if not request.user_input or not request.user_input.strip():
        raise HTTPException(status_code=400, detail="User input cannot be empty.")

    # 1. Định dạng prompt theo chuẩn của model Mistral
    # Chúng ta sẽ tạo một chuỗi prompt hoàn chỉnh để gửi đi
    prompt = "<s>[INST] You are Athena, a compassionate AI therapist. Always respond with empathy and support. [/INST]\n"
    for message in request.history:
        if message.role == 'user':
            prompt += f"<s>[INST] {message.content} [/INST]\n"
        else:
            prompt += f"{message.content}\n" # Phản hồi của assistant không cần tag
    
    prompt += f"<s>[INST] {request.user_input} [/INST]"

    # 2. Tạo payload để gửi đến API
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "repetition_penalty": 1.2,
            "return_full_text": False, # Rất quan trọng: Chỉ trả về phần text mới
        }
    }

    # 3. Gọi API và nhận kết quả
    try:
        api_result = await query_hf_api(payload)
        
        # API trả về một list, chúng ta lấy phần tử đầu tiên
        ai_response = api_result[0].get('generated_text', 'Sorry, I could not generate a response.').strip()
        
        # Chúng ta có thể thêm lại phần phân tích cảm xúc ở đây nếu muốn
        # (Sử dụng code local từ file processing.py hoặc gọi API khác)
        
        return {
            "response": ai_response,
            # Các trường analysis có thể tạm bỏ trống hoặc thêm sau
            "sentiment_analysis": {}, 
            "emotion_analysis": {}
        }

    except Exception as e:
        print("--- [ERROR] An error occurred in the /chat endpoint! ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
