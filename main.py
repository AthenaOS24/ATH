# === PHẦN 1: IMPORT VÀ CẤU HÌNH ===
import os
import httpx
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Lấy API Key từ biến môi trường
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Đường dẫn gốc tới thư mục chứa các model nhỏ
MODEL_BASE_PATH = "models"


# === PHẦN 2: LOAD CÁC MODEL NHỎ TỪ THƯ MỤC CỤC BỘ ===
print("Bắt đầu load các model phụ trợ...")

# Model 1: Moderation
moderation_path = os.path.join(MODEL_BASE_PATH, "moderation")
moderation_tokenizer = AutoTokenizer.from_pretrained(moderation_path)
moderation_model = AutoModelForSequenceClassification.from_pretrained(moderation_path)

# Model 2: Sentiment Analysis
sentiment_path = os.path.join(MODEL_BASE_PATH, "sentiment")
sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_path, tokenizer=sentiment_path)

# Model 3: Emotion Analysis
emotion_path = os.path.join(MODEL_BASE_PATH, "emotion")
emotion_analyzer = pipeline("text-classification", model=emotion_path, tokenizer=emotion_path, top_k=None)

print("✅ Tất cả các model phụ trợ đã được load thành công!")


# === PHẦN 3: LOGIC GỌI API VÀ XỬ LÝ ===

def sanitize_input(text):
    return text.strip()

def combined_sentiment_analysis(text):
    try:
        sentiment = sentiment_analyzer(text)[0]
        emotions = emotion_analyzer(text)[0]
        return {"sentiment": sentiment, "emotions": emotions}
    except Exception:
        return {"sentiment": "unknown", "emotions": []}

async def generate_response_from_openrouter(user_input: str, history: list):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set.")

    messages = [{"role": "system", "content": "You are Athena, a compassionate AI therapist."}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    json_data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": messages
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=json_data,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except httpx.HTTPStatusError as e:
            print(f"Lỗi API từ OpenRouter: {e.response.text}")
            raise HTTPException(status_code=502, detail="Error response from AI service.")
        except Exception as e:
            print(f"Lỗi không xác định khi gọi API: {e}")
            raise HTTPException(status_code=500, detail="An internal error occurred.")


# === PHẦN 4: TẠO API VỚI FASTAPI ===
app = FastAPI(title="Athena AI Therapist API")

class ChatRequest(BaseModel):
    user_input: str
    history: list = []

@app.get("/")
def read_root():
    return {"status": "Athena AI API is running"}

@app.post("/chat")
async def handle_chat(request: ChatRequest):
    print("--- [LOG 1] Đã nhận được yêu cầu /chat mới. ---")
    
    ## SỬA LỖI Ở ĐÂY ##
    # Tên thuộc tính đúng là 'request.user_input', không phải 'request.user_message'
    print(f"--- [LOG 2] Nội dung tin nhắn: '{request.user_input}' ---")
    
    sanitized_input = sanitize_input(request.user_input)
    if not sanitized_input:
        raise HTTPException(status_code=400, detail="User input is empty after sanitization.")

    try:
        print("--- [LOG 3] Bắt đầu gọi API OpenRouter... ---")
        ai_response = await generate_response_from_openrouter(sanitized_input, request.history)
        print("--- [LOG 4] Gọi API OpenRouter thành công. ---")

        print("--- [LOG 5] Bắt đầu phân tích cảm xúc... ---")
        sentiment_data = combined_sentiment_analysis(sanitized_input)
        print("--- [LOG 6] Phân tích cảm xúc thành công. ---")

        return {
            "response": ai_response,
            "sentiment_analysis": sentiment_data
        }
    except Exception as e:
        print("--- [LỖI] Đã có lỗi xảy ra trong quá trình xử lý! ---")
        print(f"Lỗi: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred. Check server logs for details.")
