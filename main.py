# === PHẦN 1: IMPORT VÀ CẤU HÌNH ===
import os
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline

# Đường dẫn gốc tới thư mục chứa các model
MODEL_BASE_PATH = "models"

# === PHẦN 2: LOAD TẤT CẢ MODEL TỪ THƯ MỤC CỤC BỘ ===
print("Bắt đầu load các model...")

# Model phụ trợ 1: Sentiment Analysis
sentiment_path = os.path.join(MODEL_BASE_PATH, "sentiment")
sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_path, tokenizer=sentiment_path)

# Model phụ trợ 2: Emotion Analysis
emotion_path = os.path.join(MODEL_BASE_PATH, "emotion")
emotion_analyzer = pipeline("text-classification", model=emotion_path, tokenizer=emotion_path, top_k=None)

## THAY ĐỔI LỚN: LOAD MÔ HÌNH CHAT MỚI TỪ HUGGING FACE ##
# Model chính: DialoGPT for Conversation
chat_model_path = os.path.join(MODEL_BASE_PATH, "dialogpt-medium")
chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_path)
chat_model = AutoModelForCausalLM.from_pretrained(chat_model_path)

# Tạo một 'pipeline' để dễ dàng sử dụng mô hình chat
chat_pipeline = pipeline("conversational", model=chat_model, tokenizer=chat_tokenizer)

print("✅ Tất cả các model đã được load thành công!")

# === PHẦN 3: LOGIC XỬ LÝ ===

def sanitize_input(text):
    return text.strip()

def combined_sentiment_analysis(text):
    try:
        sentiment = sentiment_analyzer(text)[0]
        emotions = emotion_analyzer(text)[0]
        return {"sentiment": sentiment, "emotions": emotions}
    except Exception:
        return {"sentiment": "unknown", "emotions": []}

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
    
    sanitized_input = sanitize_input(request.user_input)
    if not sanitized_input:
        raise HTTPException(status_code=400, detail="User input is empty after sanitization.")

    try:
        # Chuyển đổi history sang định dạng mà pipeline hội thoại hiểu
        # Rất tiếc, pipeline này không hỗ trợ truyền `role` user/assistant
        # Chúng ta sẽ nối các tin nhắn lại với nhau
        conversation_text = ""
        for message in request.history:
            conversation_text += message.get("content", "") + chat_tokenizer.eos_token
        
        conversation_text += sanitized_input

        # Gọi pipeline chat cục bộ
        print("--- [LOG 2] Bắt đầu xử lý với mô hình chat cục bộ... ---")
        result = chat_pipeline(conversation_text)
        
        # Lấy câu trả lời cuối cùng từ pipeline
        ai_response = result.generated_responses[-1]
        print("--- [LOG 3] Xử lý cục bộ thành công. ---")

        # Phân tích cảm xúc
        print("--- [LOG 4] Bắt đầu phân tích cảm xúc... ---")
        sentiment_data = combined_sentiment_analysis(sanitized_input)
        print("--- [LOG 5] Phân tích cảm xúc thành công. ---")

        return {
            "response": ai_response,
            "sentiment_analysis": sentiment_data
        }
    except Exception as e:
        print("--- [LỖI] Đã có lỗi xảy ra trong quá trình xử lý! ---")
        print(f"Lỗi: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred. Check server logs for details.")
