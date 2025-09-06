# models.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
from config import (
    HF_TOKEN, LLM_MODEL_ID, MODERATION_MODEL_ID,
    SENTIMENT_MODEL_ID, EMOTION_MODEL_ID
)

# Khởi tạo tất cả các mô hình là None
llm_pipeline = None
moderation_model = None
moderation_tokenizer = None
sentiment_analyzer = None
emotion_analyzer = None

def load_all_models():
    """
    Tải tất cả các mô hình cần thiết cho ứng dụng.
    Hàm này được gọi một lần khi server khởi động.
    """
    get_llm_pipeline()
    get_moderation_model()
    get_sentiment_analyzer()
    get_emotion_analyzer()

def get_llm_pipeline():
    """Tải và trả về text generation pipeline (chỉ tải 1 lần)."""
    global llm_pipeline
    if llm_pipeline is None:
        print("--- Đang tải mô hình LLM (Gemma)... ---")
        # Trên Railway không có GPU, nên chúng ta sẽ dùng CPU
        device_map = "auto"
        
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, token=HF_TOKEN)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            token=HF_TOKEN,
            device_map=device_map,
            torch_dtype=torch.float32 
            trust_remote_code=True
        )
        
        llm_pipeline = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=512
        )
        print("--- Mô hình LLM đã tải xong. ---")
    return llm_pipeline

def get_moderation_model():
    """Tải mô hình kiểm duyệt."""
    global moderation_model, moderation_tokenizer
    if moderation_model is None:
        print("--- Đang tải mô hình Moderation... ---")
        moderation_tokenizer = AutoTokenizer.from_pretrained(MODERATION_MODEL_ID)
        moderation_model = AutoModelForSequenceClassification.from_pretrained(MODERATION_MODEL_ID).to("cpu")
        print("--- Mô hình Moderation đã tải xong. ---")
    return moderation_model, moderation_tokenizer

def get_sentiment_analyzer():
    """Tải mô hình phân tích cảm xúc."""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        print("--- Đang tải mô hình Sentiment... ---")
        sentiment_analyzer = pipeline(
            "sentiment-analysis", model=SENTIMENT_MODEL_ID, tokenizer=SENTIMENT_MODEL_ID, device=-1 # -1 để chắc chắn dùng CPU
        )
        print("--- Mô hình Sentiment đã tải xong. ---")
    return sentiment_analyzer

def get_emotion_analyzer():
    """Tải mô hình phân tích cảm xúc."""
    global emotion_analyzer
    if emotion_analyzer is None:
        print("--- Đang tải mô hình Emotion... ---")
        emotion_analyzer = pipeline(
            "text-classification", model=EMOTION_MODEL_ID, top_k=None, device=-1 # -1 để chắc chắn dùng CPU
        )
        print("--- Mô hình Emotion đã tải xong. ---")
    return emotion_analyzer
