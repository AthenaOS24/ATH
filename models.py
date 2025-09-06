# models.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from config import MODERATION_MODEL_ID, SENTIMENT_MODEL_ID, EMOTION_MODEL_ID

# Khởi tạo các model cục bộ
moderation_model = None
moderation_tokenizer = None
sentiment_analyzer = None
emotion_analyzer = None

def load_local_models():
    """Tải tất cả các model phụ trợ cần thiết khi server khởi động."""
    print("--- Loading all local models for pre-processing... ---")
    get_moderation_model()
    get_sentiment_analyzer()
    get_emotion_analyzer()
    print("--- All local models loaded successfully. ---")

def get_moderation_model():
    """Tải model kiểm duyệt nội dung."""
    global moderation_model, moderation_tokenizer
    if moderation_model is None:
        print(f"--- Loading Moderation model ({MODERATION_MODEL_ID})... ---")
        moderation_tokenizer = AutoTokenizer.from_pretrained(MODERATION_MODEL_ID)
        moderation_model = AutoModelForSequenceClassification.from_pretrained(MODERATION_MODEL_ID)
    return moderation_model, moderation_tokenizer

def get_sentiment_analyzer():
    """Tải model phân tích cảm xúc."""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        print(f"--- Loading Sentiment model ({SENTIMENT_MODEL_ID})... ---")
        sentiment_analyzer = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_ID)
    return sentiment_analyzer

def get_emotion_analyzer():
    """Tải model phân tích cảm xúc chi tiết."""
    global emotion_analyzer
    if emotion_analyzer is None:
        print(f"--- Loading Emotion model ({EMOTION_MODEL_ID})... ---")
        emotion_analyzer = pipeline("text-classification", model=EMOTION_MODEL_ID, top_k=None)
    return emotion_analyzer
