# config.py
import os
from dotenv import load_dotenv

# Tải tất cả biến môi trường
load_dotenv()

# ==============================================================================
# API KEYS
# ==============================================================================
# Lấy API key cho Google Gemini từ biến môi trường
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ==============================================================================
# LOCAL MODEL CONFIGURATIONS
# ==============================================================================
# Các model này sẽ chạy trên Railway để phân tích và tiền xử lý
MODERATION_MODEL_ID = "facebook/roberta-hate-speech-dynabench-r4-target"
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL_ID = "bhadresh-savani/distilbert-base-uncased-emotion"

# ==============================================================================
# CRISIS & CONCERN PATTERNS
# ==============================================================================
# Giữ nguyên các pattern phát hiện khủng hoảng của bạn
CRISIS_PATTERNS = [
    r"\bi (want to|wanna|'m going to|gonna|will|plan to|need to) (die|kill myself|k.m.s|end it all|end my life)\b",
    r"\bi can't (go on|live|take it) (like this )?anymore\b",
]
CONCERN_PATTERNS = [
    r"\bi feel (so )?(hopeless|trapped|worthless|empty|numb)\b",
    r"\b(what's|what is) the point of (living|anything)\b",
]

# ==============================================================================
# MENTAL HEALTH RESOURCES
# ==============================================================================
# Giữ nguyên các tài nguyên sức khỏe
MENTAL_HEALTH_RESOURCES = {
    'crisis': ["**National Suicide Prevention Lifeline (US)**: Call or text 988."],
    'concern': ["**SAMHSA National Helpline (US)**: 1-800-662-HELP (4357)."],
}

print("Configuration file for Hybrid Architecture loaded successfully.")
