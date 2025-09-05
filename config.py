# config.py
import os
from dotenv import load_dotenv

# Tải biến môi trường từ file .env
load_dotenv()

# Model IDs
MODERATION_MODEL_ID = "facebook/roberta-hate-speech-dynabench-r4-target"
LLM_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL_ID = "bhadresh-savani/distilbert-base-uncased-emotion"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"

# Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")

# Crisis Patterns
CRISIS_PATTERNS = [
    r"\bi (want to|need to|am going to|will) (die|kill myself|end it all)\b",
    # ... (sao chép tất cả các pattern của bạn vào đây)
]

CONCERN_PATTERNS = [
    r"\bi've been feeling (really )?(depressed|suicidal)\b",
    # ... (sao chép tất cả các pattern của bạn vào đây)
]

MENTAL_HEALTH_RESOURCES = {
    'crisis': [
        "National Suicide Prevention Lifeline (US): 988",
        # ... (sao chép tất cả resources của bạn vào đây)
    ],
    'concern': [
        "SAMHSA Helpline (US): 1-800-662-HELP (4357)",
        # ...
    ],
    'general': [
        "Anxiety and Depression Association of America: https://adaa.org",
        # ...
    ]
}

# Cài đặt PYTORCH CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"