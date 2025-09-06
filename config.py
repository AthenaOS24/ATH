# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================
MODERATION_MODEL_ID = "facebook/roberta-hate-speech-dynabench-r4-target"

## ================== CẢI TIẾN TỐC ĐỘ TRIỆT ĐỂ ================== ##
# Đổi sang model 'gpt2'. Đây là một trong những model nhỏ và nhanh nhất,
# sẽ giảm đáng kể thời gian phản hồi.
LLM_MODEL_ID = "gpt2"
## ============================================================ ##

SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL_ID = "bhadresh-savani/distilbert-base-uncased-emotion"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"

HF_TOKEN = os.getenv("HF_TOKEN")

# ==============================================================================
# CRISIS & CONCERN PATTERNS (Giữ nguyên không thay đổi)
# ==============================================================================
CRISIS_PATTERNS = [
    r"\bi (want to|wanna|'m going to|gonna|will|plan to|need to) (die|kill myself|k.m.s|end it all|end my life)\b",
    r"\bi can't (go on|live|take it) (like this )?anymore\b",
    r"\b(i'm|i am) (seriously|really) (thinking of|considering) suicide\b",
    r"\b(goodbye|bye bye) (cruel )?(world|everyone)\b",
]
CONCERN_PATTERNS = [
    r"\bi feel (so )?(hopeless|trapped|worthless|empty|numb)\b",
    r"\b(what's|what is) the point of (living|anything)\b",
    r"\bi (just )?don't want to be here anymore\b",
    r"\bi wish i (was dead|was never born|could disappear)\b",
]

# ==============================================================================
# MENTAL HEALTH RESOURCES (Giữ nguyên không thay đổi)
# ==============================================================================
MENTAL_HEALTH_RESOURCES = {
    'crisis': [
        "**Emergency Services**: 911 (US/Canada), 112 (Europe), 000 (Australia), or your local emergency number.",
        "**National Suicide Prevention Lifeline (US)**: Call or text 988.",
        "**Crisis Text Line**: Text HOME to 741741 (US/Canada) or 85258 (UK).",
    ],
    'concern': [
        "**SAMHSA National Helpline (US)**: 1-800-662-HELP (4357).",
        "**NAMI Helpline (US)**: 1-800-950-NAMI (6264).",
    ],
    'general': [
        "**Psychology Today Therapist Finder**: https://www.psychologytoday.com/therapists",
        "**BetterHelp Online Therapy**: https://www.betterhelp.com",
    ]
}

# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
print("Configuration file loaded with performance-optimized model (gpt2).")
