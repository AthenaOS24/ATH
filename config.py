# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if available)
load_dotenv()

# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================

# Model IDs from Hugging Face
MODERATION_MODEL_ID = "facebook/roberta-hate-speech-dynabench-r4-target"

## ================== CẢI TIẾN QUAN TRỌNG NHẤT ================== ##
# Đổi sang model 'distilgpt2' nhỏ và nhanh hơn rất nhiều.
# Model này phù hợp với môi trường tài nguyên hạn chế như Railway free-tier,
# giúp ứng dụng phản hồi nhanh chóng và không bị treo.
LLM_MODEL_ID = "distilgpt2"
## ============================================================= ##

SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL_ID = "bhadresh-savani/distilbert-base-uncased-emotion"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"

# Hugging Face Token (retrieved from environment variables)
HF_TOKEN = os.getenv("HF_TOKEN")


# ==============================================================================
# CRISIS DETECTION PATTERNS
# ==============================================================================
# These patterns indicate immediate risk, planned actions, or explicit intent.

CRISIS_PATTERNS = [
    # --- Direct and explicit intentions ---
    r"\bi (want to|wanna|'m going to|gonna|will|plan to|need to) (die|kill myself|k.m.s|end it all|end my life)\b",
    r"\bi can't (go on|live|take it) (like this )?anymore\b",
    r"\b(i'm|i am) (seriously|really) (thinking of|considering) suicide\b",
    r"\b(goodbye|bye bye) (cruel )?(world|everyone)\b",
    r"\bthis is my last (message|post|day|night)\b",
    r"\bi've decided to end my life\b",

    # --- Specific methods ---
    r"\b(i'm|i am) going to (jump|hang|overdose|o.d.|shoot|cut myself)\b",
    r"\b(planning to|gonna) (jump off a bridge|hang myself|take all my pills)\b",
    r"\bi (have|got) a (gun|rope|blade|pills) and i'm going to use it\b",

    # --- Actions already taken or happening now ---
    r"\b(i'm|i am) (cutting|bleeding) myself (right now|currently)\b",
    r"\b(i've|i have) (already )?taken (pills|medication|the whole bottle) to end my life\b",
    r"\bi (just|already) (cut myself|tried to hang myself|jumped)\b",
    r"\b(i'm|i am) being (abused|raped|assaulted) (right now|currently)\b",

    # --- Farewell statements ---
    r"\b(there's|there is) no other way out\b",
    r"\beveryone would be better off without me\b",
    r"\bi have a plan to kill myself\b",
    r"\btell my (family|mom|dad) i love them\b"
]


# ==============================================================================
# CONCERN DETECTION PATTERNS
# ==============================================================================
# These patterns indicate suicidal thoughts, despair, or self-harm intentions
# but without immediate risk.

CONCERN_PATTERNS = [
    # --- Feelings of hopelessness and worthlessness ---
    r"\bi feel (so )?(hopeless|trapped|worthless|empty|numb)\b",
    r"\b(what's|what is) the point of (living|anything)\b",
    r"\b(i have|i've got|there's|there is) no (reason|point) to live\b",
    r"\bi (just )?don't want to be here anymore\b",
    r"\bi wish i (was dead|was never born|could disappear)\b",
    r"\bmy life is (meaningless|a mess|not worth living)\b",

    # --- Feeling like a burden and isolation ---
    r"\b(i'm|i am) (such )?a burden (to everyone)?\b",
    r"\bno one (cares|would miss me|understands)\b",
    r"\bi feel (so|completely) alone\b",
    r"\bi'm better off dead\b",

    # --- Non-specific suicidal intentions ---
    r"\b(i've|i have) been feeling (really )?(depressed|suicidal)\b",
    r"\bi (sometimes|often|can't stop) think(ing)? about (dying|ending it|self-harm)\b",
    r"\b(i'm|i am) struggling with (suicidal thoughts|self-harm urges)\b",
    r"\bthe pain is (unbearable|too much)\b",

    # --- Self-harm behavior (non-immediate) ---
    r"\bi (self-harm|self harm|s.h.|hurt myself|cut myself)\b",
    r"\b(i want to|i feel like) (cutting|hurting myself)\b",
    r"\bthe urge to (cut|self harm) is (so strong|back)\b"
]


# ==============================================================================
# MENTAL HEALTH RESOURCES
# ==============================================================================

MENTAL_HEALTH_RESOURCES = {
    'crisis': [
        "**Emergency Services**: 911 (US/Canada), 112 (Europe), 000 (Australia), or your local emergency number.",
        "**National Suicide Prevention Lifeline (US)**: Call or text 988.",
        "**Crisis Text Line**: Text HOME to 741741 (US/Canada) or 85258 (UK).",
        "**Samaritans (UK)**: Call 116 123.",
        "**Lifeline (Australia)**: Call 13 11 14.",
        "**IASP Crisis Centres**: Find a center near you at https://www.iasp.info/resources/Crisis_Centres/"
    ],
    'concern': [
        "**SAMHSA National Helpline (US)**: 1-800-662-HELP (4357).",
        "**NAMI Helpline (US)**: 1-800-950-NAMI (6264).",
        "**7 Cups**: Free online peer support at https://www.7cups.com",
        "**Warmline**: Find a peer-run listening line at https://warmline.org/warmdir.html#directory"
    ],
    'general': [
        "**Psychology Today Therapist Finder**: https://www.psychologytoday.com/therapists",
        "**BetterHelp Online Therapy**: https://www.betterhelp.com",
        "**TalkSpace Online Therapy**: https://www.talkspace.com",
        "**Anxiety & Depression Association of America (ADAA)**: https://adaa.org",
        "**Mind (UK mental health charity)**: https://www.mind.org.uk",
        "**Beyond Blue (Australia)**: https://www.beyondblue.org.au"
    ]
}


# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================

# Configure PyTorch to optimize GPU memory usage (has no effect on CPU, but is safe to keep)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

print("Configuration file loaded successfully with optimized LLM_MODEL_ID for performance.")
