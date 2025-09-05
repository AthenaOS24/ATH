# models.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
from config import (
    HF_TOKEN, LLM_MODEL_ID, MODERATION_MODEL_ID,
    SENTIMENT_MODEL_ID, EMOTION_MODEL_ID
)

print("--- Bắt đầu tải các mô hình AI ---")

# Xác định thiết bị (GPU nếu có)
device = 0 if torch.cuda.is_available() else -1
device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}

# 1. Moderation Model
moderation_tokenizer = AutoTokenizer.from_pretrained(MODERATION_MODEL_ID)
moderation_model = AutoModelForSequenceClassification.from_pretrained(MODERATION_MODEL_ID).to("cpu")

# 2. Llama-2-7b-chat-hf Model
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, token=HF_TOKEN)
llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    token=HF_TOKEN,
    device_map=device_map,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
text_generation_pipeline = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=llm_tokenizer,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=512
)

# 3. Sentiment and Emotion Analyzers
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL_ID,
    tokenizer=SENTIMENT_MODEL_ID,
    device=device
)
emotion_analyzer = pipeline(
    "text-classification",
    model=EMOTION_MODEL_ID,
    top_k=None,
    device=device
)

print("--- Tất cả mô hình đã được tải thành công ---")