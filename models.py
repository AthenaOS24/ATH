import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
from config import (
    HF_TOKEN, LLM_MODEL_ID, MODERATION_MODEL_ID,
    SENTIMENT_MODEL_ID, EMOTION_MODEL_ID
)

# Initialize all models to None
llm_pipeline = None
moderation_model = None
moderation_tokenizer = None
sentiment_analyzer = None
emotion_analyzer = None

def load_all_models():
    """
    Loads all models required for the application.
    This function is called once when the server starts.
    """
    get_llm_pipeline()
    get_moderation_model()
    get_sentiment_analyzer()
    get_emotion_analyzer()

def get_llm_pipeline():
    """Loads and returns the text generation pipeline (loads only once)."""
    global llm_pipeline
    if llm_pipeline is None:
        print(f"--- Loading LLM model ({LLM_MODEL_ID})... ---")
        device_map = "auto"
        
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, token=HF_TOKEN)
        
        # Some tokenizers don't have a default pad_token, assign it to eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            token=HF_TOKEN,
            device_map=device_map,
            torch_dtype=torch.float32,
            trust_remote_code=True  # Important: Allows loading custom models
        )
        
        llm_pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            do_sample=True, 
            temperature=0.7, 
            top_p=0.9, 
            max_new_tokens=512
        )
        print("--- LLM model loaded successfully. ---")
    return llm_pipeline

def get_moderation_model():
    """Loads the moderation model."""
    global moderation_model, moderation_tokenizer
    if moderation_model is None:
        print("--- Loading Moderation model... ---")
        moderation_tokenizer = AutoTokenizer.from_pretrained(MODERATION_MODEL_ID)
        moderation_model = AutoModelForSequenceClassification.from_pretrained(MODERATION_MODEL_ID).to("cpu")
        print("--- Moderation model loaded successfully. ---")
    return moderation_model, moderation_tokenizer

def get_sentiment_analyzer():
    """Loads the sentiment analysis model."""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        print("--- Loading Sentiment model... ---")
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model=SENTIMENT_MODEL_ID, 
            tokenizer=SENTIMENT_MODEL_ID, 
            device=-1 # -1 to ensure CPU usage
        )
        print("--- Sentiment model loaded successfully. ---")
    return sentiment_analyzer

def get_emotion_analyzer():
    """Loads the emotion analysis model."""
    global emotion_analyzer
    if emotion_analyzer is None:
        print("--- Loading Emotion model... ---")
        emotion_analyzer = pipeline(
            "text-classification", 
            model=EMOTION_MODEL_ID, 
            top_k=None, 
            device=-1 # -1 to ensure CPU usage
        )
        print("--- Emotion model loaded successfully. ---")
    return emotion_analyzer
