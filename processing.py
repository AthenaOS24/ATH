# processing.py
import re
import torch
from models import get_moderation_model, get_sentiment_analyzer, get_emotion_analyzer
from config import CRISIS_PATTERNS, CONCERN_PATTERNS, MENTAL_HEALTH_RESOURCES

def sanitize_input(text):
    """Sanitizes input to remove malicious code and limit length."""
    # Giữ nguyên code sanitize của bạn
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()[:1000]

def moderate_text(text):
    """Moderates text to detect harmful content using local model."""
    mod_model, mod_tokenizer = get_moderation_model()
    inputs = mod_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = mod_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # Giả sử label 1 là 'harmful'
    harmful_score = probs[0, 1].item()
    return {'is_harmful': harmful_score > 0.8, 'score': harmful_score}

def enhanced_crisis_detection(text):
    """Detects crisis or concern based on regex patterns."""
    text_lower = text.lower()
    if any(re.search(pattern, text_lower) for pattern in CRISIS_PATTERNS):
        return "crisis"
    if any(re.search(pattern, text_lower) for pattern in CONCERN_PATTERNS):
        return "concern"
    return None

def combined_sentiment_analysis(text):
    """Analyzes sentiment and emotions using local models."""
    try:
        sentiment_analyzer = get_sentiment_analyzer()
        sentiment_result = sentiment_analyzer(text)[0]
        sentiment_label = sentiment_result['label']
        sentiment_score = sentiment_result['score']

        emotion_analyzer = get_emotion_analyzer()
        emotion_result = emotion_analyzer(text)[0] # Lấy top_k=None trả về list of lists

        return sentiment_label, sentiment_score, emotion_result
    except Exception as e:
        print(f"Error in combined_sentiment_analysis: {e}")
        return "unknown", 0.0, []
