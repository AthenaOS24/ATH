import re
import torch
from datetime import datetime
from models import get_moderation_model, get_sentiment_analyzer, get_emotion_analyzer
from config import CRISIS_PATTERNS, CONCERN_PATTERNS, MENTAL_HEALTH_RESOURCES

# ==============================================================================
# SAFETY & MODERATION FUNCTIONS
# ==============================================================================

def moderate_text(text):
    """Moderates text to detect harmful content."""
    mod_model, mod_tokenizer = get_moderation_model()
    inputs = mod_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(mod_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = mod_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    harmful_score = probs[0, 1].item()
    return {'is_harmful': harmful_score > 0.7, 'score': harmful_score}

def sanitize_input(text):
    """Sanitizes input to remove malicious code and limit length."""
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s.,!?\'-]', '', text)
    max_length = 1000
    if len(text) > max_length:
        text = text[:max_length] + "... [truncated]"
    return text.strip()

def anonymize_text(text):
    """Anonymizes sensitive personal information."""
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})', '[PHONE]', text)
    text = re.sub(r'\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}', '[CREDIT_CARD]', text)
    text = re.sub(r'\d{3}-\d{2}-\d{4}', '[SSN]', text)
    return text

def enhanced_crisis_detection(text):
    """Detects crisis or concern based on regex patterns."""
    text_lower = text.lower()
    if any(re.search(pattern, text_lower) for pattern in CRISIS_PATTERNS):
        return "crisis"
    if any(re.search(pattern, text_lower) for pattern in CONCERN_PATTERNS):
        return "concern"
    return None

# ==============================================================================
# SENTIMENT & EMOTION ANALYSIS
# ==============================================================================

def analyze_sentiment(text):
    """Analyzes sentiment (positive, negative, neutral)."""
    try:
        analyzer = get_sentiment_analyzer()
        result = analyzer(text)[0]
        label = result['label'].lower()
        score = result['score']
        return label, score
    except Exception:
        return "neutral", 0.5

def analyze_emotions(text, top_n=3):
    """Detects the top emotions in the text."""
    try:
        analyzer = get_emotion_analyzer()
        results = analyzer(text)[0]
        top_emotions = sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]
        return [(emo['label'], emo['score']) for emo in top_emotions]
    except Exception:
        return [("unknown", 0.5)]

def combined_sentiment_analysis(text):
    """Combines crisis detection with sentiment analysis."""
    urgency = enhanced_crisis_detection(text)
    if urgency:
        return urgency, 1.0, [(f"{urgency}_detected", 1.0)]
    
    sentiment, sent_score = analyze_sentiment(text)
    emotions = analyze_emotions(text)
    
    if any(emo[0] in ['sadness', 'anger', 'fear'] and emo[1] > 0.7 for emo in emotions):
        sentiment = 'negative'
    
    return sentiment, sent_score, emotions

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_time_based_greeting():
    """Generates a greeting based on the time of day."""
    hour = datetime.now().hour
    if 5 <= hour < 12: return "Good morning"
    if 12 <= hour < 17: return "Good afternoon"
    if 17 <= hour < 22: return "Good evening"
    return "Hello"

def recommend_resources(urgency_level):
    """Recommends resources based on the urgency level."""
    resources = MENTAL_HEALTH_RESOURCES.get(urgency_level, [])
    resources.extend(MENTAL_HEALTH_RESOURCES['general'])
    return list(set(resources))
