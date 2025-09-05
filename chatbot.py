# chatbot.py
import random
from models import text_generation_pipeline, llm_tokenizer
from processing import (
    sanitize_input, moderate_text, enhanced_crisis_detection,
    combined_sentiment_analysis, get_time_based_greeting, recommend_resources
)

# Lưu ý: Chúng ta không cần RAG (retriever) cho phiên bản API ban đầu
# để đơn giản hóa. Bạn có thể thêm lại sau.

def format_prompt_with_context(user_input, conversation_history="", is_first_message=False, urgency_level=None):
    # ... (Sao chép hàm format_prompt_with_context của bạn vào đây)
    # Thay đổi nhỏ: nhận conversation_history là một chuỗi string
    greeting = get_time_based_greeting()
    # ... (phần còn lại của hàm)
    return random.choice(prompt_templates)

def generate_response(user_input: str, history: list):
    """
    Hàm chính để tạo phản hồi từ chatbot.
    `history` là một danh sách các tin nhắn, ví dụ:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    sanitized_input = sanitize_input(user_input)
    
    input_moderation = moderate_text(sanitized_input)
    if input_moderation['is_harmful']:
        return "I'm sorry, but I can't engage with harmful content. Let's focus on positive and constructive topics."

    urgency_level = enhanced_crisis_detection(sanitized_input)

    # Chuyển đổi history từ list of dicts thành string
    conversation_text = ""
    for message in history:
        if message["role"] == "user":
            conversation_text += f"User: {message['content']}\n"
        else:
            conversation_text += f"Athena: {message['content']}\n"

    is_first_message = not history
    
    formatted_prompt = format_prompt_with_context(
        sanitized_input,
        conversation_text,
        is_first_message=is_first_message,
        urgency_level=urgency_level
    )
    
    output = text_generation_pipeline(
        formatted_prompt,
        max_length=len(llm_tokenizer.encode(formatted_prompt)) + 512,
        num_return_sequences=1
    )
    
    response = output[0]['generated_text']
    response_start = response.find("[/INST]") + len("[/INST]")
    if response_start > len("[/INST]") - 1:
        response = response[response_start:].strip()

    output_moderation = moderate_text(response)
    if output_moderation['is_harmful']:
        return "I apologize, but I can't provide a helpful response to that. Would you like to talk about something else?"

    if urgency_level:
        resources = "\n".join(recommend_resources(urgency_level))
        if urgency_level == "crisis":
            response += f"\n\n🔴 I'm very concerned about your safety. Please consider reaching out to:\n{resources}"
        else:
            response += f"\n\n🟡 These resources might be helpful:\n{resources}"
            
    return response