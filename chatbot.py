# chatbot.py
import random
from models import get_llm_pipeline
from processing import (
    sanitize_input, moderate_text, enhanced_crisis_detection,
    combined_sentiment_analysis, get_time_based_greeting, recommend_resources
)

def format_prompt_with_context(user_input, conversation_history="", is_first_message=False, retrieved_contexts=None, urgency_level=None):
    """
    Tạo một prompt đơn giản, dễ hiểu cho các model cơ bản như distilgpt2.
    """
    
    # Bỏ qua các phần phức tạp như greeting, sentiment vì model cơ bản không tận dụng được
    # và chỉ tập trung vào việc tạo ra một cuộc hội thoại đơn giản.
    
    # Bắt đầu prompt với một chỉ dẫn ngắn gọn
    prompt = "This is a conversation with Athena, a helpful AI assistant.\n\n"
    
    # Thêm lịch sử hội thoại nếu có
    if conversation_history:
        prompt += conversation_history
        
    # Thêm tin nhắn mới của người dùng
    prompt += f"User: {user_input}\n"
    
    # Yêu cầu model đóng vai Athena và trả lời
    prompt += "Athena:"
    
    return prompt


def generate_response(user_input: str, history: list):
    """
    Main function to generate a response from the chatbot.
    """
    sanitized_input = sanitize_input(user_input)
    
    input_moderation = moderate_text(sanitized_input)
    if input_moderation['is_harmful']:
        return "I'm sorry, but I can't engage with harmful content. Let's focus on positive and constructive topics."

    urgency_level = enhanced_crisis_detection(sanitized_input)

    conversation_text = ""
    for message in history:
        if message["role"] == "user":
            conversation_text += f"User: {message['content']}\n"
        else:
            conversation_text += f"Athena: {message['content']}\n"

    is_first_message = not history
    
    # Get the pipeline, will automatically load if not already
    pipe = get_llm_pipeline()
    llm_tokenizer = pipe.tokenizer
    
    formatted_prompt = format_prompt_with_context(
        sanitized_input,
        conversation_text,
        is_first_message=is_first_message,
        urgency_level=urgency_level
    )
    
    output = pipe(
        formatted_prompt,
        max_new_tokens=256, 
        num_return_sequences=1,
        truncation=True,
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
            response += f"\nI'm very concerned about your safety. Please consider reaching out to:\n{resources}"
        else:
            response += f"\nThese resources might be helpful:\n{resources}"
            
    return response
