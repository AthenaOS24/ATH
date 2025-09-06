# chatbot.py
import random
from models import get_llm_pipeline
from processing import (
    sanitize_input, moderate_text, enhanced_crisis_detection,
    combined_sentiment_analysis, get_time_based_greeting, recommend_resources
)

# === HÀM TẠO PROMPT GIỮ NGUYÊN ===
def format_prompt_with_context(user_input, conversation_history="", is_first_message=False, retrieved_contexts=None, urgency_level=None):
    prompt = "This is a conversation with Athena, a helpful AI assistant.\n\n"
    if conversation_history:
        prompt += conversation_history
    prompt += f"User: {user_input}\n"
    prompt += "Athena:"
    return prompt


# === VIẾT LẠI HOÀN TOÀN HÀM GENERATE_RESPONSE ===
def generate_response(user_input: str, history: list):
    """
    Hàm chính tạo phản hồi - Phiên bản tối ưu tốc độ và logic.
    """
    # 1. Các bước xử lý an toàn đầu vào (giữ nguyên)
    sanitized_input = sanitize_input(user_input)
    if moderate_text(sanitized_input)['is_harmful']:
        return "I can't engage with harmful content. Let's talk about something else."

    # 2. Tạo lịch sử hội thoại (giữ nguyên)
    conversation_text = ""
    for message in history:
        role = "User" if message["role"] == "user" else "Athena"
        conversation_text += f"{role}: {message['content']}\n"

    # 3. Tạo prompt hoàn chỉnh
    formatted_prompt = format_prompt_with_context(
        sanitized_input,
        conversation_text,
        is_first_message=not history
    )

    # 4. Gọi model với tham số tối ưu tốc độ
    pipe = get_llm_pipeline()
    output = pipe(
        formatted_prompt,
        # Giảm mạnh độ dài để phản hồi siêu nhanh
        max_new_tokens=80,  
        num_return_sequences=1,
        truncation=True,
        # Chống lặp lại từ
        repetition_penalty=1.2,
        # Ngăn model nói "User:" hoặc "Athena:"
        bad_words_ids=[[8147], [4113, 25], [34, 121, 34, 25]] 
    )

    # 5. Logic cắt chuỗi chính xác và đáng tin cậy
    full_text = output[0]['generated_text']
    
    # Cắt chuỗi dựa trên độ dài của prompt, cách này chính xác 100%
    response = full_text[len(formatted_prompt):].strip()

    # 6. Xử lý các trường hợp phản hồi bị lỗi
    # Cắt bỏ nếu model tự sinh ra vai trò "User:"
    if "User:" in response:
        response = response.split("User:")[0].strip()
    if not response:
        response = "I'm not sure how to respond. Can you rephrase?"

    # 7. Xử lý an toàn đầu ra và thêm tài nguyên (giữ nguyên)
    if moderate_text(response)['is_harmful']:
        return "My apologies, I generated a response I cannot share. Let's try another topic."
    
    urgency_level = enhanced_crisis_detection(sanitized_input)
    if urgency_level:
        resources = "\n".join(recommend_resources(urgency_level))
        response += f"\n\nThese resources might be helpful:\n{resources}"
        
    return response
