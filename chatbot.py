# chatbot.py
import random
from models import get_llm_pipeline
from processing import (
    sanitize_input, moderate_text, enhanced_crisis_detection,
    combined_sentiment_analysis, get_time_based_greeting, recommend_resources
)

def format_prompt_with_context(user_input, conversation_history="", is_first_message=False, retrieved_contexts=None, urgency_level=None):
    greeting = get_time_based_greeting()
    intro_instruction = f"""
    You are starting a new conversation. Begin by introducing yourself as:
    "{greeting}, I'm Athena, your AI therapist."
    Then respond to the user's first message with empathy and understanding.
    """ if is_first_message else ""
    context_text = ""
    if retrieved_contexts and len(retrieved_contexts) > 0:
        context_text = "RETRIEVED EXAMPLES (use these as a guide for your response):\n"
        for i, context in enumerate(retrieved_contexts):
            context_text += f"Example {i+1}: {context.strip()}\n\n"
    urgency_alert = ""
    if urgency_level == "crisis":
        urgency_alert = "\n\nCRISIS ALERT: The user has expressed immediate dangerous thoughts. Provide compassionate support while strongly recommending professional help."
    elif urgency_level == "concern":
        urgency_alert = "\n\nCONCERN ALERT: The user has expressed serious but not immediate concerns. Provide supportive listening and suggest resources."
    sentiment, score, emotions = combined_sentiment_analysis(user_input)
    emo_text = ", ".join([f"{e[0]} ({e[1]:.2f})" for e in emotions])
    sentiment_text = f"\n\nUSER SENTIMENT: {sentiment} (confidence: {score:.2f}). Emotions: {emo_text}. Adjust your tone accordingly - be more supportive for negative sentiment."
    
    prompt_templates = [
        f"""<s>[INST] <<SYS>>
        You are Athena, a compassionate AI therapist. Reflect on the user's emotions and provide a supportive response, focusing on validating their feelings.
        Avoid repeating phrases from the conversation history.
        {intro_instruction}
        {context_text}
        {urgency_alert}
        {sentiment_text}
        Previous conversation: {conversation_history}
        <</SYS>>
        User: {user_input} [/INST]""",
        f"""<s>[INST] <<SYS>>
        You are Athena, a caring AI therapist. Ask an open-ended question to explore the user's situation further, then provide a supportive suggestion.
        Avoid repeating phrases from the conversation history.
        {intro_instruction}
        {context_text}
        {urgency_alert}
        {sentiment_text}
        Previous conversation: {conversation_history}
        <</SYS>>
        User: {user_input} [/INST]""",
        f"""<s>[INST] <<SYS>>
        You are Athena, a warm AI therapist. Share a coping strategy tailored to the user's input, followed by an empathetic acknowledgment.
        Avoid repeating phrases from the conversation history.
        {intro_instruction}
        {context_text}
        {urgency_alert}
        {sentiment_text}
        Previous conversation: {conversation_history}
        <</SYS>>
        User: {user_input} [/INST]"""
    ]
    
    return random.choice(prompt_templates)


def generate_response(user_input: str, history: list):
    """
    Hàm chính để tạo phản hồi từ chatbot.
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
    
    # Lấy pipeline, sẽ tự động tải nếu chưa có
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
