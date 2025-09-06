# === PART 1: IMPORTS AND CONFIGURATION ===
import os
import traceback
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Get Hugging Face API Key from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# URLs for models on the Hugging Face Inference API
CHAT_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
## ONLY CHANGE HERE ##
# Switched to a very popular and stable sentiment model
SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"


# === PART 2: API CALL FUNCTIONS ===

# Helper function to query the HF API
async def query_hf_api(api_url: str, payload: dict, client: httpx.AsyncClient):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN is not configured on the server.")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        response = await client.post(api_url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        print(f"API Error from Hugging Face ({api_url}): {e.response.text}")
        return {"error": f"API Error: {e.response.status_code}", "detail": e.response.text}
    except Exception as e:
        print(f"Unknown error calling {api_url}: {e}")
        return {"error": "Unknown error during API call."}


# === PART 3: FASTAPI API CREATION ===
app = FastAPI(title="Athena AI Therapist API (Hugging Face-Only)")

class ChatRequest(BaseModel):
    user_input: str
    history: list = []

@app.get("/")
def read_root():
    return {"status": "Athena AI API is running"}

@app.post("/chat")
async def handle_chat(request: ChatRequest):
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="User input is empty.")

    try:
        past_user_inputs = [msg['content'] for msg in request.history if msg['role'] == 'user']
        generated_responses = [msg['content'] for msg in request.history if msg['role'] == 'assistant']

        chat_payload = {
            "inputs": {
                "past_user_inputs": past_user_inputs,
                "generated_responses": generated_responses,
                "text": request.user_input
            },
            "parameters": {"max_new_tokens": 250}
        }
        analysis_payload = {"inputs": request.user_input}

        async with httpx.AsyncClient() as client:
            chat_task = query_hf_api(CHAT_API_URL, chat_payload, client)
            sentiment_task = query_hf_api(SENTIMENT_API_URL, analysis_payload, client)
            emotion_task = query_hf_api(EMOTION_API_URL, analysis_payload, client)

            results = await asyncio.gather(chat_task, sentiment_task, emotion_task)

            chat_result = results[0]
            sentiment_result = results[1]
            emotion_result = results[2]

        ai_response = chat_result.get('generated_text', 'Sorry, I could not generate a response.').strip()

        return {
            "response": ai_response,
            "sentiment_analysis": sentiment_result,
            "emotion_analysis": emotion_result
        }
    except Exception as e:
        print("--- [ERROR] An error occurred in the /chat endpoint! ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
