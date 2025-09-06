# main.py
from fastapi import FastAPI

app = FastAPI(
    title="Athena AI Health Check",
    description="A simple API to check if the server is running.",
    version="0.1.0"
)

# KHÔNG CÓ startup event, KHÔNG import AI models

@app.get("/", tags=["Status"])
def read_root():
    """Endpoint để kiểm tra API có hoạt động không."""
    print("--- Health check endpoint '/' was called successfully. ---")
    return {"status": "OK", "message": "Simple FastAPI app is running!"}

@app.post("/chat", tags=["Chat"])
def handle_chat():
    """
    Endpoint chat giả lập, không dùng AI.
    """
    print("--- Mock chat endpoint '/chat' was called successfully. ---")
    return {"response": "This is a dummy response from the simple app."}
