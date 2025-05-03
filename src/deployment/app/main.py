from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from app.model import load_model, generate_response
from app.schema import PromptRequest, PromptResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

app = FastAPI(title="NVIDIA NeMo Text Generation API")

# Get the directory where main.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")

app.mount("/static", StaticFiles(directory=static_dir), name="static")

model = load_model()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        response = generate_response(request.prompt, model)
        return {"generated_text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path) as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
