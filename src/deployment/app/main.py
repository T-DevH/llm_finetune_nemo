from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.model import load_model, generate_response
from app.schema import PromptRequest, PromptResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

model = load_model()

@app.post("/generate", response_model=PromptResponse)
def generate_text(request: PromptRequest):
    return generate_response(request.prompt, model)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    with open("app/static/index.html") as f:
        return HTMLResponse(content=f.read())
