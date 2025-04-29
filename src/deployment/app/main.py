import logging
from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
from app.model import load_model, generate_response
from app.config import Settings
from pydantic import BaseModel

app = FastAPI()

@app.get("/healthcheck")
def healthcheck():
    return {"status": "healthy"}
settings = Settings()
model = load_model(settings.MODEL_PATH)

class Query(BaseModel):
    input_text: str

@app.post("/generate")
def generate(query: Query):
    output = generate_response(model, query.input_text)
    return {"response": output}