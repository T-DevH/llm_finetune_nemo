from pydantic import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "/app/model/model.nemo"