from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "/home/tarik-devh/Projects/llm_finetune_nemo/results/megatron_gpt_peft_adapter_tuning/checkpoints/megatron_gpt_peft_adapter_tuning.nemo"
    
    class Config:
        env_file = ".env"