from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
import torch

def load_model():
    model = MegatronGPTModel.restore_from("/home/tarik-devh/Projects/llm_finetune_nemo/results/megatron_gpt_peft_adapter_tuning/checkpoints/megatron_gpt_peft_adapter_tuning.nemo", map_location="cuda")
    model.eval()
    return model

def generate_response(prompt: str, model) -> dict:
    response = model.generate(inputs=[prompt], max_length=64)
    return {"response": response[0]}
