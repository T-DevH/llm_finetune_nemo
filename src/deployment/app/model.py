import torch
import logging
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.text_generation_strategy import TextGenerationStrategy
from nemo.collections.nlp.modules.common.text_generation_utils import generate

logger = logging.getLogger(__name__)

def load_model(model_path):
    try:
        # Load the fine-tuned model with LoRA adapters
        model = MegatronGPTModel.restore_from(
            model_path,
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Set model to evaluation mode
        model.eval()
        
        # Configure generation parameters
        model.set_inference_config({
            "min_tokens_to_generate": 1,
            "max_tokens_to_generate": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 0,
            "repetition_penalty": 1.2,
            "add_BOS": True,
            "greedy": False,
        })
        
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_response(model, input_text):
    try:
        # Prepare input
        input_ids = model.tokenizer.text_to_ids(input_text)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                top_k=0,
                repetition_penalty=1.2,
                add_BOS=True,
                greedy=False,
            )
        
        # Decode response
        response = model.tokenizer.ids_to_text(output_ids[0].cpu().numpy().tolist())
        
        # Clean up response (remove input text if present)
        if response.startswith(input_text):
            response = response[len(input_text):].strip()
        
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise