#!/usr/bin/env python3

import os
import logging
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
import lightning.pytorch as pl
import uvicorn
from dataclasses import dataclass
from nemo.core.config import hydra_runner
from nemo.utils import logging as nemo_logging
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf
from typing import List
import json
from pathlib import Path
import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp.modules.common.megatron.utils import parallel_state
import torch.nn.functional as F
from nemo.collections.nlp.modules.common.text_generation_utils import megatron_gpt_generate
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.modules.common.text_generation_utils import generate

# Initialize distributed environment
def init_distributed():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1
    )

# Initialize distributed environment
init_distributed()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve")

# Define model paths
BASE_PATH = "/workspace/model/base_model.nemo"
LORA_PATH = "/workspace/model/lora_model.nemo"

# Initialize FastAPI app
app = FastAPI(title="Megatron GPT LoRA API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="/workspace/app/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("/workspace/app/UI/megatron-ui.html")

# Initialize trainer with GPU
trainer = None
if torch.cuda.is_available():
    trainer = pl.Trainer(devices=1, accelerator='gpu')

try:
    # Load base model
    logger.info("Loading base model...")
    model = MegatronGPTModel.restore_from(restore_path=BASE_PATH, trainer=trainer)
    model.eval()
    
    # Load LoRA adapter weights
    logger.info("Loading LoRA adapter weights...")
    try:
        # Load LoRA weights directly from the .nemo file (regular tar archive)
        import tempfile
        import os
        from pathlib import Path
        import tarfile
        
        # Create a temporary directory to extract the .nemo file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the .nemo file (it's a regular tar archive)
            with tarfile.open(LORA_PATH, 'r') as tar:
                tar.extractall(temp_dir)
            
            # Load the state dict from the extracted files
            adapter_path = os.path.join(temp_dir, 'model_weights.ckpt')
            if not os.path.exists(adapter_path):
                raise ValueError(f"Could not find weights file in {LORA_PATH}")
            
            adapter_state = torch.load(adapter_path, map_location='cpu')
            if isinstance(adapter_state, dict) and 'state_dict' in adapter_state:
                adapter_weights = adapter_state['state_dict']
            else:
                adapter_weights = adapter_state
            
            # Filter out non-adapter weights
            adapter_weights = {k: v for k, v in adapter_weights.items() if 'adapter_layer' in k}
            
            # Update only the adapter layers in the base model
            current_state = model.state_dict()
            current_state.update(adapter_weights)
            model.load_state_dict(current_state)
            
            logger.info("Successfully loaded and merged LoRA adapter weights")
            
    except Exception as lora_error:
        raise ValueError(f"Error loading LoRA weights: {str(lora_error)}")
    
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

# API models
class GenerationRequest(BaseModel):
    text: str
    max_length: int = 100  # Made optional with default value
    min_length: int = 0
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    greedy: bool = False
    repetition_penalty: float = 1.0
    add_BOS: bool = False

@dataclass
class LengthParam:
    max_length: int
    min_length: int = 0

@dataclass
class SamplingParam:
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 0.9
    greedy: bool = False
    repetition_penalty: float = 1.0
    add_BOS: bool = False

@app.post("/generate")
async def generate_text(request: GenerationRequest) -> GenerationResponse:
    """Generate text using the loaded model."""
    try:
        # Format the input with a structured prompt
        formatted_input = f"User: {request.text}\nAgent:"
        logger.info(f"Formatted input: {formatted_input}")
        
        # Tokenize input
        tokens = model.tokenizer.text_to_ids(formatted_input)
        tokens = torch.tensor([tokens], device=model.device)
        seq_length = tokens.shape[1]
        logger.info(f"Input tensor shape: {tokens.shape}")
        
        # Set up length parameters
        max_length = request.max_length if request.max_length is not None else 100
        min_length = 20  # Minimum length for a reasonable response
        compute_logprobs = False
        
        # Set up sampling parameters with more focused values
        temperature = 0.5  # Lower temperature for more focused generation
        top_k = 0
        top_p = 0.9
        use_greedy = True  # Use greedy decoding for more predictable responses
        repetition_penalty = 1.2  # Slightly higher repetition penalty
        add_BOS = False
        all_probs = False
        compute_attention_mask = True
        
        logger.info(f"Generation parameters: max_length={max_length}, min_length={min_length}, "
                   f"temperature={temperature}, top_k={top_k}, top_p={top_p}, "
                   f"greedy={use_greedy}, repetition_penalty={repetition_penalty}")
        
        # Initialize distributed environment if not already initialized
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1
            )
        
        # Generate text
        try:
            # Create attention mask
            attention_mask = torch.ones(1, seq_length, seq_length, device=tokens.device, dtype=torch.bool)
            
            # Initialize generation
            current_length = seq_length
            generated_tokens = []
            
            # Generate tokens
            for _ in range(max_length - seq_length):
                # Get model output
                output = model.forward(
                    tokens,
                    position_ids=torch.arange(0, current_length, device=tokens.device).unsqueeze(0),
                    attention_mask=attention_mask,
                    labels=None
                )
                
                # Get logits for the last token
                logits = output.logits[:, -1, :]
                logger.info(f"Logits shape: {output.logits.shape}")
                
                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty > 0:
                    for token_id in set(generated_tokens):
                        logits[0, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample or use greedy decoding
                if use_greedy:
                    next_token = torch.argmax(logits, dim=-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for end of sequence
                if next_token.item() == model.tokenizer.eos_id:
                    break
                
                # Add token to sequence
                generated_tokens.append(next_token.item())
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                current_length += 1
                
                # Update attention mask
                new_seq_length = tokens.shape[1]
                attention_mask = torch.ones(1, new_seq_length, new_seq_length, device=tokens.device, dtype=torch.bool)
            
            # Decode the generated tokens
            generated_text = model.tokenizer.ids_to_text(generated_tokens)
            logger.info(f"Generated text: {generated_text}")
            
            # Extract only the agent's response
            if "Agent:" in generated_text:
                response = generated_text.split("Agent:")[-1].strip()
            else:
                response = generated_text.strip()
            
            # Validate response
            if len(response) < min_length or not any(c.isalpha() for c in response):
                response = "I apologize, but I'm having trouble generating a proper response. Please try rephrasing your request or contact our customer service directly."
            
            return GenerationResponse(text=response)
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in generate_text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in generate_text: {str(e)}")

# Helper functions for sampling
def top_k_sampling(logits, k):
    v, _ = torch.topk(logits, k)
    min_value = v[:, -1].unsqueeze(-1)
    logits = torch.where(logits < min_value, torch.ones_like(logits) * -float('Inf'), logits)
    return logits

def top_p_sampling(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, -float('Inf'))
    return logits

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000)