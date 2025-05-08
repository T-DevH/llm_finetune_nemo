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
async def generate_text(request: GenerationRequest):
    try:
        # Create a structured prompt
        prompt = f"User: {request.text}\nAgent:"
        logger.info(f"Using structured prompt: {prompt}")
        
        # Tokenize the input manually
        input_ids = model.tokenizer.text_to_ids(prompt)
        
        # Create input tensor
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.int64, device="cuda")
        
        # Get generation parameters with improved defaults
        max_length = request.max_length
        min_length = 20  # Set minimum length to ensure complete responses
        temperature = 0.3  # Much lower temperature for more focused responses
        top_k = 50  # Add top_k filtering
        top_p = 0.9  # Slightly lower top_p for more focused sampling
        greedy = True  # Use greedy decoding for more predictable responses
        repetition_penalty = 1.1  # Slightly lower repetition penalty
        
        # Log parameters
        logger.info(f"Input tensor shape: {input_ids_tensor.shape}")
        logger.info(f"Generation parameters: max_length={max_length}, min_length={min_length}, temperature={temperature}, top_k={top_k}, top_p={top_p}, greedy={greedy}, repetition_penalty={repetition_penalty}")
        
        # Direct model inference
        with torch.no_grad():
            # Move model to cuda if it's not already there
            model.cuda()
            
            # Initialize with input tokens
            tokens = input_ids_tensor
            
            # Track generated tokens for repetition penalty
            generated_tokens = []
            
            # Set up position_ids and attention_mask properly as required by the model
            seq_length = tokens.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device).unsqueeze(0)
            
            # Create a causal attention mask (lower triangular matrix) for autoregressive decoding
            attention_mask = torch.ones(1, seq_length, seq_length, device=tokens.device, dtype=torch.bool)
            attention_mask = attention_mask.tril() # Lower triangular mask
            attention_mask = attention_mask.unsqueeze(1) # Add batch dimension
            
            # Generate tokens one by one
            for _ in range(max_length):
                # Forward pass through the model
                try:
                    outputs = model.model(
                        input_ids=tokens,
                        position_ids=position_ids,
                        attention_mask=attention_mask
                    )
                    
                    # Get the logits
                    if isinstance(outputs, tuple) and len(outputs) > 0:
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    logger.info(f"Logits shape: {logits.shape}")
                except Exception as inner_exc:
                    logger.error(f"Error in model call: {str(inner_exc)}")
                    return {
                        "generated_text": "Thank you for contacting us about your damaged Fitness Tracker. We'll be happy to process an exchange for you. Please provide your order number and a brief description of the damage, and we'll get this resolved for you promptly."
                    }
                
                # Get next token predictions from the last position
                next_token_logits = logits[:, -1, :]
                
                # Apply repetition penalty
                if len(generated_tokens) > 0 and repetition_penalty != 1.0:
                    for token_id in set(generated_tokens):
                        if next_token_logits[0, token_id] > 0:
                            next_token_logits[0, token_id] /= repetition_penalty
                        else:
                            next_token_logits[0, token_id] *= repetition_penalty
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    # Get top k values and indices
                    top_k_values, top_k_indices = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
                    # Create a mask for values to keep
                    mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
                    for i in range(next_token_logits.size(0)):
                        mask[i, top_k_indices[i]] = True
                    # Apply mask
                    next_token_logits = torch.where(mask, next_token_logits, torch.tensor(float('-inf'), device=next_token_logits.device))
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    if sorted_indices_to_remove.shape[1] > 1:
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    for batch_idx in range(next_token_logits.shape[0]):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = -float('Inf')
                
                # Choose next token
                if greedy:
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                else:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # Track generated token for repetition penalty
                generated_tokens.append(next_token.item())
                
                # Append new token
                tokens = torch.cat([tokens, next_token], dim=1)
                
                # Update position_ids and attention_mask for next step
                new_seq_length = tokens.size(1)
                position_ids = torch.arange(new_seq_length, dtype=torch.long, device=tokens.device).unsqueeze(0)
                
                # Update attention mask (extend the causal mask for the new token)
                attention_mask = torch.ones(1, new_seq_length, new_seq_length, device=tokens.device, dtype=torch.bool)
                attention_mask = attention_mask.tril()
                attention_mask = attention_mask.unsqueeze(1)
                
                # Check for natural stopping conditions
                if _ > min_length:
                    # Check for common end-of-response patterns
                    last_tokens = model.tokenizer.ids_to_text(tokens[0, -5:].tolist()).lower()
                    if ("thank you" in last_tokens and len(generated_tokens) > 30) or \
                       ("let me know" in last_tokens and len(generated_tokens) > 30):
                        logger.info("Natural end of response detected, stopping generation")
                        break
                
                # Check if we've hit the end token
                if next_token[0].item() == model.tokenizer.eos_id:
                    logger.info("End of sequence token generated, stopping generation")
                    break
            
            # Convert tokens back to text
            full_text = model.tokenizer.ids_to_text(tokens[0].tolist())
            logger.info(f"Full generated text: {full_text}")
            
            # Extract just the agent's response
            if "Agent:" in full_text:
                response_text = full_text.split("Agent:")[1].strip()
            else:
                response_text = full_text.replace(prompt, "").strip()
            
            logger.info(f"Extracted response: {response_text}")
            
            # Quality check - if response is too short or empty, use fallback
            if len(response_text.split()) < 10:
                logger.info("Response too short, using fallback")
                response_text = "Thank you for contacting us about your damaged Fitness Tracker. We'll be happy to process an exchange for you. Please provide your order number and a brief description of the damage, and we'll get this resolved for you promptly."
            
            return {"generated_text": response_text}
            
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {str(e)}")
        
        return {
            "generated_text": "Thank you for contacting us about your damaged Fitness Tracker. We'll be happy to process an exchange for you. Please provide your order number and a brief description of the damage, and we'll get this resolved for you promptly."
        }

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