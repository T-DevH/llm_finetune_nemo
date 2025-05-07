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

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    try:
        # Set up length parameters
        length_params = {
            "max_length": request.max_length,
            "min_length": request.min_length,
            "compute_logprobs": True
        }
        
        # Set up sampling parameters
        sampling_params = {
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "use_greedy": request.greedy,
            "repetition_penalty": request.repetition_penalty,
            "all_probs": False,
            "compute_logprob": True
        }
        
        # Generate text
        output = model.generate(
            inputs=[request.text],
            length_params=length_params,
            sampling_params=sampling_params
        )
        
        # Get the generated text
        if isinstance(output, torch.Tensor):
            output_tokens = output[0].tolist()
        else:
            output_tokens = output[0]
            
        generated_text = model.tokenizer.ids_to_text(output_tokens)
        
        return {"generated_text": generated_text}
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000)