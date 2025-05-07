#!/bin/bash
# src/deployment/start_nim.sh

# Exit on error
set -e

# Configuration
CONFIG_FILE="config/config.yaml"
MODEL_DIR="models"
LORA_ADAPTERS_DIR="lora_adapters/megatron_gpt_345m_tuned"
CONTAINER_NAME="nim-lora-server"
NEMO_CONTAINER="nvcr.io/nvidia/nemo:25.04.rc2"

# Set environment variables
export BASE_MODEL_PATH="/home/tarik-devh/Projects/llm_finetune_nemo/models/megatron_gpt_345m/megatron_gpt_345m.nemo"
export LORA_PATH="/home/tarik-devh/Projects/llm_finetune_nemo/results/megatron_gpt_peft_adapter_tuning/checkpoints/megatron_gpt_peft_adapter_tuning.nemo"

# Create necessary directories
mkdir -p "${MODEL_DIR}"
mkdir -p "${LORA_ADAPTERS_DIR}"

# Check if base model exists
if [ ! -f "${BASE_MODEL_PATH}" ]; then
    echo "Error: Base model not found at ${BASE_MODEL_PATH}"
    echo "Please download the base model and place it in the models directory"
    exit 1
fi

# Check if LoRA model exists
if [ ! -f "${LORA_PATH}" ]; then
    echo "Error: LoRA model not found at ${LORA_PATH}"
    echo "Please ensure the LoRA model exists"
    exit 1
fi

# Copy models to their respective directories
echo "Copying base model to ${MODEL_DIR}/base_model.nemo"
cp "${BASE_MODEL_PATH}" "${MODEL_DIR}/base_model.nemo"

echo "Copying LoRA model to ${MODEL_DIR}/lora_model.nemo"
cp "${LORA_PATH}" "${MODEL_DIR}/lora_model.nemo"

# Run the NeMo container
docker run -it --rm \
  --runtime=nvidia \
  --gpus all \
  --shm-size=8GB \
  -p 8000:8000 \
  -v "$(pwd)/${MODEL_DIR}:/workspace/model" \
  -v "$(pwd)/app:/workspace/app" \
  -v "$(pwd)/UI:/workspace/app/UI" \
  -w /workspace \
  ${NEMO_CONTAINER} \
  bash -c '
    # Install only the extra dependencies needed for the FastAPI server
    pip install fastapi uvicorn hydra-core lightning transformers sentencepiece braceexpand webdataset h5py ijson
    
    # Create static directory
    mkdir -p /workspace/app/static
    
    # Run the FastAPI server
    cd /workspace/app
    python -m uvicorn serve:app --host 0.0.0.0 --port 8000
  '