# LLM Fine-Tuning with NVIDIA NeMo and TensorRT-LLM

This project demonstrates how to fine-tune a NeMo model using PEFT adapters and export it to TensorRT-LLM format for optimized inference using NVIDIA Inference Manager (NIM).

## Project Structure

```
.
├── configs/                      # Training configuration files
├── data/                        # Training data directory
├── models/                      # Model checkpoints and exports
├── results/                     # Training results and logs
└── src/
    ├── deployment/              # Deployment-related code
    │   ├── app/                # Application code
    │   │   ├── export_model.py # Model export script
    │   │   └── server.py       # FastAPI server
    │   ├── config/             # Deployment configuration
    │   │   └── config.yaml     # Export configuration
    │   ├── export_model.sh     # Script to export model to TensorRT-LLM
    │   └── start_nim.sh        # Script to start NIM server
    └── data_generation/        # Data generation scripts
```

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- Python 3.11 (for local development)

## Usage

### 1. Export Model to TensorRT-LLM

To export a trained NeMo model to TensorRT-LLM format:

```bash
cd src/deployment
./export_model.sh \
    ../configs/training_config.yaml \
    ../results/megatron_gpt_peft_adapter_tuning/checkpoints/megatron_gpt_peft_adapter_tuning.nemo \
    ../exported_model
```

This script uses the official NeMo container (`nvcr.io/nvidia/nemo:25.04.rc2`) to perform the export.

### 2. Start NIM Server

Start the NVIDIA Inference Manager server for optimized model serving:

```bash
./start_nim.sh exported_model
```

This will start the NIM container (`nvcr.io/nvidia/nim/megatron-gpt:24.03`) with your exported model.

### 3. Start FastAPI Server

In a separate terminal, start the FastAPI server:

```bash
docker run -it --gpus all \
    -p 8000:8000 \
    -e NIM_API_ENDPOINT=http://host.docker.internal:8001/generate \
    nvcr.io/nvidia/nemo:25.04.rc2 \
    python src/deployment/app/server.py
```

The FastAPI server will communicate with the NIM server for model inference.

## Configuration

The export process uses a configuration file (`configs/training_config.yaml`) that specifies:
- Model architecture parameters
- Compute requirements
- Export settings

Example configuration:
```yaml
model:
  model:
    num_layers: 12
    hidden_size: 768
    num_attention_heads: 12
    max_position_embeddings: 2048
    precision: "16-mixed"
    compute:
      gpu_required: 1
      memory: "16GB"
      precision: "fp16"
```

## Development

### Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r src/deployment/requirements.txt
```

## Notes

- The export process uses the official NeMo container for TensorRT-LLM conversion
- Model serving is handled by NVIDIA Inference Manager for optimal performance
- The FastAPI server is lightweight and only handles request routing
- Make sure your GPU has enough memory for the model export process
- The exported model will be saved in the specified export directory

## Troubleshooting

If you encounter issues:

1. Check GPU availability:
```bash
nvidia-smi
```

2. Verify Docker NVIDIA runtime:
```bash
docker info | grep "Runtimes"
```

3. Ensure the model checkpoint exists and is accessible
4. Check the configuration file for correct model parameters
5. Verify sufficient disk space for the export process

## License

[Your License Information]
