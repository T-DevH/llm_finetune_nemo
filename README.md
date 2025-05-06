# LLM Fine-Tuning with NVIDIA NeMo

This repository is designed to deliver a streamlined and practical developer experience with NVIDIA NeMo — a powerful, modular framework that serves as the Swiss Army knife for building, fine-tuning, and deploying large language models. It encapsulates best practices for Parameter-Efficient Fine-Tuning (PEFT) using LoRA, and provides a full training pipeline including configurable scripts, real-time training progress monitoring, and model management. Whether you're experimenting with small-scale prototypes or fine-tuning large models for production, this repo offers an end-to-end workflow tailored for both research and applied use.

## Features

- Fine-tuning of NVIDIA NeMo models (345M parameter GPT model)
- Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- Configurable training parameters and model settings
- Automatic checkpointing and early stopping
- GPU-optimized training pipeline with mixed precision (bf16)
- Comprehensive monitoring of training progress
- Memory-efficient training with gradient checkpointing
- Optimized data loading and preprocessing
- Automatic model downloading and extraction

## Prerequisites

- NVIDIA GPU with CUDA support (A100 or newer recommended)
- Docker and NVIDIA Container Toolkit installed
- NGC account with API key
- Sufficient disk space for model and training data (at least 30GB recommended)
- Memory: At least 16GB RAM recommended
- Python 3.9+ for local development

## Environment Setup

1. Set your NGC API key (you can find this at https://ngc.nvidia.com/setup/api-key):
```bash
export NGC_CLI_API_KEY=your_api_key
```

2. Authenticate with NVIDIA NGC:
```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: Your NGC API key
```

3. Create and activate the Python virtual environment:
```bash
python -m venv nemo_lora
source nemo_lora/bin/activate
```

## Project Structure

```
.
├── configs/                    # Configuration files
│   ├── training_config.yaml    # Training configuration
│   └── model_config.yaml       # Model configuration
├── data/                       # Training data
│   └── alpaca_data.json        # Alpaca format dataset
├── src/
│   ├── training/              # Training scripts
│   │   └── train.py           # Main training script
│   └── deployment/            # Deployment files
│       ├── app/               # FastAPI application
│       │   ├── main.py        # API endpoints
│       │   ├── model.py       # Model interface
│       │   ├── schema.py      # Data schemas
│       │   └── static/        # Frontend files
│       │       └── index.html # Web interface
│       ├── Dockerfile         # Container definition
│       └── requirements.txt    # Python dependencies
└── README.md                  # This file
```

## Data Format

The training, validation, and test data should be in JSONL format with the following structure:
```json
{"input": "Your input text here", "output": "Your output text here"}
```

Example:
```json
{"input": "This is a sample training text.", "output": "training"}
```

## Data Generation

The project includes a synthetic data generation module (`src/data_generation/generate_retail_data.py`) that creates training data for retail customer service scenarios. This module:

- Generates realistic customer service conversations with:
  - Various products (smartphones, laptops, headphones, etc.)
  - Different types of issues (not working, damaged, missing parts, etc.)
  - Multiple resolution actions (return, replace, refund, repair, exchange)
  - Different customer tones (frustrated, polite, angry, confused, satisfied)

- Creates structured JSONL files with:
  - System prompt for the model
  - Customer input (query/issue)
  - Expected output (customer service response)

- Supports generating datasets for:
  - Training (default: 1000 samples)
  - Validation (default: 300 samples)
  - Testing (default: 300 samples)

- Automatically cleans up old data files before generating new ones

Usage:
```bash
python src/data_generation/generate_retail_data.py \
    --train_samples 1000 \
    --val_samples 300 \
    --test_samples 300 \
    --output_dir data
```

## Model Extraction

The `extract_nemo_model.sh` script handles downloading and preparing the base model:

1. Downloads the Megatron GPT 345M model from NGC
2. Places it in the correct directory structure
3. Verifies the extraction

Usage:
```bash
./scripts/extract_nemo_model.sh
```

## Training

1. Prepare your training data in JSONL format:
```jsonl
{"input": "Your input text here", "output": "Your output text here"}
```

2. Update the training configuration in `configs/training_config.yaml`:
```yaml
name: megatron_gpt_peft_full_tuning

trainer:
  devices: 1
  accelerator: gpu
  precision: bf16-mixed
  max_steps: 20000
  val_check_interval: 200
  gradient_clip_val: 1.0

model:
  global_batch_size: 128
  micro_batch_size: 4
  restore_from_path: /workspace/models/megatron_gpt_345m.nemo
  
  data:
    train_ds:
      file_names: [/workspace/data/train/data.jsonl]
      max_seq_length: 2048
      prompt_template: '{input} {output}'
    
    validation_ds:
      file_names: [/workspace/data/val/data.jsonl]
      max_seq_length: 2048
      prompt_template: '{input} {output}'

  peft:
    peft_scheme: lora
    lora_tuning:
      variant: nemo
      target_modules:
        - attention_qkv
      adapter_dim: 32
      alpha: 32
      adapter_dropout: 0.0
```

3. Start training using the provided script:
```bash
./scripts/run_nemo_container.sh
```

The training script will:
- Use the NeMo container with all necessary dependencies
- Mount your workspace to `/workspace` in the container
- Run the training with the specified configuration
- Save checkpoints and logs in the `results` directory

You can monitor the training progress using:
```bash
./scripts/monitor_training.sh
```

## Training Progress Visualization

The project includes a script to visualize training progress using TensorBoard event files. The script creates a plot showing the training loss over time, with annotations for minimum and maximum loss values.

Usage:
```bash
# Generate a plot from TensorBoard logs
python scripts/plot_training_loss.py \
    --log_dir results/megatron_gpt_peft_adapter_tuning \
    --output training_loss_plot.png
```

The script features:
- Automatic detection of loss metrics in TensorBoard event files
- Clear visualization of training progression
- Markers for significant steps (every 1000 steps)
- Min/max loss annotations
- High-resolution output (300 DPI)
- Grid lines for better readability

Requirements:
```bash
pip install tensorboard matplotlib
```

## Results and Checkpoints

Training results and model checkpoints are saved in the `results/` directory. The system saves:
- Best model checkpoint based on validation loss
- Final model checkpoint
- Training logs and metrics

## Deployment

### Method 1: Direct NeMo Deployment

1. Build the Docker container:
```bash
cd src/deployment
docker build -t nemo-finetune .
```

2. Run the container:
```bash
docker run -it --gpus all \
  -v $(pwd)/model:/app/model \
  -p 8000:8000 \
  nemo-finetune
```

3. Access the API:
- Web Interface: http://localhost:8000
- API Endpoint: http://localhost:8000/generate
- Health Check: http://localhost:8000/health

### Method 2: TensorRT-LLM with NIM (Recommended)

This method provides better inference performance using TensorRT-LLM and NVIDIA Inference Manager (NIM).

1. Export the model to TensorRT-LLM format:
```bash
cd src/deployment
./export_model.sh \
    configs/model_config.yaml \
    results/megatron_gpt_peft_adapter_tuning/checkpoints/megatron_gpt_peft_adapter_tuning.nemo \
    exported_model
```

2. Start the NIM container for model serving:
```bash
docker run -it --gpus all \
  -v $(pwd)/exported_model:/model \
  -p 8001:8000 \
  nvcr.io/nvidia/nim/megatron-gpt:24.03 \
  bash -c "NIM_API_MODEL_PATH=/model NIM_API_PORT=8000 start-nim"
```

3. Start the FastAPI application:
```bash
docker run -it --gpus all \
  -p 8000:8000 \
  -e NIM_API_ENDPOINT=http://host.docker.internal:8001/generate \
  nemo-finetune
```

4. Access the API:
- Web Interface: http://localhost:8000
- API Endpoint: http://localhost:8000/generate
- Health Check: http://localhost:8000/health

## API Usage

### Generate Text

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

Response:
```json
{
  "response": "The capital of France is Paris."
}
```

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU device to use (default: 0)
- `SHMEM_SIZE`: Shared memory size (default: 8G)
- `NIM_API_ENDPOINT`: NIM API endpoint URL (default: http://localhost:8001/generate)

## Troubleshooting

1. CUDA Out of Memory:
   - Reduce batch size in training config
   - Use gradient checkpointing
   - Enable 8-bit quantization

2. Slow Inference:
   - Use TensorRT-LLM deployment method
   - Enable model caching
   - Optimize batch size

3. Container Issues:
   - Ensure NVIDIA Container Toolkit is installed
   - Check GPU visibility in container
   - Verify shared memory settings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA NeMo team for the framework
- Alpaca team for the dataset format
- Hugging Face for the base models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

# NeMo PEFT Adapter Tuning with TensorRT-LLM Export

This project demonstrates how to fine-tune a NeMo model using PEFT adapters and export it to TensorRT-LLM format for optimized inference.

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
    │   ├── Dockerfile          # Docker configuration
    │   └── requirements.txt    # Python dependencies
    └── training/               # Training-related code
        ├── data_prep.py       # Data preparation script
        └── train.py           # Training script
```

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- Python 3.11 (for local development)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Build the Docker image:
```bash
docker build -t nemo-trtllm-exporter -f src/deployment/Dockerfile .
```

## Usage

### Export Model to TensorRT-LLM

To export a trained NeMo model to TensorRT-LLM format:

```bash
docker run --gpus all -v $(pwd):/app nemo-trtllm-exporter \
    python src/deployment/app/export_model.py \
    --config src/deployment/config/config.yaml \
    --checkpoint results/megatron_gpt_peft_adapter_tuning/checkpoints/megatron_gpt_peft_adapter_tuning.nemo \
    --export-dir exported_model
```

### Configuration

The export process uses a configuration file (`src/deployment/config/config.yaml`) that specifies:
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

3. Run the export script locally:
```bash
python src/deployment/app/export_model.py \
    --config src/deployment/config/config.yaml \
    --checkpoint results/megatron_gpt_peft_adapter_tuning/checkpoints/megatron_gpt_peft_adapter_tuning.nemo \
    --export-dir exported_model
```

## Notes

- The Docker image uses Python 3.11 from the deadsnakes PPA to ensure compatibility with NeMo 25.04
- The export process requires specific versions of PyTorch and related packages
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
