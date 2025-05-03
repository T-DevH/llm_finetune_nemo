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

1. Prepare your training data in Alpaca format (JSON):
```json
[
    {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris."
    }
]
```

2. Update the training configuration in `configs/training_config.yaml`:
```yaml
model:
  name: "gpt2"
  pretrained_model_name: "gpt2"
  adapter_tuning:
    enabled: true
    adapter_dim: 8
    adapter_dropout: 0.1

trainer:
  max_steps: 1000
  val_check_interval: 100
  gradient_clip_val: 1.0
  precision: 16
  accelerator: "gpu"
  devices: 1

data:
  train_file: "data/alpaca_data.json"
  validation_split: 0.1
  max_seq_length: 512
  batch_size: 4
  num_workers: 4
```

3. Start training:
```bash
python src/training/train.py
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
./export_model.sh <config_path> <checkpoint_path> <export_dir>
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
