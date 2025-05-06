# LLM Fine-Tuning with NVIDIA NeMo and TensorRT-LLM

This project demonstrates how to fine-tune a NeMo model using PEFT adapters and export it to TensorRT-LLM format for optimized inference.

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- Python 3.11 (for local development)
- NVIDIA NGC account (for accessing containers)

## Project Structure

```
.
├── configs/                      # Training and deployment configuration files
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
    │   └── start_nim.sh        # Script to start NIM server
    └── data_generation/        # Data generation scripts
```

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd llm_finetune_nemo
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Authenticate with NVIDIA NGC:
```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: Your NGC API key
```

## Training

### 1. Data Preparation

Place your training data in the `data/` directory. The data should be in the following format:
```json
{
    "text": "Your training text here"
}
```

### 2. Configuration

Edit `configs/training_config.yaml` to set your training parameters:

```yaml
model:
  model_name: "gpt-345m"
  pretrained: true
  peft:
    adapter_tuning:
      enabled: true
      adapter_dim: 64
      type: "parallel_adapter"

trainer:
  devices: 1
  accelerator: "gpu"
  precision: "16-mixed"
  max_steps: 1000
  val_check_interval: 100

data:
  train_file: "data/train.jsonl"
  val_file: "data/val.jsonl"
  batch_size: 8
```

### 3. Training

Start the training:

```bash
python src/train.py --config configs/training_config.yaml
```

Training progress and checkpoints will be saved in the `results/` directory.

## Model Export and Deployment

### 1. Export Model to TensorRT-LLM

After training, export your model to TensorRT-LLM format:

```bash
cd src/deployment
./start_nim.sh exported_model
```

This script:
- Uses the NeMo container to load your trained model
- Exports it to TensorRT-LLM format
- The exported model will be in the specified directory

### 2. Model Serving

The exported model can be served using FastAPI:

```bash
python src/deployment/app/server.py
```

## Configuration Details

### Training Configuration

The training configuration (`configs/training_config.yaml`) includes:
- Model architecture parameters
- PEFT adapter settings
- Training hyperparameters
- Data configuration

### Export Configuration

The export configuration (`src/deployment/config/config.yaml`) specifies:
- Model architecture parameters
- Compute requirements
- Export settings

## Development

For local development:

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Set up pre-commit hooks:
```bash
pre-commit install
```

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

3. Training issues:
   - Check GPU memory usage
   - Verify data format
   - Adjust batch size if needed

4. Export issues:
   - Ensure model checkpoint exists
   - Check configuration parameters
   - Verify sufficient disk space

## License

[Your License Information]

