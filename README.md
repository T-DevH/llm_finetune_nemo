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
│   ├── train/                   # Training data
│   └── val/                     # Validation data
├── models/                      # Pre-trained model checkpoints
├── results/                     # Training results and logs
├── scripts/                     # Utility scripts
│   ├── run_nemo_container.sh    # Main training script
│   ├── monitor_training.sh      # Training monitoring
│   └── plot_training_loss.py    # Loss visualization
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

2. Download the base model:
```bash
mkdir -p models
# Download megatron_gpt_345m.nemo from NGC and place in models/
```

3. Authenticate with NVIDIA NGC:
```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: Your NGC API key
```

## Training

### 1. Data Preparation

Place your training data in the appropriate directories:
- `data/train/data.jsonl` - Training data
- `data/val/data.jsonl` - Validation data

The data should be in JSONL format with each line containing:
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
  precision: "bf16-mixed"
  max_steps: 1000
  val_check_interval: 100

data:
  train_ds:
    file_names: ["data/train/data.jsonl"]
    concat_sampling_probabilities: [1.0]
  validation_ds:
    file_names: ["data/val/data.jsonl"]
    concat_sampling_probabilities: [1.0]
```

### 3. Training

Start the training using the NeMo container:

```bash
bash scripts/run_nemo_container.sh
```

This script:
- Launches the NeMo container with GPU support
- Mounts your workspace
- Runs the Megatron-GPT fine-tuning script with your configuration
- Saves checkpoints and logs to the `results/` directory

Monitor training progress:
```bash
bash scripts/monitor_training.sh
```

Visualize training loss:
```bash
python scripts/plot_training_loss.py
```

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

