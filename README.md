# LLM Fine-tuning with NVIDIA NeMo

This project provides a streamlined approach to fine-tuning large language models using NVIDIA's NeMo framework. It includes scripts for model training, progress monitoring, and configuration management.

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
├── configs/               # Configuration files
│   └── training_config.yaml  # Main training configuration
├── data/                 # Training data
│   ├── train/           # Training data (JSONL format)
│   ├── val/            # Validation data (JSONL format)
│   └── test/           # Test data (JSONL format)
├── models/              # Model files
│   ├── megatron_gpt_345m.nemo  # Downloaded NEMO model
│   └── megatron_gpt_345m/      # Extracted model directory
├── scripts/             # Utility scripts
│   ├── run_nemo_container.sh  # Main training script
│   ├── monitor_training.sh    # Training monitoring script
│   └── extract_nemo_model.sh  # Model download and extraction script
├── src/                 # Source code
│   ├── data_generation/ # Data generation utilities
│   └── deployment/     # Deployment scripts
├── logs/               # Training logs
└── results/            # Training results and checkpoints
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

## Getting Started

1. Clone the repository:
```bash
git clone <repository-url>
cd llm_finetune_nemo
```

2. Make the scripts executable:
```bash
chmod +x scripts/*.sh
```

3. Download and extract the model:
```bash
./scripts/extract_nemo_model.sh
```
This will:
- Download the 345M GPT model from NGC
- Save it to `models/megatron_gpt_345m.nemo`
- Extract it to `models/megatron_gpt_345m/`

4. Prepare your data:
   - Place training data in `data/train/data.jsonl`
   - Place validation data in `data/val/data.jsonl`
   - Place test data in `data/test/data.jsonl`

5. Configure training parameters in `configs/training_config.yaml`:
   - Adjust parameters based on your hardware and requirements
   - See the Parameter Recommendations section below

6. Run the container and start training:
```bash
./scripts/run_nemo_container.sh
```

The script will:
- Pull the NeMo container (nvcr.io/nvidia/nemo:25.04.nemotron-h)
- Verify the workspace structure
- Start the training process with the specified configuration

## Parameter Recommendations

| Setting | Current | Recommended | Why |
|---------|---------|-------------|-----|
| train_ds.num_workers | 0 | 8 or 16 | Faster dataloading |
| validation_ds.num_workers | 0 | 8 or 16 | Faster validation |
| train_ds.prefetch_factor | - | 2 | Smoother batch pipeline |
| validation_ds.prefetch_factor | - | 2 | Smoother val batches |
| warmup_steps | 50 | 200 | Better learning rate ramp |
| hidden_dropout | 0.0 | 0.1 | Slight regularization |
| attention_dropout | 0.0 | 0.1 | Slight regularization |
| ffn_dropout | 0.0 | 0.1 | Slight regularization |
| early_stopping.patience | 10 | 20 | Avoid premature stopping |
| memmap_workers | 2 | 4 (optional) | Faster data indexing |

## Training Configuration

The training configuration is stored in `configs/training_config.yaml`. Key parameters include:

### Model Configuration
- Base model: Megatron GPT 345M
- Sequence length: 2048 tokens
- Batch size: 128 (global), 4 (micro)
- Mixed precision: bf16
- PEFT (LoRA) configuration:
  - Target modules: attention_qkv
  - Adapter dimension: 32
  - Alpha: 32
  - Dropout: 0.0

### Training Parameters
- Maximum steps: 20,000
- Validation interval: Every 200 steps
- Learning rate: 0.0001
- Optimizer: Fused Adam
- Learning rate scheduler: Cosine Annealing
- Warmup steps: 50
- Gradient clipping: 1.0

### Early Stopping
- Monitor: validation loss
- Patience: 10
- Minimum delta: 0.001

## Monitoring Training Progress

1. Make the monitoring script executable:
```bash
chmod +x scripts/monitor_training.sh
```

2. Run the monitoring script:
```bash
./scripts/monitor_training.sh
```

The script displays:
- Current training step and epoch
- Training and validation loss
- Learning rate
- GPU memory usage
- Updates every minute

## Results and Checkpoints

Training results and model checkpoints are saved in the `results/` directory. The system saves:
- Best model checkpoint based on validation loss
- Final model checkpoint
- Training logs and metrics

## Troubleshooting

Common issues and solutions:
1. Memory issues: 
   - Adjust `micro_batch_size` in the config file
   - Enable gradient checkpointing
   - Reduce sequence length if needed
2. Training instability: 
   - Adjust learning rate or warmup steps
   - Enable dropout for regularization
   - Increase batch size if possible
3. Container errors: 
   - Verify NGC authentication and API key
   - Check Docker and NVIDIA Container Toolkit installation
4. Data format errors: 
   - Ensure JSONL files follow the required format
   - Validate data using the data generation utilities
5. Model extraction issues:
   - Check NGC API key and authentication
   - Verify sufficient disk space
   - Ensure Python environment has required dependencies
6. Performance issues:
   - Increase number of workers for data loading
   - Enable prefetching
   - Optimize memmap workers

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA NeMo team for the framework
- NVIDIA NIM team for the optimized models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
