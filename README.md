# LLM Fine-Tuning with NVIDIA NeMo.

This project demonstrates how to fine-tune a NeMo model using PEFT adapters.

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

### 1. Setup Deployment Environment
The deployment setup is located in `src/deployment/` and provides everything needed to serve your fine-tuned model via a REST API:

```
src/deployment/
├── app/                # Application code
│   └── serve.py        # FastAPI server with text generation implementation
├── UI/                 # Web interface 
│   └── megatron-ui.html # Simple UI for testing the model
├── config/             # Deployment configuration
│   └── config.yaml     # Server and model configuration
└── start_nim.sh        # Script to start the server with proper initialization
```

### 2. Understanding the Server Implementation
The `serve.py` implementation handles:

- Distributed environment initialization required for NeMo 2.0 models
- Loading of the base model and LoRA adapter weights
- Text generation with proper parameter handling
- REST API endpoints for model interaction

Key components:

- **Distributed Setup**: Initializes the PyTorch distributed environment
- **Model Loading**: Loads the base model and merges LoRA adapter weights
- **Text Generation**: Implements token-by-token generation with sampling controls
- **API Endpoints**: Provides HTTP endpoints for generation and health checks

### 3. Start the Server
To start the server with your fine-tuned model:
```bash
cd src/deployment
./start_nim.sh
```

This script:
- Creates necessary directories for models and LoRA adapters
- Copies your base model and LoRA adapter to the correct locations
- Starts the FastAPI server with GPU support using the NeMo 2.0 container
- Makes the server available at http://localhost:8000

### 4. API Reference
The server provides the following endpoints:

#### Health Check
- **Endpoint**: GET /health
- **Response**: `{"status": "healthy"}` if server is running

#### Text Generation
- **Endpoint**: POST /generate
- **Request Body**:
```json
{
  "text": "Your input text here",
  "max_length": 100,
  "min_length": 0,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 0,
  "greedy": false,
  "repetition_penalty": 1.0
}
```

Parameters:
- `text` (string, required): Input text to continue
- `max_length` (int, optional): Maximum number of tokens to generate (default: 100)
- `min_length` (int, optional): Minimum number of tokens to generate (default: 0)
- `temperature` (float, optional): Controls randomness - lower for more deterministic output (default: 0.7)
- `top_p` (float, optional): Nucleus sampling parameter (default: 0.9)
- `top_k` (int, optional): Top-k sampling parameter (default: 0)
- `greedy` (boolean, optional): Use greedy decoding instead of sampling (default: false)
- `repetition_penalty` (float, optional): Penalize repeated tokens (default: 1.0)

- **Response**:
```json
{
  "generated_text": "The model generated text based on your input..."
}
```

### 5. Web Interface
A simple web interface is available at the root endpoint (/). Access it by opening http://localhost:8000 in your browser to test the model interactively.

### Best Practices for Deployment

#### Model Initialization
- **Distributed Environment**: Always initialize the distributed environment before loading NeMo 2.0 models using `parallel_state.initialize_model_parallel()`.
- **LoRA Adapter Management**: When loading LoRA adapters, filter weights to include only adapter-specific parameters to avoid mixing with base model weights.
- **Error Handling**: Implement comprehensive error handling during model loading to catch issues early.

#### Text Generation
- **Structured Prompts**: For customer service or specific domain applications, use structured prompts (e.g., "User: {query}\nAgent: ") to give the model context about expected output format.
- **Parameter Tuning**:
  - For more deterministic and focused responses, use lower temperature (0.5-0.6)
  - For customer service applications, consider using greedy decoding
  - Apply repetition penalty (1.2-1.3) to prevent repetitive text
  - Set appropriate top_p values (0.9-0.95) for diverse but coherent responses
- **Response Processing**:
  - Extract only the relevant part of the generated text if using structured prompts
  - Implement natural stopping conditions to end generation at appropriate points
  - Consider quality checks to ensure the response meets minimum standards

#### Performance Optimization
- **Attention Mask Handling**: Use boolean masks with appropriate shapes to avoid masked_fill errors.
- **Batch Processing**: If handling multiple requests, consider implementing batching.
- **GPU Memory Management**: Use `torch.cuda.empty_cache()` after large batches to free memory.

#### API Design
- **Fallback Responses**: Always implement fallback mechanisms for error cases.
- **Parameter Validation**: Validate and constrain parameter ranges to avoid issues.
- **Logging**: Implement comprehensive logging to track usage patterns and diagnose issues.

#### Security Considerations
- **Input Validation**: Validate user inputs to prevent injection attacks.
- **Rate Limiting**: Implement rate limiting to prevent abuse.
- **CORS Policy**: Configure CORS settings appropriately for your application.

### Troubleshooting Deployment
Common issues and solutions:

#### Model Loading Errors
- Check that model paths are correct
- Verify CUDA is available (`torch.cuda.is_available()`)
- Ensure sufficient GPU memory

#### Generation Errors
- "masked_fill only supports boolean masks": Ensure attention masks are boolean type
- "object of type 'NoneType' has no len()": Check distributed environment initialization
- "missing required positional arguments": Verify parameter naming matches model expectations

#### Performance Issues
- Monitor GPU memory usage with `nvidia-smi`
- Implement caching for frequently generated responses
- Consider model quantization for improved performance

#### Distributed Setup
- If using multiple GPUs, ensure proper environment variables (MASTER_ADDR, MASTER_PORT)
- Verify model parallel configuration matches training setup

### Future Work
1. **Enhance Fine-tuning Approaches**:
   - Advanced LoRA configuration for better parameter efficiency
   - Improved prompt engineering techniques
   - Instruction tuning for better task-specific performance
   - Quality improvement through better training data curation

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

This project is licensed under the MIT License 


