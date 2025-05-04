# Use the official NeMo container with preinstalled stack
FROM nvcr.io/nvidia/nemo:25.04.nemotron-h

# Set working directory
WORKDIR /app

# Install CUDA headers if needed (e.g. for nvcc or custom ops)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    gnupg2 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    cuda-nvcc-12-1 \
    cuda-cudart-dev-12-1 && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.1-1_all.deb

# Uninstall conflicting preinstalled packages
RUN pip uninstall -y torch torchaudio torchvision fastapi pydantic transformers numba urllib3 pytorch-lightning lightning

# Reinstall NeMo-compatible core libraries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --force-reinstall \
    "torch==2.2.0" \
    "torchaudio==2.2.0" \
    "torchvision==0.17.0" \
    "fastapi==0.115.4" \
    "pydantic>=2.9.1" \
    "transformers>=4.47.0,<4.48.0" \
    "numba>=0.59.1,<0.61.0a0" \
    "urllib3>=1.21.1,<1.27" \
    "pillow==10.3.0" \
    "lightning>=2.2.0" \
    "torchmetrics>=1.3.0"

# Install NVIDIA Transformer Engine from source (for fp8 support)
RUN pip install --no-cache-dir git+https://github.com/NVIDIA/TransformerEngine.git@v0.13 --force-reinstall

# Install additional Python dependencies from your project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files into the container
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV SHMEM_SIZE=8G

# Expose FastAPI port (optional for serving)
EXPOSE 8000

# Default entrypoint (can be overridden by your shell script)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--timeout-keep-alive", "120"]
