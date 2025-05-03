# Use the official NeMo container with preinstalled torch stack and NeMo 2.2.1
FROM nvcr.io/nvidia/nemo:25.04.nemotron-h

# Set working directory
WORKDIR /app

# Install CUDA toolkit headers if needed for nvcc or custom ops
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

# First, uninstall conflicting packages
RUN pip uninstall -y torch torchaudio torchvision fastapi pydantic transformers numba urllib3

# Install specific versions of core dependencies
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchaudio==2.2.0 \
    torchvision==0.17.0 \
    fastapi==0.115.4 \
    pydantic>=2.9.1 \
    transformers>=4.47.0,<4.48.0 \
    numba>=0.59.1,<0.61.0a0 \
    urllib3>=1.21.1,<1.27 \
    pillow==10.3.0 --force-reinstall

# Install Transformer Engine from source
RUN pip install --no-cache-dir git+https://github.com/NVIDIA/TransformerEngine.git@v0.13 --force-reinstall

# Install additional Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV SHMEM_SIZE=8G

# Set ulimit for shared memory
RUN ulimit -l unlimited

# Expose FastAPI port
EXPOSE 8000

# Launch the FastAPI app with increased workers and timeout
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--timeout-keep-alive", "120"]
