FROM nvcr.io/nvidia/nemo:23.10

WORKDIR /app

# Install CUDA toolkit headers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cuda-nvcc-12-1 \
    cuda-cudart-dev-12-1 \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-keyring_1.1-1_all.deb

# Install specific versions of PyTorch packages first
RUN pip install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --force-reinstall

# Install transformer-engine from source with specific version
RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@v0.13 --force-reinstall

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Create model directory
RUN mkdir -p /app/model

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]