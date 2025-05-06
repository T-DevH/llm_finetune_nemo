# Use the NeMo base container with Megatron-LM & transformer-engine preinstalled
FROM nvcr.io/nvidia/nemo:25.04.rc2

# Set working directory
WORKDIR /app

# Install Python 3.11 and its development files
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        gnupg2 \
        curl \
        build-essential && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-nvcc-12-1 \
        cuda-cudart-dev-12-1 && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.1-1_all.deb

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Uninstall packages that might conflict
RUN pip uninstall -y \
    torchaudio \
    torchvision \
    fastapi \
    pydantic \
    transformers \
    numba \
    urllib3 \
    pytorch-lightning \
    lightning \
    numpy

# Update pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install specific versions of packages that are compatible with NeMo 25.04
RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    "torch==2.2.2" \
    "torchaudio==2.2.2" \
    "torchvision==0.17.2" \
    "fastapi==0.115.4" \
    "uvicorn==0.22.0" \
    "pydantic>=2.9.1,<2.11" \
    "pydantic-settings==2.1.0" \
    "python-multipart==0.0.6" \
    "tensorboard>=2.13.0" \
    "python-dotenv==1.0.0" \
    "requests" \
    "lightning>=2.2.0,<2.3.0" \
    "omegaconf>=2.3.0"

# Install NVIDIA packages
RUN pip install --no-cache-dir nvidia-pyindex && \
    pip install --no-cache-dir nvidia-tensorrt

# Copy source files into the container
COPY ./src /app/src

# Set Python path to include the app directory
ENV PYTHONPATH=/app

# Default command to run the FastAPI server
CMD ["python", "src/deployment/app/server.py"]
