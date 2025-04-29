#!/bin/bash

set -e  # Exit on any error

# Set model parameters
MODEL_NAME="megatron_gpt_345m"
OUTPUT_DIR="models"
NEMO_FILE="$OUTPUT_DIR/$MODEL_NAME.nemo"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if model already exists
if [ -f "$NEMO_FILE" ]; then
    echo "✅ Model already exists at $NEMO_FILE"
    exit 0
fi

# Download the model
echo "⬇️  Downloading model..."
echo "Using NGC CLI for download..."

# Download using NGC CLI
cd "$OUTPUT_DIR"
ngc registry model download-version "nvidia/nemo/megatron_gpt_345m:1"

# Find the downloaded directory and file
DOWNLOAD_DIR=$(find . -maxdepth 1 -type d -name "megatron_gpt_345m_v1*" | head -n 1)
if [ -n "$DOWNLOAD_DIR" ]; then
    # Move the .nemo file from the download directory
    mv "$DOWNLOAD_DIR"/*.nemo "./$MODEL_NAME.nemo"
    # Clean up download directory
    rm -rf "$DOWNLOAD_DIR"
    echo "✅ Download complete!"
    echo "Model saved to: $NEMO_FILE"
else
    echo "❌ Error: Downloaded directory not found"
    exit 1
fi
