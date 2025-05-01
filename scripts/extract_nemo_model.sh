#!/bin/bash

set -e  # Exit on any error

# Clean up any existing models directory
if [ -d "models" ]; then
    echo "Removing existing models directory..."
    rm -rf models/
fi

MODEL_NAME="megatron_gpt_345m"
MODEL_URL="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/megatron_gpt_345m/1/files?redirect=true&path=megatron_gpt_345m.nemo"
OUTPUT_DIR="models"
NEMO_FILE="$OUTPUT_DIR/$MODEL_NAME.nemo"
MODEL_DIR="$OUTPUT_DIR/$MODEL_NAME"
FINAL_NEMO_FILE="$MODEL_DIR/$MODEL_NAME.nemo"

# Create output directory
mkdir -p "$MODEL_DIR"

# Function to check if model is already extracted
check_extraction() {
    if [ -d "$MODEL_DIR" ] && [ -f "$FINAL_NEMO_FILE" ]; then
        echo "Model already extracted at $MODEL_DIR"
        return 0
    fi
    return 1
}

# Check if model needs to be downloaded and extracted
if [ -f "$FINAL_NEMO_FILE" ] && check_extraction; then
    echo "Model already exists and is extracted"
    exit 0
fi

# Download the model if needed
if [ ! -f "$NEMO_FILE" ]; then
    echo "Downloading model from NGC to $NEMO_FILE..."
    curl -L "$MODEL_URL" -o "$NEMO_FILE"
    if [ ! -f "$NEMO_FILE" ]; then
        echo "Error: Failed to download model"
        exit 1
    fi
    echo "Download complete!"
fi

# Move the .nemo file to the correct location
echo "Moving model to correct location..."
mv "$NEMO_FILE" "$FINAL_NEMO_FILE"

# Verify extraction
if check_extraction; then
    echo "Model download and extraction complete!"
    echo "- NEMO file: $FINAL_NEMO_FILE"
    echo "- Model directory: $MODEL_DIR"
    echo "Directory structure:"
    tree "$OUTPUT_DIR"
else
    echo "Error: Model extraction verification failed"
    exit 1
fi
