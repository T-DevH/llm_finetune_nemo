#!/bin/bash

set -e  # Exit on any error

# Set model parameters
MODEL_NAME="megatron_gpt_345m"
MODEL_URL="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/megatron_gpt_345m/1/files?redirect=true&path=megatron_gpt_345m.nemo"
OUTPUT_DIR="models"
NEMO_FILE="$OUTPUT_DIR/$MODEL_NAME.nemo"
EXTRACT_DIR="$OUTPUT_DIR/$MODEL_NAME"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to check if model is already extracted
check_extraction() {
    if [ -d "$EXTRACT_DIR" ] && [ -f "$EXTRACT_DIR/model_config.yaml" ]; then
        echo "Model already extracted at $EXTRACT_DIR"
        return 0
    fi
    return 1
}

# Check if model needs to be downloaded and extracted
if [ -f "$NEMO_FILE" ] && check_extraction; then
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

# Remove existing extracted directory if it exists
if [ -d "$EXTRACT_DIR" ]; then
    echo "Removing existing extracted model..."
    rm -rf "$EXTRACT_DIR"
fi

# Extract the model
echo "Extracting model to $EXTRACT_DIR..."
python3 << END
import os
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

try:
    # Load and extract model
    model = MegatronGPTModel.restore_from("$NEMO_FILE")
    os.makedirs("$EXTRACT_DIR", exist_ok=True)
    model.save_to("$EXTRACT_DIR")
    print("Model extracted successfully to $EXTRACT_DIR")
except Exception as e:
    print(f"Error during model extraction: {str(e)}")
    exit(1)
END

# Verify extraction
if check_extraction; then
    echo "Model download and extraction complete!"
    echo "- NEMO file: $NEMO_FILE"
    echo "- Extracted model: $EXTRACT_DIR"
else
    echo "Error: Model extraction verification failed"
    exit 1
fi 