#!/bin/bash

# Exit on error
set -e

# Parameters
LOCAL_WORKSPACE="/home/tarik-devh/Projects/llm_finetune_nemo"
CONTAINER_WORKSPACE="/workspace"
# Note: You need to authenticate with NVIDIA NGC first:
# docker login nvcr.io
# Username: $oauthtoken
# Password: Your NGC API key
DOCKER_IMAGE="nvcr.io/nvidia/nemo:25.04.nemotron-h"
TRAINING_SCRIPT="/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py"

# Verify config file exists
if [ ! -f "${LOCAL_WORKSPACE}/configs/training_config.yaml" ]; then
    echo "Error: training_config.yaml not found at ${LOCAL_WORKSPACE}/configs/training_config.yaml"
    exit 1
fi

# Run container with config verification
docker run --gpus all \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${LOCAL_WORKSPACE}:${CONTAINER_WORKSPACE}" \
    -w "${CONTAINER_WORKSPACE}" \
    "${DOCKER_IMAGE}" \
    bash -c "
        echo '=== Checking workspace structure ===';
        ls -la /workspace;
        echo '=== Checking configs directory ===';
        ls -la /workspace/configs;
        echo '=== Checking data directory ===';
        ls -la /workspace/data;
        echo '=== Checking data/train directory ===';
        ls -la /workspace/data/train;
        echo '=== Checking data/val directory ===';
        ls -la /workspace/data/val;
        echo '=== Verifying config file ===';
        cat /workspace/configs/training_config.yaml;
        echo '=== Starting training ===';
        cd /opt/NeMo/examples/nlp/language_modeling/tuning/;
        python megatron_gpt_finetuning.py \
            trainer.devices=1 \
            trainer.accelerator=gpu \
            trainer.precision=bf16-mixed \
            model.restore_from_path=/workspace/models/megatron_gpt_345m/megatron_gpt_345m.nemo \
            model.data.train_ds.file_names=[/workspace/data/train/data.jsonl] \
            +model.data.train_ds.concat_sampling_probabilities=[1.0] \
            model.data.validation_ds.file_names=[/workspace/data/val/data.jsonl] \
            +model.data.validation_ds.concat_sampling_probabilities=[1.0] \
            exp_manager.exp_dir=/workspace/results \
            ++config_path=/workspace/configs \
            ++config_name=training_config.yaml
    "
