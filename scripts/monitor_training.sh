#!/bin/bash

# Check if logs directory exists
if [ ! -d "../logs" ]; then
    echo "Error: logs directory not found"
    exit 1
fi

# Find the most recent log file
LATEST_LOG=$(ls -t ../logs/*.log | head -n1)

if [ -z "$LATEST_LOG" ]; then
    echo "No log files found in logs directory"
    exit 1
fi

echo "Monitoring training progress from: $LATEST_LOG"
echo "Press Ctrl+C to stop monitoring"
echo "----------------------------------------"

# Function to display metrics in a formatted way
display_metrics() {
    clear
    echo "Training Progress"
    echo "----------------------------------------"
    echo "Step: $1"
    echo "Epoch: $2"
    echo "Training Loss: $3"
    echo "Validation Loss: $4"
    echo "Learning Rate: $5"
    echo "GPU Memory Usage: $6"
    echo "----------------------------------------"
    echo "Last Update: $(date)"
}

# Monitor the log file
tail -f "$LATEST_LOG" | while read -r line; do
    # Extract metrics using grep and sed
    if [[ $line == *"step"* ]]; then
        step=$(echo "$line" | grep -oP 'step: \K[0-9]+')
        epoch=$(echo "$line" | grep -oP 'epoch: \K[0-9]+')
        train_loss=$(echo "$line" | grep -oP 'loss: \K[0-9.]+')
        val_loss=$(echo "$line" | grep -oP 'val_loss: \K[0-9.]+')
        lr=$(echo "$line" | grep -oP 'lr: \K[0-9.e-]+')
        gpu_mem=$(echo "$line" | grep -oP 'gpu_mem: \K[0-9.]+')
        
        if [ ! -z "$step" ]; then
            display_metrics "$step" "$epoch" "$train_loss" "$val_loss" "$lr" "$gpu_mem"
        fi
    fi
done 