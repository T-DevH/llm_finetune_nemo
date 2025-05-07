#!/bin/bash

# Exit on error
set -e

# Configuration
SERVER_PORT=8000
SERVER_HOST="localhost"

# Function to make API calls
make_request() {
    local endpoint=$1
    local method=$2
    local data=$3
    
    echo "Testing $endpoint..."
    echo "Request:"
    echo "$data" | jq '.' 2>/dev/null || echo "$data"
    echo -e "\nResponse:"
    
    if [ "$method" = "GET" ]; then
        curl -s "http://${SERVER_HOST}:${SERVER_PORT}${endpoint}"
    else
        curl -s -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "http://${SERVER_HOST}:${SERVER_PORT}${endpoint}"
    fi
    
    echo -e "\n----------------------------------------\n"
}

# Test server health
make_request "/health" "GET" ""

# Test base model
make_request "/generate" "POST" '{
    "prompt": "This is a test of the Megatron GPT model:",
    "max_tokens": 50,
    "temperature": 0.7,
    "use_lora": false
}'

# Test LoRA model
make_request "/generate" "POST" '{
    "prompt": "This is a test of the Megatron GPT model with LoRA:",
    "max_tokens": 50,
    "temperature": 0.7,
    "use_lora": true
}' 