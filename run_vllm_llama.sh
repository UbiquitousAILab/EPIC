#!/bin/bash

# Input arguments: GPU ID (e.g., 0,1) and port number
GPUS=$1
PORT=$2

if [ -z "$GPUS" ] || [ -z "$PORT" ]; then
  echo "Usage: sh run_vllm.sh <GPU_IDs> <PORT>"
  echo "Example: sh run_vllm.sh 4,5 8010"
  exit 1
fi

# GPU configuration
export CUDA_VISIBLE_DEVICES=$GPUS
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HUGGINGFACE_TOKEN={your_huggingface_token_here}

# Calculate number of GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

huggingface-cli login --token $HUGGINGFACE_TOKEN

echo "🚀 Starting vLLM server with GPU(s): $GPUS on port: $PORT"

# Run vLLM server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size $NUM_GPUS \
  --max_model_len 8192 \
  --gpu-memory-utilization 0.85 \
  --port $PORT \
  --seed 0
