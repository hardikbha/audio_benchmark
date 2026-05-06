#!/bin/bash
# Start vLLM server for Audio LLMs
# Usage: ./start_vllm.sh [model_name]

MODEL=${1:-"Qwen/Qwen2-Audio-7B-Instruct"}
PORT=8000
GPU_MEMORY=0.8

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Starting vLLM Server"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU Memory: $GPU_MEMORY"
echo "=========================================="

# Activate conda environment
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate vllm 2>/dev/null || conda activate base
fi

# Check if model is local path or HuggingFace
if [[ "$MODEL" == /* ]]; then
    echo "Using local model path: $MODEL"
else
    echo "Using HuggingFace model: $MODEL"
fi

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEMORY" \
    --max-model-len 8192 \
    --dtype auto
