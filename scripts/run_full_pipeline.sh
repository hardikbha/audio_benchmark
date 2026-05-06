#!/bin/bash
# Audio Benchmark Pipeline
# Usage: ./run_full_pipeline.sh [MODEL] [API_BASE] [LIMIT] [OPTIONS]
#
# Examples:
#   ./run_full_pipeline.sh gpt-4o https://api.openai.com/v1
#   ./run_full_pipeline.sh Qwen/Qwen2.5-7B-Instruct http://localhost:8000/v1 100 --in_process

set -e

# Arguments
MODEL_NAME="${1:-gpt-4o}"
API_BASE="${2:-http://localhost:8000/v1}"
LIMIT="${3:-}"
EXTRA_ARGS="${@:4}"

# Paths
DATASET="data/audio_dataset/dataset.json"
AUDIO_BASE="data/audio_dataset/audio_assets"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/benchmark_${MODEL_NAME//\//_}_${TIMESTAMP}"

echo "============================================================"
echo "  AUDIO BENCHMARK PIPELINE"
echo "============================================================"
echo "Model:     $MODEL_NAME"
echo "API Base:  $API_BASE"
echo "Dataset:   $DATASET"
echo "Output:    $OUTPUT_DIR"
echo "Limit:     ${LIMIT:-all}"
echo "Extra:     $EXTRA_ARGS"
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python src/run_inference.py \
    --mode end_to_end \
    --dataset $DATASET \
    --audio_base $AUDIO_BASE \
    --api_base $API_BASE \
    --model $MODEL_NAME \
    --checkpoint 10 \
    --output $OUTPUT_DIR/results.json"

# Add limit if specified
if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Add extra args
if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

# Step 1: Inference
echo "[1/3] Running Inference..."
echo "------------------------------------------------------------"
$CMD 2>&1 | tee "$OUTPUT_DIR/inference.log"
echo "[1/3] Inference complete."
echo ""

# Step 2: Evaluation
echo "[2/3] Evaluating Results..."
echo "------------------------------------------------------------"
python src/evaluate.py \
    --results "$OUTPUT_DIR/results.json" \
    --mode end_to_end \
    --output "$OUTPUT_DIR/evaluation_report" \
    2>&1 | tee "$OUTPUT_DIR/evaluation.log"
echo "[2/3] Evaluation complete."
echo ""

# Step 3: Summary
echo "[3/3] Generating Summary..."
echo "------------------------------------------------------------"

if [ -f "$OUTPUT_DIR/evaluation_report.metrics.json" ]; then
    python3 -c "
import json

with open('$OUTPUT_DIR/evaluation_report.metrics.json') as f:
    report = json.load(f)

metrics = report.get('metrics', {})
print(f\"Answer Accuracy: {metrics.get('ans_acc', 0)*100:.1f}%\")
print(f\"Tool Accuracy:   {metrics.get('tool_acc', 0)*100:.1f}%\")
print(f\"Avg Tool Calls:  {metrics.get('avg_tool_calls', 0):.1f}\")
"
else
    echo "No evaluation report found."
fi

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "============================================================"
echo "Results: $OUTPUT_DIR/results.json"
echo "Report:  $OUTPUT_DIR/evaluation_report.metrics.json"
echo "============================================================"
