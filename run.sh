#!/bin/bash
#PBS -N nonaudio_run
#PBS -q workq
#PBS -l select=1:ncpus=32:ngpus=2
#PBS -l walltime=48:00:00
#PBS -o ${BENCHMARK_ROOT}
#PBS -e ${BENCHMARK_ROOT}

set -uo pipefail
umask 027

###############################################################################
# run.sh
# - Runs specified non-audio models sequentially (one model at a time)
# - Uses 2 GPUs (tensor parallel) per model for vLLM
# - Uses remaining visible GPUs for tool execution process
# - Samples 100 random queries from dataset_500.json
# - Runs both end_to_end and step_by_step + evaluation for both
###############################################################################

REPO_ROOT="${BENCHMARK_ROOT}"
MODEL_ROOT="${BENCHMARK_ROOT}"
RUNS_ROOT="${REPO_ROOT}/outputs/RUN_SH"

DATASET_FULL="${REPO_ROOT}/data/audio_dataset/dataset_500.json"
TOOLMETA="${REPO_ROOT}/data/audio_dataset/toolmeta.json"
AUDIO_BASE="${REPO_ROOT}/data/audio_dataset/audio_assets"

CONDA_ENV="${CONDA_ENV:-audio_benchmark_vllm}"
VENV_FALLBACK="${REPO_ROOT}/envs/muq_env/bin/activate"

# Controls
RANDOM_QUERY_COUNT="${RANDOM_QUERY_COUNT:-100}"
RANDOM_SEED="${RANDOM_SEED:-$RANDOM}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
MAX_TURNS="${MAX_TURNS:-8}"
INFER_TIMEOUT_SEC="${INFER_TIMEOUT_SEC:-7200}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-8192}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${CONTEXT_WINDOW}}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
TP_SIZE="${TP_SIZE:-2}"

export NATIVE_MODEL_API_KEY="${NATIVE_MODEL_API_KEY:-EMPTY}"
export OPENAI_MODEL_MAX_LEN="${OPENAI_MODEL_MAX_LEN:-${CONTEXT_WINDOW}}"
export MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-${CONTEXT_WINDOW}}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-${CONTEXT_WINDOW}}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_COMPILE_DISABLE=1

# Requested model list (display_name|directory_name)
MODEL_SPECS=(
  "deepseek-llm-7b-chat|deepseek-llm-7b-chat"
  "Qwen3-8B|Qwen3-8B"
  "Qwen3-14B|Qwen3-14B"
  "Qwen3-32B|Qwen3-32B"
  "internlm2 5-7b|internlm2_5-7b"
  "internlm2 5-20b|internlm2_5-20b"
  "internlm3-8b|internlm3-8b-instruct"
)

timestamp() { date +%Y%m%d_%H%M%S; }
log() { echo "[$(date +'%F %T')] $*"; }

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g; s/_+/_/g'
}

activate_runtime_env() {
  if [[ -f "/home/soft/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source /home/soft/anaconda3/etc/profile.d/conda.sh
  fi

  if command -v conda >/dev/null 2>&1; then
    if conda activate "${CONDA_ENV}" >/dev/null 2>&1; then
      log "Activated conda env: ${CONDA_ENV}"
      return 0
    fi
  fi

  if [[ -f "${VENV_FALLBACK}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_FALLBACK}"
    log "Activated venv fallback: ${VENV_FALLBACK}"
    return 0
  fi

  echo "ERROR: could not activate runtime environment"
  return 1
}

require_python_deps() {
  local py="$1"
  "${py}" - <<'PY'
import importlib.util
import sys
required = ["torch", "qwen_agent", "peft", "openai", "vllm"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("ERROR: missing python packages:", ", ".join(missing))
    sys.exit(1)
print("Python dependency check passed.")
PY
}

wait_for_vllm() {
  local base_url="$1"
  local tries=120
  for ((i=1; i<=tries; i++)); do
    if curl -fsS "${base_url}/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

extract_first_model_id() {
  "${PYTHON_BIN}" - <<'PY'
import json
import sys
obj = json.load(sys.stdin)
data = obj.get("data") or []
print(data[0].get("id", "") if data else "")
PY
}

cleanup() {
  log "Cleanup: terminating child processes..."
  pkill -P $$ 2>/dev/null || true
}
trap cleanup EXIT

run_inference_vllm_mode() {
  # args: model_name model_path mode output_json log_dir
  local name="$1"
  local model_path="$2"
  local mode="$3"
  local output_json="$4"
  local log_dir="$5"

  local mode_dir="${log_dir}/${mode}"
  mkdir -p "${mode_dir}"

  local port=$((8100 + MODEL_GPU_IDS_ARR[0]))
  local base_url="http://127.0.0.1:${port}/v1"
  local model_id=""
  local rc=0

  log "Launching vLLM for ${name} on ${MODEL_GPU_IDS} (TP=${TP_SIZE})"
  CUDA_VISIBLE_DEVICES="${MODEL_GPU_IDS}" \
  "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model "${model_path}" \
    --port "${port}" \
    --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TP_SIZE}" \
    > "${mode_dir}/vllm_server.log" 2>&1 &
  local vllm_pid=$!

  if ! wait_for_vllm "${base_url}"; then
    log "FAIL: vLLM not ready for ${name} (${mode})"
    kill "${vllm_pid}" 2>/dev/null || true
    wait "${vllm_pid}" 2>/dev/null || true
    return 71
  fi

  model_id="$(curl -fsS "${base_url}/models" | extract_first_model_id)"
  if [[ -z "${model_id}" ]]; then
    log "FAIL: cannot resolve model id from vLLM for ${name} (${mode})"
    kill "${vllm_pid}" 2>/dev/null || true
    wait "${vllm_pid}" 2>/dev/null || true
    return 72
  fi

  log "Running inference ${name} (${mode}) via vLLM model_id='${model_id}' using tool GPUs=${TOOL_GPU_IDS}"
  if command -v timeout >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="${TOOL_GPU_IDS}" \
    timeout --signal=TERM "${INFER_TIMEOUT_SEC}" \
      "${PYTHON_BIN}" src/run_inference.py \
      --mode "${mode}" \
      --provider vllm \
      --model "${model_id}" \
      --api_base "${base_url}" \
      --api_key "EMPTY" \
      --in_process \
      --toolmeta "${TOOLMETA}" \
      --dataset "${DATASET_SUBSET}" \
      --audio_base "${AUDIO_BASE}" \
      --max_turns "${MAX_TURNS}" \
      --checkpoint "${CHECKPOINT_EVERY}" \
      --resume \
      --output "${output_json}" \
      > "${mode_dir}/run_inference_vllm.log" 2>&1 || rc=$?
  else
    CUDA_VISIBLE_DEVICES="${TOOL_GPU_IDS}" \
      "${PYTHON_BIN}" src/run_inference.py \
      --mode "${mode}" \
      --provider vllm \
      --model "${model_id}" \
      --api_base "${base_url}" \
      --api_key "EMPTY" \
      --in_process \
      --toolmeta "${TOOLMETA}" \
      --dataset "${DATASET_SUBSET}" \
      --audio_base "${AUDIO_BASE}" \
      --max_turns "${MAX_TURNS}" \
      --checkpoint "${CHECKPOINT_EVERY}" \
      --resume \
      --output "${output_json}" \
      > "${mode_dir}/run_inference_vllm.log" 2>&1 || rc=$?
  fi

  log "Stopping vLLM for ${name} (${mode})"
  kill "${vllm_pid}" 2>/dev/null || true
  wait "${vllm_pid}" 2>/dev/null || true

  return "${rc}"
}

run_eval_pair() {
  # args: model_name model_out log_dir
  local name="$1"
  local model_out="$2"
  local log_dir="$3"

  local e2e_results="${model_out}/results.json"
  local step_results="${model_out}/results_step_by_step.json"
  local e2e_prefix="${model_out}/evaluation_report_end_to_end"
  local step_prefix="${model_out}/evaluation_report_step_by_step"
  local e2e_log="${log_dir}/evaluate_end_to_end.log"
  local step_log="${log_dir}/evaluate_step_by_step.log"
  local final_log="${log_dir}/evaluate.log"

  local rc_eval_e2e=0
  local rc_eval_step=0

  if [[ -f "${e2e_results}" ]]; then
    "${PYTHON_BIN}" src/evaluate.py --mode end_to_end --results "${e2e_results}" --output "${e2e_prefix}" > "${e2e_log}" 2>&1 || rc_eval_e2e=$?
  else
    echo "[${name}] End-to-end evaluation skipped: ${e2e_results} not found." > "${e2e_log}"
  fi

  if [[ -f "${step_results}" ]]; then
    "${PYTHON_BIN}" src/evaluate.py --mode step_by_step --results "${step_results}" --output "${step_prefix}" > "${step_log}" 2>&1 || rc_eval_step=$?
  else
    echo "[${name}] Step-by-step evaluation skipped: ${step_results} not found." > "${step_log}"
  fi

  {
    echo "============================================================"
    echo "Model Evaluation Summary: ${name}"
    echo "Generated: $(date +'%F %T')"
    echo "============================================================"
    echo
    echo "---------------- End-to-End Evaluation ----------------"
    cat "${e2e_log}"
    echo
    echo "--------------- Step-by-Step Evaluation ---------------"
    cat "${step_log}"
  } > "${final_log}"

  [[ "${rc_eval_e2e}" -eq 0 && "${rc_eval_step}" -eq 0 ]]
}

# setup
cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}"; exit 2; }
source ~/.bashrc >/dev/null 2>&1 || true

mkdir -p "${RUNS_ROOT}"
RUN_TS="$(timestamp)"
OUT_ROOT="${RUNS_ROOT}/run_${RUN_TS}"
mkdir -p "${OUT_ROOT}"/{logs,status,per_model}

activate_runtime_env || exit 3
PYTHON_BIN="$(command -v python || true)"
[[ -n "${PYTHON_BIN}" ]] || { echo "ERROR: python not found"; exit 3; }
require_python_deps "${PYTHON_BIN}" || exit 3

export PYTHONPATH="${REPO_ROOT}/AudioToolAgent:${REPO_ROOT}/src:${PYTHONPATH:-}"

# GPU split: first 2 for model, remaining for tools
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a ALL_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
  ALL_GPUS=(0 1 2 3)
fi

if [[ "${#ALL_GPUS[@]}" -lt 2 ]]; then
  echo "ERROR: need at least 2 GPUs visible"
  exit 4
fi

MODEL_GPU_IDS_ARR=("${ALL_GPUS[0]}" "${ALL_GPUS[1]}")
MODEL_GPU_IDS="${MODEL_GPU_IDS_ARR[0]},${MODEL_GPU_IDS_ARR[1]}"

if [[ "${#ALL_GPUS[@]}" -ge 3 ]]; then
  TOOL_GPU_IDS_ARR=("${ALL_GPUS[@]:2}")
  TOOL_GPU_IDS="$(IFS=,; echo "${TOOL_GPU_IDS_ARR[*]}")"
else
  # fallback: tools share second model GPU if only 2 GPUs available
  TOOL_GPU_IDS="${ALL_GPUS[1]}"
fi

log "Run root: ${OUT_ROOT}"
log "Model root: ${MODEL_ROOT}"
log "Dataset full: ${DATASET_FULL}"
log "Model GPUs: ${MODEL_GPU_IDS}"
log "Tool GPUs: ${TOOL_GPU_IDS}"
log "Random query count: ${RANDOM_QUERY_COUNT} (seed=${RANDOM_SEED})"

# build random 100-query subset
DATASET_SUBSET="${OUT_ROOT}/dataset_random_${RANDOM_QUERY_COUNT}.json"
"${PYTHON_BIN}" - <<PY
import json, random
src = "${DATASET_FULL}"
out = "${DATASET_SUBSET}"
n = int("${RANDOM_QUERY_COUNT}")
seed = int("${RANDOM_SEED}")
with open(src) as f:
    data = json.load(f)
if not isinstance(data, list):
    raise SystemExit(f"Dataset must be a list, got {type(data)}")
if n > len(data):
    raise SystemExit(f"Requested {n} > dataset size {len(data)}")
random.seed(seed)
subset = random.sample(data, n)
with open(out, "w") as f:
    json.dump(subset, f, indent=2)
print(f"Wrote {len(subset)} random samples to {out}")
PY

# optional tool status check
if [[ -f "${REPO_ROOT}/scripts/check_tools_status.py" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_tools_status.py" > "${OUT_ROOT}/logs/check_tools_status.log" 2>&1 || true
fi

# main loop (one model at a time)
for spec in "${MODEL_SPECS[@]}"; do
  name="${spec%%|*}"
  dir="${spec##*|}"
  tag="$(slugify "${name}")"

  model_path="${MODEL_ROOT}/${dir}"
  model_out="${OUT_ROOT}/per_model/${tag}"
  log_dir="${OUT_ROOT}/logs/${tag}"
  status_file="${OUT_ROOT}/status/${tag}.status"
  mkdir -p "${model_out}" "${log_dir}"

  {
    echo "name=${name}"
    echo "model_dir=${dir}"
    echo "model_path=${model_path}"
    echo "gpu_model=${MODEL_GPU_IDS}"
    echo "gpu_tools=${TOOL_GPU_IDS}"
    echo "start_ts=$(date +'%F %T')"
  } > "${status_file}"

  if [[ ! -d "${model_path}" ]]; then
    log "SKIP: missing model directory for ${name}: ${model_path}"
    echo "state=SKIP_missing_model_dir" >> "${status_file}"
    echo "end_ts=$(date +'%F %T')" >> "${status_file}"
    continue
  fi

  log "START: ${name}"

  rc_e2e=0
  rc_step=0
  rc_eval=0

  run_inference_vllm_mode "${name}" "${model_path}" "end_to_end" "${model_out}/results.json" "${log_dir}" || rc_e2e=$?
  run_inference_vllm_mode "${name}" "${model_path}" "step_by_step" "${model_out}/results_step_by_step.json" "${log_dir}" || rc_step=$?

  run_eval_pair "${name}" "${model_out}" "${log_dir}" || rc_eval=$?

  echo "infer_rc_end_to_end=${rc_e2e}" >> "${status_file}"
  echo "infer_rc_step_by_step=${rc_step}" >> "${status_file}"
  echo "eval_rc=${rc_eval}" >> "${status_file}"
  echo "end_ts=$(date +'%F %T')" >> "${status_file}"

  if [[ "${rc_e2e}" -eq 0 && "${rc_step}" -eq 0 ]]; then
    echo "state=OK" >> "${status_file}"
    log "DONE: ${name} OK"
  else
    echo "state=FAIL_infer" >> "${status_file}"
    log "DONE: ${name} FAILED (e2e=${rc_e2e}, step=${rc_step}, eval=${rc_eval})"
  fi
done

log "All done. Outputs: ${OUT_ROOT}"
