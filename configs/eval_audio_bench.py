#!/usr/bin/env python3
"""
Audio Benchmark Evaluation Configuration
Mirrors GTA's eval_gta_bench.py structure.
"""

from pathlib import Path

# ============================================================================
# Base Paths
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "audio_dataset"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ============================================================================
# Model Configurations
# ============================================================================

MODELS = {
    "qwen2-audio-7b": {
        "abbr": "qwen2-audio-7b",
        "type": "vllm",  # or "local" for transformers
        "model_path": "Qwen/Qwen2-Audio-7B-Instruct",
        "api_base": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "max_turns": 10,
        "temperature": 0.1,
        "max_tokens": 1024,
    },
    "qwen25-omni-7b": {
        "abbr": "qwen25-omni-7b",
        "type": "vllm",
        "model_path": "Qwen/Qwen2.5-Omni-7B-Instruct",
        "api_base": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "max_turns": 10,
        "temperature": 0.1,
        "max_tokens": 1024,
    },
}

# Default model
DEFAULT_MODEL = "qwen2-audio-7b"

# ============================================================================
# Tool Server Configuration
# ============================================================================

TOOL_SERVER = {
    "host": "0.0.0.0",
    "port": 16181,
    "url": "http://localhost:16181",
}

# Path to tool metadata
TOOLMETA_PATH = DATA_DIR / "toolmeta.json"

# ============================================================================
# Dataset Configuration
# ============================================================================

DATASET = {
    "path": DATA_DIR / "dataset.json",
    "audio_base": DATA_DIR / "audio_assets",
}

# ============================================================================
# Evaluation Modes
# ============================================================================

# Mode: "every" (end-to-end) or "every_with_gt" (step-by-step)
EVAL_MODE = "every"

# Step-by-step configuration
STEP_BY_STEP_CONFIG = {
    "mode": "every_with_gt",
    "use_toolmeta": True,
    "use_tool_server": False,
}

# End-to-end configuration
END_TO_END_CONFIG = {
    "mode": "every",
    "use_toolmeta": False,
    "use_tool_server": True,
}

# ============================================================================
# Metrics Configuration
# ============================================================================

# Tool categories for F1 computation
TOOL_CATEGORIES = {
    "Perception": [
        "whisper", "funasr", "language_id", "silero_vad",
        "gender_detection", "audio_caption", "conette"
    ],
    "Analysis": [
        "nisqa", "deepfake_audio", "speaker_verification", "resemblyzer",
        "diarizen", "pyannote_segmentation", "nemo_diarizer",
        "speechmos", "wav2vec2_quality", "audioldm_eval", "muq"
    ],
    "Transformation": [
        "demucs", "sepformer", "sepformer_wham", "asteroid_separate",
        "deepfilternet", "sgmse", "sb_sgmse", "espnet_enhance"
    ],
    "Detection": [
        "audioseal", "chromaprint", "audio_fingerprint", "clap_embed", "r1_aqa"
    ],
}

# Metric weights for overall score
METRIC_WEIGHTS = {
    "instruction_accuracy": 0.15,
    "tool_accuracy": 0.25,
    "argument_accuracy": 0.20,
    "summary_accuracy": 0.15,
    "answer_accuracy": 0.25,
}

# ============================================================================
# Inference Configuration
# ============================================================================

INFERENCE_CONFIG = {
    "batch_size": 8,
    "max_workers": 4,
    "timeout": 120,  # seconds per task
    "retry_count": 2,
}
