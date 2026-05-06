#!/usr/bin/env python3
"""
Generate tool_registry.json for the Audio Authenticity Benchmark.
Combines programmatic introspection of tool classes with manual metadata.
"""
import json
import os
import sys
from typing import Dict, Any

# Ensure we can import audiotoolagent
sys.path.append("${BENCHMARK_ROOT}")
from audiotoolagent.orchestrators.qwen25_omni import create_all_tools

# Manual Metadata Enrichment
METADATA_OVERRIDE = {
    "deepfake_audio": {
        "category": "Security & Authenticity",
        "model_id": "SSL_Anti-spoofing",
        "backend": "Local (PyTorch)",
        "license": "MIT",
        "paper_link": "https://arxiv.org/abs/2102.09914",
        "latency": "Medium (~1s/sample)",
    },
    "audioseal": {
        "category": "Security & Authenticity",
        "model_id": "facebook/audioseal",
        "backend": "Local (HuggingFace)",
        "license": "CC-BY-NC-4.0",
        "paper_link": "https://arxiv.org/abs/2401.17264",
        "latency": "Fast (<0.5s)",
    },
    "whisper": {
        "category": "Perception",
        "model_id": "openai/whisper-large-v3",
        "backend": "Local (faster-whisper)",
        "license": "MIT",
        "paper_link": "https://arxiv.org/abs/2212.04356",
        "latency": "Variable",
    },
    "diarizen": {
        "category": "Perception",
        "model_id": "pyannote/speaker-diarization-3.1",
        "backend": "Local (PyTorch)",
        "license": "MIT",
        "paper_link": "https://arxiv.org/abs/2304.09128", 
        "latency": "Slow",
    },
    "language_id": {
        "category": "Perception",
        "model_id": "speechbrain/lang-id-voxlingua107-ecapa",
        "backend": "Local (SpeechBrain)",
        "license": "Apache 2.0",
        "paper_link": "https://arxiv.org/abs/2012.02518",
        "latency": "Fast",
    },
    "speaker_verification": {
        "category": "Perception",
        "model_id": "speechbrain/spkrec-ecapa-voxceleb",
        "backend": "Local (SpeechBrain)",
        "license": "Apache 2.0",
        "paper_link": "https://arxiv.org/abs/2005.07143",
        "latency": "Fast",
    },
    "silero_vad": {
        "category": "Perception",
        "model_id": "snakers4/silero-vad",
        "backend": "Local (ONNX)",
        "license": "MIT",
        "paper_link": "https://github.com/snakers4/silero-vad",
        "latency": "Real-time",
    },
    "funasr_tool": {
        "category": "Perception",
        "model_id": "alibaba-damo/speech_paraformer-large_asr_nat",
        "backend": "Local (FunASR)",
        "license": "MIT",
        "paper_link": "https://arxiv.org/abs/2305.11013",
        "latency": "Fast",
    },
    "audio_caption": {
        "category": "Perception",
        "model_id": "microsoft/BEATs_iter3_plus_AS2M_finetuned_on_AS2M",
        "backend": "Local (PyTorch/Transformers)",
        "license": "MIT",
        "paper_link": "https://arxiv.org/abs/2212.09058",
    },
    "nemo_diarizer": {
        "category": "Perception",
        "model_id": "nvidia/nemo_diarizer",
        "backend": "Local (NeMo)",
        "license": "Apache 2.0",
        "paper_link": "https://arxiv.org/abs/2110.05267",
    },
    "resemblyzer_tool": {
        "category": "Perception",
        "model_id": "resemblyzer/voice_encoder",
        "backend": "Local (PyTorch)",
        "license": "Apache 2.0",
        "paper_link": "https://arxiv.org/abs/1710.10467", # GE2E
    },
    "pyannote_segmentation": {
        "category": "Perception",
        "model_id": "pyannote/segmentation-3.0",
        "backend": "Local (PyTorch)",
        "license": "MIT",
        "paper_link": "https://arxiv.org/abs/2104.04045",
    },
     "demucs": {
        "category": "Operation",
        "model_id": "facebook/htdemucs",
        "backend": "Local (PyTorch)",
        "license": "MIT",
        "paper_link": "https://arxiv.org/abs/1911.13254",
    },
    "deepfilternet": {
        "category": "Operation",
        "model_id": "Rikorose/DeepFilterNet3",
        "backend": "Local (Rust/PyTorch)",
        "license": "MIT",
        "paper_link": "https://arxiv.org/abs/2305.08227",
    },
    "sepformer_wham": {
        "category": "Operation",
        "model_id": "speechbrain/sepformer-wsj02mix",
        "backend": "Local (SpeechBrain)",
        "license": "Apache 2.0",
        "paper_link": "https://arxiv.org/abs/2010.13154",
    },
    "espnet_enhance": {
        "category": "Operation",
        "model_id": "espnet/enh_train_enh_conv_tasnet",
        "backend": "Local (ESPnet)",
        "license": "Apache 2.0",
        "paper_link": "https://arxiv.org/abs/1812.05944",
    },
    "asteroid_separate": {
        "category": "Operation",
        "model_id": "mpariente/ConvTasNet_WHAM_sepclean",
        "backend": "Local (Asteroid)",
        "license": "MIT",
        "paper_link": "https://arxiv.org/abs/1809.07454",
    },
    "sb_sgmse": {
        "category": "Operation",
        "model_id": "speechbrain/mtl-mimic-voicebank",
        "backend": "Local (SpeechBrain)",
        "license": "Apache 2.0",
        "paper_link": "https://arxiv.org/abs/2303.15572", # SGMSE paper
    },
    "nisqa": {
        "category": "Quality & Reasoning",
        "model_id": "gabrielmittag/NISQA",
        "backend": "Local (PyTorch)",
        "license": "MIT",
        "paper_link": "https://arxiv.org/abs/2104.09494",
        "latency": "Fast",
    },
    "wav2vec2_quality": {
        "category": "Quality & Reasoning",
        "model_id": "facebook/wav2vec2-base",
        "backend": "Local (Transformers)",
        "license": "Apache 2.0",
        "paper_link": "https://arxiv.org/abs/2006.11477",
    },
    "audioldm_eval": {
        "category": "Quality & Reasoning",
        "model_id": "audioldm/focalloss_eval",
        "backend": "Local",
        "license": "MIT",
        "paper_link": "https://arxiv.org/abs/2301.12503",
    },
    "r1_aqa": {
        "category": "Quality & Reasoning",
        "model_id": "Custom-R1-Distill",
        "backend": "Local (VLLM)",
        "license": "Apache 2.0",
        "paper_link": "N/A",
    },
    "chromaprint": {
        "category": "Quality & Reasoning",
        "model_id": "fpcalc",
        "backend": "Local (Binary)",
        "license": "MIT",
        "paper_link": "https://github.com/acoustid/chromaprint",
    },
    "clap_embed": {
         "category": "Quality & Reasoning",
         "model_id": "laion/clap-htsat-unfused",
         "backend": "Local (Transformers)",
         "license": "MIT",
         "paper_link": "https://arxiv.org/abs/2211.06687",
    }
}

def main():
    print("Generating Tool Registry JSON...")
    
    tools = create_all_tools()
    registry = {}
    
    for name, tool in tools.items():
        # Get programmatic info
        desc = getattr(tool, 'description', 'No description available.')
        params = getattr(tool, 'parameters', [])
        
        # Get manual metadata
        meta = METADATA_OVERRIDE.get(name, {})
        
        # Validate parameters format
        inputs_map = {}
        if isinstance(params, list):
            for p in params:
                if isinstance(p, dict):
                    inputs_map[p.get('name', 'unknown')] = p.get('description', '')
                elif isinstance(p, str):
                    # Try parsing if it's a JSON string
                    try:
                        p_dict = json.loads(p)
                        if isinstance(p_dict, dict):
                             inputs_map[p_dict.get('name', 'unknown')] = p_dict.get('description', '')
                    except:
                        pass
        
        entry = {
            "name": name,
            "description": desc,
            "inputs_schema": params,
            # Defaults if missing in metadata
            "category": meta.get("category", "Uncategorized"),
            "model_id": meta.get("model_id", "Unknown"),
            "backend": meta.get("backend", "Local"),
            "license": meta.get("license", "Unknown"),
            "paper_link": meta.get("paper_link", ""),
            "latency": meta.get("latency", "Unknown"),
            "inputs": inputs_map, # Simplified view
            "outputs_schema": {"type": "json", "description": "See tool description for details"}, # Generic for now
            "is_augmented": True
        }
        
        registry[name] = entry
        
    out_path = "${BENCHMARK_ROOT}"
    
    # Save prettified JSON
    with open(out_path, 'w') as f:
        json.dump(registry, f, indent=2)
        
    print(f"✅ Registry saved to {out_path} ({len(registry)} tools)")

if __name__ == "__main__":
    main()
