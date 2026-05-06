#!/bin/bash
# Setup isolated conda environments for tools with conflicting dependencies
# This script creates separate environments for tools that have numpy/torch conflicts

set -e

echo "=============================================="
echo "Setting up isolated conda environments"
echo "=============================================="

# Ensure conda is initialized
# source "$(conda info --base)/etc/profile.d/conda.sh"
source /home/soft/anaconda3/etc/profile.d/conda.sh

# ====================================
# Environment 1: whisper_env
# For: whisper, silero_vad, language_id, chromaprint
# ====================================
echo ""
echo "[1/3] Creating whisper_env..."

if conda env list | grep -q "whisper_env"; then
    echo "       whisper_env already exists, skipping..."
else
    conda create -n whisper_env python=3.10 -y
    conda activate whisper_env
    
    pip install numpy==1.24.0
    pip install torch==2.0.1 torchaudio==2.0.2
    pip install transformers
    pip install openai-whisper
    pip install speechbrain
    pip install librosa soundfile
    pip install faster-whisper
    
    # Install AudioToolAgent in this env
    cd ${BENCHMARK_ROOT}
    pip install -e . --no-deps
    
    conda deactivate
    echo "       whisper_env created ✓"
fi

# ====================================
# Environment 2: nisqa_env
# For: nisqa, speechmos
# ====================================
echo ""
echo "[2/3] Creating nisqa_env..."

if conda env list | grep -q "nisqa_env"; then
    echo "       nisqa_env already exists, skipping..."
else
    conda create -n nisqa_env python=3.10 -y
    conda activate nisqa_env
    
    pip install numpy==1.24.0
    pip install torch==2.0.1 torchaudio==2.0.2
    pip install transformers
    pip install librosa soundfile
    
    # NISQA specific
    pip install nisqa
    
    # Install AudioToolAgent in this env
    cd ${BENCHMARK_ROOT}
    pip install -e . --no-deps
    
    conda deactivate
    echo "       nisqa_env created ✓"
fi

# ====================================
# Environment 3: funasr_env
# For: funasr
# ====================================
echo ""
echo "[3/3] Creating funasr_env..."

if conda env list | grep -q "funasr_env"; then
    echo "       funasr_env already exists, skipping..."
else
    conda create -n funasr_env python=3.10 -y
    conda activate funasr_env
    
    pip install numpy==1.24.0
    pip install torch==2.0.1 torchaudio==2.0.2
    pip install transformers
    pip install funasr modelscope
    
    # Install AudioToolAgent in this env
    cd ${BENCHMARK_ROOT}
    pip install -e . --no-deps
    
    conda deactivate
    echo "       funasr_env created ✓"
fi

echo ""
echo "=============================================="
echo "All environments set up successfully!"
echo "=============================================="
echo ""
echo "Environments created:"
echo "  - whisper_env: whisper, silero_vad, language_id, chromaprint"
echo "  - nisqa_env:   nisqa, speechmos"
echo "  - funasr_env:  funasr"
echo ""
echo "To test:"
echo "  conda activate whisper_env && python isolated_tool_exec.py whisper test_args.json"
