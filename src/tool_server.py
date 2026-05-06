#!/usr/bin/env python3
"""
FastAPI Tool Server for AudioToolAgent
Exposes all 24 audio tools via HTTP endpoints (GTA-compatible architecture).

Usage:
    python tool_server.py --port 16181 --host 0.0.0.0
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add AudioToolAgent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "AudioToolAgent"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("tool_server")

# ============================================================================
# Tool Registry
# ============================================================================

# Map tool names to their classes (lazy loaded)
TOOL_REGISTRY = {
    # Core Analysis
    "nisqa": ("audiotoolagent.tools.nisqa", "NisqaTool"),
    "deepfake_audio": ("audiotoolagent.tools.deepfake_audio", "DeepfakeAudioTool"),
    "silero_vad": ("audiotoolagent.tools.silero_vad", "SileroVADTool"),
    "language_id": ("audiotoolagent.tools.language_id", "LanguageIdentificationTool"),
    "gender_detection": ("audiotoolagent.tools.gender_detection", "GenderDetectionTool"),
    
    # Transcription
    "whisper": ("audiotoolagent.tools.whisper", "WhisperTool"),
    "funasr": ("audiotoolagent.tools.funasr_tool", "FunASRTool"),
    
    # Speaker Analysis
    "speaker_verification": ("audiotoolagent.tools.speaker_verification", "SpeakerVerificationTool"),
    "resemblyzer": ("audiotoolagent.tools.resemblyzer_tool", "ResemblyzerTool"),
    "diarizen": ("audiotoolagent.tools.diarizen", "DiarizenTool"),
    "pyannote_segmentation": ("audiotoolagent.tools.pyannote_segmentation", "PyAnnoteSegmentationTool"),
    "nemo_diarizer": ("audiotoolagent.tools.nemo_diarizer", "NemoDiarizerTool"),
    
    # Audio Separation
    "demucs": ("audiotoolagent.tools.demucs", "DemucsTool"),
    "sepformer": ("audiotoolagent.tools.sepformer_wham", "SepFormerTool"),
    "sepformer_wham": ("audiotoolagent.tools.sepformer_wham", "SepFormerWHAMTool"),
    "asteroid_separate": ("audiotoolagent.tools.asteroid_separate", "AsteroidSeparateTool"),
    
    # Audio Enhancement
    "deepfilternet": ("audiotoolagent.tools.deepfilternet", "DeepFilterNetTool"),
    "sgmse": ("audiotoolagent.tools.sgmse_tool", "SGMSETool"),
    "sb_sgmse": ("audiotoolagent.tools.sb_sgmse", "SBSGMSETool"),
    "espnet_enhance": ("audiotoolagent.tools.espnet_enhance", "ESPnetEnhanceTool"),
    
    # Quality Assessment
    "speechmos": ("audiotoolagent.tools.speechmos", "SpeechMOSTool"),
    "wav2vec2_quality": ("audiotoolagent.tools.wav2vec2_quality", "Wav2Vec2QualityTool"),
    "audioldm_eval": ("audiotoolagent.tools.audioldm_eval", "AudioLDMEvalTool"),
    "muq": ("audiotoolagent.tools.muq_tool", "MuQTool"),
    
    # Watermarking & Fingerprinting
    "audioseal": ("audiotoolagent.tools.audioseal_tool", "AudioSealTool"),
    "chromaprint": ("audiotoolagent.tools.chromaprint", "ChromaprintTool"),
    "audio_fingerprint": ("audiotoolagent.tools.chromaprint", "AudioFingerprintTool"),
    
    # Captioning & Embeddings
    "audio_caption": ("audiotoolagent.tools.audio_caption", "AudioCaptionTool"),
    "conette": ("audiotoolagent.tools.conette_tool", "CoNeTTEModelTool"),
    "clap_embed": ("audiotoolagent.tools.clap_embed", "CLAPEmbedTool"),
    
    # Advanced QA
    "r1_aqa": ("audiotoolagent.tools.r1_aqa_tool", "R1AQATool"),
}

# Audio tool categories (for GTA-style F1 metrics)
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

# ============================================================================
# Isolated Execution Configuration
# Tools that need separate conda environments due to dependency conflicts
# ============================================================================

ISOLATED_TOOLS = {
    # tool_name: conda_env_name
    "nisqa": "nisqa_env",
    "whisper": "whisper_env",
    "silero_vad": "whisper_env",  # same numpy requirements as whisper
    "language_id": "whisper_env",
    "funasr": "funasr_env",
    "chromaprint": "whisper_env",
    "speechmos": "nisqa_env",
    "deepfake_audio": "whisper_env",  # Uses transformers/torch
    "sepformer": "whisper_env",  # Uses speechbrain/torch
    "sepformer_wham": "whisper_env",
}

# Environment activation commands
CONDA_INIT = "source /home/soft/anaconda3/etc/profile.d/conda.sh"

# Loaded tool instances (lazy loaded)
_loaded_tools: Dict[str, Any] = {}


def get_tool_instance(tool_name: str) -> Any:
    """Get or create a tool instance (lazy loading)."""
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    if tool_name not in _loaded_tools:
        module_path, class_name = TOOL_REGISTRY[tool_name]
        try:
            import importlib
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)
            if tool_class is None:
                raise ImportError(f"Tool class {class_name} is None (optional dependency missing)")
            _loaded_tools[tool_name] = tool_class(device="auto")
            logger.info(f"Loaded tool: {tool_name}")
        except Exception as e:
            logger.error(f"Failed to load tool {tool_name}: {e}")
            raise
    
    return _loaded_tools[tool_name]


def execute_isolated(tool_name: str, args: Dict[str, Any], env_name: str) -> Dict[str, Any]:
    """
    Execute a tool in an isolated conda environment via subprocess.
    
    This solves numpy/dependency conflicts by running tools in separate environments.
    """
    import subprocess
    import tempfile
    
    logger.info(f"Running {tool_name} in isolated environment: {env_name}")
    
    # Write args to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(args, f)
        args_file = f.name
    
    # Output file
    output_file = args_file.replace('.json', '_output.json')
    
    try:
        # Build command to run in isolated environment
        script_path = Path(__file__).parent / "isolated_tool_exec.py"
        project_root = Path(__file__).parent.parent
        
        cmd = f"""
{CONDA_INIT}
conda activate {env_name}
export PYTHONPATH="${{PYTHONPATH}}:{project_root}/AudioToolAgent"
python {script_path} {tool_name} {args_file} -o {output_file}
"""
        
        # Execute in subprocess
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Read output
        if Path(output_file).exists():
            with open(output_file) as f:
                return json.load(f)
        else:
            return {
                "success": False,
                "error": f"Subprocess failed: {result.stderr}",
                "stdout": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Tool execution timed out (300s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        # Cleanup temp files
        try:
            os.unlink(args_file)
            if Path(output_file).exists():
                os.unlink(output_file)
        except:
            pass


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="AudioToolAgent Server",
    description="HTTP API for AudioToolAgent tools (GTA-compatible)",
    version="1.0.0"
)


class ToolRequest(BaseModel):
    """Request body for tool execution."""
    audio_path: Optional[str] = None
    audio_path_1: Optional[str] = None
    audio_path_2: Optional[str] = None
    # Generic kwargs that tools might need
    model_name: Optional[str] = None
    beam_size: Optional[int] = None
    threshold: Optional[float] = None
    top_k: Optional[int] = None
    task: Optional[str] = None
    mode: Optional[str] = None
    min_speech_ms: Optional[int] = None
    min_silence_ms: Optional[int] = None
    
    class Config:
        extra = "allow"  # Allow additional fields


class ToolResponse(BaseModel):
    """Response from tool execution."""
    success: bool
    tool_name: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "AudioToolAgent Server",
        "version": "1.0.0",
        "tools_available": len(TOOL_REGISTRY),
        "endpoints": ["/tools", "/tool/{tool_name}", "/tools/{tool_name}/meta"]
    }


@app.get("/tools")
def list_tools():
    """List all available tools."""
    tools = []
    for name, (module, class_name) in TOOL_REGISTRY.items():
        category = next(
            (cat for cat, tool_list in TOOL_CATEGORIES.items() if name in tool_list),
            "Other"
        )
        tools.append({
            "name": name,
            "class": class_name,
            "category": category,
            "loaded": name in _loaded_tools
        })
    return {"tools": tools, "total": len(tools)}


@app.get("/tools/{tool_name}/meta")
def get_tool_metadata(tool_name: str):
    """Get metadata for a specific tool."""
    if tool_name not in TOOL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
    
    # Try to load toolmeta.json if it exists
    toolmeta_path = Path(__file__).parent.parent / "data" / "audio_dataset" / "toolmeta.json"
    if toolmeta_path.exists():
        with open(toolmeta_path) as f:
            all_meta = json.load(f)
            if tool_name in all_meta:
                return all_meta[tool_name]
    
    # Fallback: return basic info
    module, class_name = TOOL_REGISTRY[tool_name]
    return {
        "name": tool_name,
        "class": class_name,
        "module": module,
        "description": f"Audio tool: {tool_name}"
    }


@app.post("/tool/{tool_name}", response_model=ToolResponse)
def execute_tool(tool_name: str, request: ToolRequest):
    """Execute a specific tool with the given parameters."""
    if tool_name not in TOOL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
    
    # Build args dict from request
    args = request.model_dump(exclude_none=True)
    
    try:
        # Check if this tool needs isolated execution (for dependency conflicts)
        if tool_name in ISOLATED_TOOLS:
            env_name = ISOLATED_TOOLS[tool_name]
            logger.info(f"Executing {tool_name} in isolated env: {env_name}")
            
            isolated_result = execute_isolated(tool_name, args, env_name)
            
            if isolated_result.get("success"):
                return ToolResponse(
                    success=True,
                    tool_name=tool_name,
                    result=isolated_result.get("result")
                )
            else:
                return ToolResponse(
                    success=False,
                    tool_name=tool_name,
                    error=isolated_result.get("error", "Isolated execution failed")
                )
        
        # Standard execution (tool runs in current process)
        tool = get_tool_instance(tool_name)
        
        # Call the tool
        logger.info(f"Executing tool {tool_name} with args: {args}")
        result = tool.call(args)
        
        # Parse result if it's a JSON string
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = {"output": result}
        
        return ToolResponse(
            success=True,
            tool_name=tool_name,
            result=result
        )
        
    except Exception as e:
        logger.error(f"Tool execution failed: {e}\n{traceback.format_exc()}")
        return ToolResponse(
            success=False,
            tool_name=tool_name,
            error=str(e)
        )


@app.post("/tool/{tool_name}/batch")
def execute_tool_batch(tool_name: str, requests: list[ToolRequest]):
    """Execute a tool on multiple inputs."""
    results = []
    for req in requests:
        result = execute_tool(tool_name, req)
        results.append(result)
    return {"results": results}


# ============================================================================
# Additional Endpoints (GTA Compatibility)
# ============================================================================

@app.get("/categories")
def get_categories():
    """Get tool categories for GTA-style F1 metrics."""
    return TOOL_CATEGORIES


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "loaded_tools": len(_loaded_tools)}


@app.post("/preload")
def preload_tools(tool_names: list[str] = None):
    """Preload tools into memory."""
    if tool_names is None:
        tool_names = list(TOOL_REGISTRY.keys())
    
    loaded = []
    failed = []
    for name in tool_names:
        try:
            get_tool_instance(name)
            loaded.append(name)
        except Exception as e:
            failed.append({"tool": name, "error": str(e)})
    
    return {"loaded": loaded, "failed": failed}


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AudioToolAgent Tool Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=16181, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()
    
    logger.info(f"Starting AudioToolAgent Server on {args.host}:{args.port}")
    logger.info(f"Available tools: {len(TOOL_REGISTRY)}")
    
    uvicorn.run(
        "tool_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info"
    )


if __name__ == "__main__":
    main()
