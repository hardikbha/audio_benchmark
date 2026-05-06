#!/usr/bin/env python3
"""
Isolated Tool Executor
Runs tools in separate conda environments via subprocess to avoid dependency conflicts.

Usage:
    python isolated_tool_exec.py <tool_name> <json_args_file>
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add AudioToolAgent to path
PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_ROOT / "AudioToolAgent"))


def execute_tool(tool_name: str, args: dict) -> dict:
    """Execute a tool and return results."""
    
    # Tool import mapping
    TOOL_IMPORTS = {
        "nisqa": ("audiotoolagent.tools.nisqa", "NisqaTool"),
        "whisper": ("audiotoolagent.tools.whisper", "WhisperTool"),
        "silero_vad": ("audiotoolagent.tools.silero_vad", "SileroVADTool"),
        "language_id": ("audiotoolagent.tools.language_id", "LanguageIdentificationTool"),
        "gender_detection": ("audiotoolagent.tools.gender_detection", "GenderDetectionTool"),
        "deepfake_audio": ("audiotoolagent.tools.deepfake_audio", "DeepfakeAudioTool"),
        "speaker_verification": ("audiotoolagent.tools.speaker_verification", "SpeakerVerificationTool"),
        "demucs": ("audiotoolagent.tools.demucs", "DemucsTool"),
        "deepfilternet": ("audiotoolagent.tools.deepfilternet", "DeepFilterNetTool"),
        "chromaprint": ("audiotoolagent.tools.chromaprint", "ChromaprintTool"),
        "audioseal": ("audiotoolagent.tools.audioseal_tool", "AudioSealTool"),
        "diarizen": ("audiotoolagent.tools.diarizen", "DiarizenTool"),
        "sepformer": ("audiotoolagent.tools.sepformer_wham", "SepFormerTool"),
        "sepformer_wham": ("audiotoolagent.tools.sepformer_wham", "SepFormerWHAMTool"),
        "sgmse": ("audiotoolagent.tools.sgmse_tool", "SGMSETool"),
        "speechmos": ("audiotoolagent.tools.speechmos", "SpeechMOSTool"),
        "funasr": ("audiotoolagent.tools.funasr_tool", "FunASRTool"),
        "resemblyzer": ("audiotoolagent.tools.resemblyzer_tool", "ResemblyzerTool"),
        "clap_embed": ("audiotoolagent.tools.clap_embed", "CLAPEmbedTool"),
        "muq": ("audiotoolagent.tools.muq_tool", "MuQTool"),
    }
    
    if tool_name not in TOOL_IMPORTS:
        return {"error": f"Unknown tool: {tool_name}"}
    
    try:
        import importlib
        module_path, class_name = TOOL_IMPORTS[tool_name]
        module = importlib.import_module(module_path)
        tool_class = getattr(module, class_name)
        
        # Initialize and call tool
        tool = tool_class(device="auto")
        result = tool.call(args)
        
        # Parse result if JSON string
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = {"output": result}
        
        return {"success": True, "result": result}
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(description="Execute tool in isolation")
    parser.add_argument("tool_name", help="Name of the tool to execute")
    parser.add_argument("args_file", help="Path to JSON file with tool arguments")
    parser.add_argument("--output", "-o", help="Output file for results (default: stdout)")
    args = parser.parse_args()
    
    # Load args from file
    with open(args.args_file) as f:
        tool_args = json.load(f)
    
    # Execute tool
    result = execute_tool(args.tool_name, tool_args)
    
    # Output result
    result_json = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(result_json)
    else:
        print(result_json)


if __name__ == "__main__":
    main()
