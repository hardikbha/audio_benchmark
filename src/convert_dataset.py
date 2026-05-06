#!/usr/bin/env python3
"""
Dataset Converter: Convert existing JSONL to GTA-compatible JSON format.

Usage:
    python convert_dataset.py \
        --input benchmark_dataset.jsonl \
        --output data/audio_dataset/dataset.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def convert_jsonl_to_gta(input_path: str, output_path: str):
    """
    Convert benchmark_dataset.jsonl to GTA-compatible JSON format.
    
    JSONL format (current):
    {
        "id": "TASK_0001",
        "category": "C1_QualityAnalysis",
        "difficulty": "Easy",
        "audio_asset_id": "asset_0001",
        "user_query": "What is the quality of this audio?",
        "expected_answer_format": "...",
        "gold_answer": "The audio has MOS 4.2",
        "required_tools_min_count": 2,
        "reference_tool_trace": [
            {"Thought": "...", "Action": {"tool_name": "nisqa", "args": {...}}}
        ],
        "grading_rubric": "..."
    }
    
    GTA format (target):
    {
        "TASK_0001": {
            "question": "What is the quality of this audio?",
            "file": ["asset_0001.wav"],
            "steps": [
                {"tool": "nisqa", "args": {"audio_path": "..."}, "output": "..."}
            ],
            "answer": "The audio has MOS 4.2"
        }
    }
    """
    dataset = {}
    
    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} - JSON error: {e}")
                continue
            
            task_id = item.get("id", f"TASK_{line_num:04d}")
            
            # Convert audio asset to file path
            audio_asset = item.get("audio_asset_id", "")
            audio_files = [f"{audio_asset}.wav"] if audio_asset else []
            
            # Convert tool trace
            steps = []
            tool_trace = item.get("reference_tool_trace", [])
            for step in tool_trace:
                action = step.get("Action", {})
                if isinstance(action, dict):
                    tool_name = action.get("tool_name", "")
                    tool_args = action.get("args", {})
                else:
                    tool_name = ""
                    tool_args = {}
                
                steps.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "output": "[placeholder]",  # Will be filled during execution
                    "thought": step.get("Thought", "")
                })
            
            # Build GTA-format entry
            dataset[task_id] = {
                "question": item.get("user_query", ""),
                "file": audio_files,
                "steps": steps,
                "answer": item.get("gold_answer", ""),
                # Preserve additional metadata
                "category": item.get("category", ""),
                "difficulty": item.get("difficulty", ""),
                "required_tools_min_count": item.get("required_tools_min_count", 1),
                "grading_rubric": item.get("grading_rubric", "")
            }
    
    # Save GTA format
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Converted {len(dataset)} tasks to GTA format")
    print(f"Output: {output_path}")
    
    # Print category distribution
    categories = {}
    for task_id, task in dataset.items():
        cat = task.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to GTA format")
    parser.add_argument("--input", default="benchmark_dataset.jsonl",
                       help="Input JSONL file path")
    parser.add_argument("--output", default="data/audio_dataset/dataset.json",
                       help="Output JSON file path")
    args = parser.parse_args()
    
    convert_jsonl_to_gta(args.input, args.output)


if __name__ == "__main__":
    main()
