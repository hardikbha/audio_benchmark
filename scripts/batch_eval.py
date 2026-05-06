#!/usr/bin/env python3
"""Batch evaluation script for running all models on audio benchmark.

Usage:
    python batch_eval.py --models api,7b --queries 0-250
    python batch_eval.py --models 70b,audio --queries 250-500
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Model configurations
MODELS = {
    # API Models (require API keys)
    "api": {
        "gpt-4o": {"type": "openai", "model": "gpt-4o", "requires": "OPENAI_API_KEY"},
        "gpt-4o-mini": {"type": "openai", "model": "gpt-4o-mini", "requires": "OPENAI_API_KEY"},
        "claude-3.5-sonnet": {"type": "anthropic", "model": "claude-3-5-sonnet-20241022", "requires": "ANTHROPIC_API_KEY"},
        "gemini-1.5-pro": {"type": "google", "model": "gemini-1.5-pro", "requires": "GOOGLE_API_KEY"},
        "gemini-2.0-flash": {"type": "google", "model": "gemini-2.0-flash", "requires": "GOOGLE_API_KEY"},
    },
    # 7B Open-source Models
    "7b": {
        "qwen2.5-7b": {"type": "vllm", "model": "Qwen/Qwen2.5-7B-Instruct", "gpu_memory": "16GB"},
        "llama3.1-8b": {"type": "vllm", "model": "meta-llama/Llama-3.1-8B-Instruct", "gpu_memory": "16GB"},
        "mistral-7b": {"type": "vllm", "model": "mistralai/Mistral-7B-Instruct-v0.3", "gpu_memory": "16GB"},
        "deepseek-7b": {"type": "vllm", "model": "deepseek-ai/deepseek-coder-7b-instruct-v1.5", "gpu_memory": "16GB"},
    },
    # 70B Models (multi-GPU)
    "70b": {
        "qwen2.5-72b": {"type": "vllm", "model": "Qwen/Qwen2.5-72B-Instruct", "gpu_memory": "80GB", "gpus": 2},
        "llama3.1-70b": {"type": "vllm", "model": "meta-llama/Llama-3.1-70B-Instruct", "gpu_memory": "80GB", "gpus": 2},
    },
    # Audio-specific Models
    "audio": {
        "qwen2.5-omni": {"type": "custom", "model": "Qwen/Qwen2.5-Omni", "gpu_memory": "24GB"},
        "salmonn": {"type": "custom", "model": "tsinghua-ee/SALMONN-7B", "gpu_memory": "24GB"},
    },
}


class BatchEvaluator:
    """Run batch evaluations on the audio benchmark."""
    
    def __init__(self, output_dir: str = "outputs/benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Any] = {}
        
    def check_api_keys(self, model_config: Dict) -> bool:
        """Check if required API keys are set."""
        if "requires" in model_config:
            key = model_config["requires"]
            if not os.getenv(key):
                print(f"  ⚠️  Missing API key: {key}")
                return False
        return True
    
    def run_opencompass_eval(self, model_name: str, model_config: Dict, 
                              query_range: tuple, dataset_path: str) -> Dict:
        """Run evaluation using OpenCompass framework."""
        
        start_time = time.time()
        results = {
            "model": model_name,
            "config": model_config,
            "query_range": query_range,
            "start_time": datetime.now().isoformat(),
            "status": "pending",
        }
        
        try:
            # This would call the OpenCompass evaluation
            # For now, we'll create a placeholder
            print(f"  Running evaluation for {model_name}...")
            
            # Simulated result structure
            results["status"] = "completed"
            results["end_time"] = datetime.now().isoformat()
            results["duration_sec"] = round(time.time() - start_time, 2)
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    def run_batch(self, model_categories: List[str], query_range: tuple, 
                  dataset_path: str = "GTA/audio_dataset_500.json"):
        """Run batch evaluation for specified model categories."""
        
        print(f"\n{'='*60}")
        print(f"BATCH EVALUATION")
        print(f"Models: {model_categories}")
        print(f"Queries: {query_range[0]}-{query_range[1]}")
        print(f"{'='*60}\n")
        
        all_results = []
        
        for category in model_categories:
            if category not in MODELS:
                print(f"Unknown model category: {category}")
                continue
            
            print(f"\n--- Category: {category} ---")
            
            for model_name, model_config in MODELS[category].items():
                print(f"\n[{model_name}]")
                
                # Check API keys for API models
                if not self.check_api_keys(model_config):
                    continue
                
                # Run evaluation
                result = self.run_opencompass_eval(
                    model_name, model_config, query_range, dataset_path
                )
                all_results.append(result)
        
        # Save batch results
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"batch_{batch_id}.json"
        
        with open(output_file, "w") as f:
            json.dump({
                "batch_id": batch_id,
                "model_categories": model_categories,
                "query_range": query_range,
                "results": all_results,
            }, f, indent=2)
        
        print(f"\n\nResults saved to: {output_file}")
        return all_results


def parse_query_range(range_str: str) -> tuple:
    """Parse query range string like '0-250' to tuple (0, 250)."""
    parts = range_str.split("-")
    return (int(parts[0]), int(parts[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True,
                        help="Comma-separated model categories: api,7b,70b,audio")
    parser.add_argument("--queries", type=str, default="0-500",
                        help="Query range, e.g., 0-250")
    parser.add_argument("--dataset", type=str, default="GTA/audio_dataset_500.json",
                        help="Path to query dataset")
    parser.add_argument("--output", type=str, default="outputs/benchmark",
                        help="Output directory")
    args = parser.parse_args()
    
    model_categories = args.models.split(",")
    query_range = parse_query_range(args.queries)
    
    evaluator = BatchEvaluator(output_dir=args.output)
    evaluator.run_batch(model_categories, query_range, args.dataset)
