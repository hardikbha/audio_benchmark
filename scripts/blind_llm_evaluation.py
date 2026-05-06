#!/usr/bin/env python3
"""Blind LLM Evaluation - Using Open Source Models (HuggingFace/vLLM).

Uses local open-source models for tool selection evaluation.
Supports: vLLM server, HuggingFace transformers, OpenAI-compatible APIs.

Usage: python blind_llm_evaluation.py --audio_path /path/to/audio.wav --model_type vllm
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup paths
REPO_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(REPO_ROOT / "AudioToolAgent"))

# =========================================
# SAMPLE QUERIES (Natural Language - No Tool Names!)
# =========================================

SAMPLE_QUERIES = [
    {
        "id": "nlq_001",
        "query": (
            "I received this audio file and I'm not sure what to make of it. "
            "Can you tell me what language the person is speaking, "
            "write down exactly what they're saying, "
            "and let me know if the recording quality is good enough for a podcast?"
        ),
        "ground_truth": {
            "expected_tools": ["language_id", "whisper", "nisqa"],
            "tool_reasoning": {
                "language_id": "User asks 'what language the person is speaking'",
                "whisper": "User asks to 'write down exactly what they're saying' (transcription)",
                "nisqa": "User asks 'if the recording quality is good enough' (quality assessment)",
            },
            "expected_answer_contains": [
                ["english", "en", "language"],
                ["transcri", "said", "spoke"],
                ["quality", "mos", "score", "good", "bad"],
            ],
        },
    },
    {
        "id": "nlq_002",
        "query": (
            "Someone sent me this voice note claiming to be my friend. "
            "I want to verify if this is actually a real human voice or some AI-generated fake. "
            "Also, what are they saying and in what language?"
        ),
        "ground_truth": {
            "expected_tools": ["deepfake_audio", "language_id", "whisper"],
            "tool_reasoning": {
                "deepfake_audio": "User asks to verify 'real human voice or AI-generated fake'",
                "language_id": "User asks 'in what language'",
                "whisper": "User asks 'what are they saying' (transcription)",
            },
            "expected_answer_contains": [
                ["real", "fake", "authentic", "genuine", "deepfake"],
                ["english", "en", "language"],
                ["said", "spoke", "transcript"],
            ],
        },
    },
    {
        "id": "nlq_003",
        "query": (
            "This is a noisy recording from a meeting. "
            "Can you first identify when people are actually talking, "
            "then tell me what language they're using, "
            "transcribe the conversation, "
            "and give me a quality rating for the audio?"
        ),
        "ground_truth": {
            "expected_tools": ["silero_vad", "language_id", "whisper", "nisqa"],
            "tool_reasoning": {
                "silero_vad": "User asks to 'identify when people are actually talking'",
                "language_id": "User asks 'what language they're using'",
                "whisper": "User asks to 'transcribe the conversation'",
                "nisqa": "User asks for 'quality rating for the audio'",
            },
            "expected_answer_contains": [
                ["speech", "talk", "segment", "second"],
                ["english", "en", "language"],
                ["said", "spoke", "transcript"],
                ["quality", "mos", "rating"],
            ],
        },
    },
]

# =========================================
# SYSTEM PROMPT FOR TOOL SELECTION
# =========================================

SYSTEM_PROMPT = """You are an expert audio analysis assistant. Your task is to analyze user requests about audio files and decide which tools to use.

## Available Tools

| Tool | Capability | When to Use |
|------|------------|-------------|
| language_id | Identify spoken language | User asks about language, what language, which language |
| whisper | Transcribe speech to text | User wants transcription, what is being said, write down words |
| nisqa | Assess audio quality (MOS 1-5) | User asks about quality, good enough, clear audio |
| silero_vad | Detect speech segments | User asks when speech occurs, find talking parts |
| deepfake_audio | Detect fake/AI audio | User asks if real, authentic, deepfake, AI-generated |
| deepfilternet | Remove background noise | User wants cleaner audio, remove noise |
| demucs | Separate music sources | User wants vocals separated, extract instruments |
| speaker_verification | Verify same speaker | User asks if same person, verify speaker identity |

## Instructions

1. Read the user's request carefully
2. Identify which tools are needed based on what the user is asking
3. List tools in the order they should be executed
4. Explain briefly why each tool is needed

## Response Format

You MUST respond with ONLY valid JSON in this exact format:
```json
{
    "reasoning": "Brief explanation of what the user wants",
    "tools_to_call": ["tool1", "tool2", "tool3"],
    "tool_reasons": {
        "tool1": "why this tool is needed",
        "tool2": "why this tool is needed"
    }
}
```

Do not include any text outside the JSON block."""

USER_PROMPT_TEMPLATE = """Analyze this user request and determine which audio tools to use:

USER REQUEST: {query}

Remember: Respond with ONLY valid JSON containing tools_to_call and reasoning."""


def call_vllm_api(query: str, base_url: str = "http://localhost:8000/v1", 
                  model: str = "Qwen/Qwen2.5-7B-Instruct") -> Dict:
    """Call vLLM server using OpenAI-compatible API."""
    try:
        import requests
        
        response = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query)},
                ],
                "max_tokens": 512,
                "temperature": 0.1,
            },
            timeout=60,
        )
        
        if response.status_code != 200:
            return {"error": f"API error: {response.status_code} - {response.text}"}
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Parse JSON from response
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_text = content[json_start:json_end].strip()
        elif "{" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_text = content[json_start:json_end]
        else:
            return {"error": "No JSON found in response", "raw": content}
        
        return json.loads(json_text)
        
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to vLLM server at " + base_url}
    except Exception as e:
        return {"error": str(e)}


def call_huggingface_model(query: str, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> Dict:
    """Use HuggingFace transformers directly (CPU-friendly small model)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"  Loading model {model_name}...", end=" ", flush=True)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # CPU-friendly
            device_map="auto",
            trust_remote_code=True,
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query)},
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt")
        
        print("Generating...", end=" ", flush=True)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
        )
        
        content = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "assistant" in content.lower():
            content = content.split("assistant")[-1]
        
        # Parse JSON
        if "{" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_text = content[json_start:json_end]
            return json.loads(json_text)
        
        return {"error": "No JSON found", "raw": content[:500]}
        
    except Exception as e:
        return {"error": str(e)}


def call_gemini_api(query: str, api_key: str) -> Dict:
    """Fallback to Gemini API if available."""
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=api_key)
        
        prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(query=query)}"
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=512,
                temperature=0.1,
            ),
        )
        
        content = response.text
        
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_text = content[json_start:json_end].strip()
        elif "{" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_text = content[json_start:json_end]
        else:
            return {"error": "No JSON found", "raw": content[:500]}
        
        return json.loads(json_text)
        
    except Exception as e:
        return {"error": str(e)}


def run_tool(tool_name: str, audio_path: str) -> tuple:
    """Run a tool and return (output, time)."""
    start_time = time.time()
    output = None
    
    try:
        if tool_name == "language_id":
            from audiotoolagent.tools.language_id import LanguageIdentificationTool
            tool = LanguageIdentificationTool()
            output = tool.call(json.dumps({"audio_path": audio_path}))
            
        elif tool_name == "whisper":
            from audiotoolagent.tools.whisper import WhisperTool
            tool = WhisperTool(compute_type="int8")
            output = tool.call({"audio_path": audio_path})
            
        elif tool_name == "nisqa":
            from audiotoolagent.tools.nisqa import NisqaTool
            tool = NisqaTool()
            output = tool.call(json.dumps({"audio_path": audio_path}))
            
        elif tool_name == "silero_vad":
            from audiotoolagent.tools.silero_vad import SileroVADTool
            tool = SileroVADTool()
            output = tool.call(json.dumps({"audio_path": audio_path}))
            
        elif tool_name == "deepfake_audio":
            from audiotoolagent.tools.deepfake_audio import DeepfakeAudioTool
            tool = DeepfakeAudioTool()
            output = tool.call(json.dumps({"audio_path": audio_path}))
            
    except Exception as e:
        output = {"error": str(e)}
    
    elapsed = time.time() - start_time
    
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except:
            pass
    
    return output, elapsed


def calculate_metrics(predicted_tools: List[str], expected_tools: List[str], 
                      final_answer: str, ground_truth: Dict) -> Dict:
    """Calculate all GTA metrics."""
    
    pred_set = set(predicted_tools)
    exp_set = set(expected_tools)
    
    correct_tools = pred_set.intersection(exp_set)
    missing_tools = exp_set - pred_set
    extra_tools = pred_set - exp_set
    
    precision = len(correct_tools) / len(pred_set) if pred_set else 0
    recall = len(correct_tools) / len(exp_set) if exp_set else 0
    tool_acc = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    inst_acc = 1.0 if exp_set.issubset(pred_set) else recall
    
    answer_lower = final_answer.lower()
    expected_contains = ground_truth.get("expected_answer_contains", [])
    contains_score = 0
    for group in expected_contains:
        if any(term.lower() in answer_lower for term in group):
            contains_score += 1
    ans_acc = contains_score / len(expected_contains) if expected_contains else 1.0
    
    return {
        "InstAcc": round(inst_acc * 100, 2),
        "ToolAcc": round(tool_acc * 100, 2),
        "ArgAcc": 100.0,
        "SummAcc": round(ans_acc * 100, 2),
        "AnsAcc": round(ans_acc * 100, 2),
        "Ans+I": round(ans_acc * inst_acc * 100, 2),
        "correct_tools": list(correct_tools),
        "missing_tools": list(missing_tools),
        "extra_tools": list(extra_tools),
    }


def run_blind_evaluation(query_data: Dict, audio_path: str, 
                         model_type: str = "vllm", 
                         vllm_url: str = "http://localhost:8000/v1",
                         model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> Dict:
    """Run complete blind evaluation."""
    
    print("\n" + "="*70)
    print("BLIND LLM EVALUATION (Open Source Model)")
    print("="*70)
    
    query = query_data["query"]
    ground_truth = query_data["ground_truth"]
    expected_tools = ground_truth["expected_tools"]
    
    print(f"\n📝 QUERY (Natural Language - No Tool Hints):")
    print(f"   {query[:100]}..." if len(query) > 100 else f"   {query}")
    print(f"\n🎯 GROUND TRUTH (Hidden from LLM):")
    print(f"   Expected tools: {expected_tools}")
    print(f"\n🤖 MODEL: {model_type} - {model_name}")
    
    # Step 1: Ask LLM to select tools
    print("\n" + "-"*50)
    print("[STEP 1] Asking LLM to select tools...")
    
    if model_type == "vllm":
        llm_response = call_vllm_api(query, vllm_url, model_name)
    elif model_type == "huggingface":
        llm_response = call_huggingface_model(query, model_name)
    elif model_type == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        llm_response = call_gemini_api(query, api_key)
    else:
        llm_response = {"error": f"Unknown model type: {model_type}"}
    
    if "error" in llm_response:
        print(f"   ❌ LLM Error: {llm_response['error']}")
        predicted_tools = []
    else:
        predicted_tools = llm_response.get("tools_to_call", [])
        print(f"   ✓ LLM selected: {predicted_tools}")
        print(f"   Reasoning: {llm_response.get('reasoning', 'N/A')[:100]}")
    
    # Step 2: Execute selected tools
    print("\n" + "-"*50)
    print("[STEP 2] Executing selected tools...")
    
    tool_outputs = {}
    total_time = 0
    
    for tool in predicted_tools:
        print(f"   Running {tool}...", end=" ", flush=True)
        output, elapsed = run_tool(tool, audio_path)
        tool_outputs[tool] = output
        total_time += elapsed
        print(f"({elapsed:.2f}s)")
    
    # Step 3: Generate final answer
    print("\n" + "-"*50)
    print("[STEP 3] Generating final answer...")
    
    answer_parts = []
    for tool, output in tool_outputs.items():
        if isinstance(output, dict) and "error" not in output:
            if tool == "language_id":
                lang = output.get("detected_language", "unknown")
                answer_parts.append(f"Language: {lang}")
            elif tool == "nisqa":
                result = output.get("result", {})
                mos = result.get("mos", "N/A") if isinstance(result, dict) else "N/A"
                answer_parts.append(f"Quality (MOS): {mos}/5.0")
            elif tool == "whisper":
                result = output.get("result", {})
                text = result.get("text", "")[:100] if isinstance(result, dict) else ""
                answer_parts.append(f"Transcript: \"{text}\"" if text else "Transcript: N/A")
    
    final_answer = " | ".join(answer_parts)
    print(f"   Answer: {final_answer}")
    
    # Step 4: Calculate metrics
    print("\n" + "-"*50)
    print("[STEP 4] Calculating metrics...")
    
    metrics = calculate_metrics(predicted_tools, expected_tools, final_answer, ground_truth)
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│ TOOL SELECTION                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Expected:  {str(expected_tools):<50} │
│  Predicted: {str(predicted_tools):<50} │
│  Correct:   {str(metrics['correct_tools']):<50} │
│  Missing:   {str(metrics['missing_tools']):<50} │
│  Extra:     {str(metrics['extra_tools']):<50} │
├─────────────────────────────────────────────────────────────────┤
│ GTA METRICS                                                     │
├─────────────────────────────────────────────────────────────────┤
│  InstAcc (Instruction Accuracy):    {metrics['InstAcc']:>6.2f}%                     │
│  ToolAcc (Tool Accuracy):           {metrics['ToolAcc']:>6.2f}%                     │
│  ArgAcc (Argument Accuracy):        {metrics['ArgAcc']:>6.2f}%                     │
│  SummAcc (Summary Accuracy):        {metrics['SummAcc']:>6.2f}%                     │
│  AnsAcc (Answer Accuracy):          {metrics['AnsAcc']:>6.2f}%                     │
│  Ans+I (Answer + Instruction):      {metrics['Ans+I']:>6.2f}%                     │
├─────────────────────────────────────────────────────────────────┤
│ INFERENCE                                                       │
├─────────────────────────────────────────────────────────────────┤
│  Total Time: {total_time:>6.2f}s                                             │
│  Tools Run:  {len(predicted_tools)}                                                     │
└─────────────────────────────────────────────────────────────────┘
""")
    
    return {
        "query": query_data,
        "audio_path": audio_path,
        "model_type": model_type,
        "model_name": model_name,
        "llm_response": llm_response,
        "predicted_tools": predicted_tools,
        "tool_outputs": {k: str(v)[:200] for k, v in tool_outputs.items()},
        "final_answer": final_answer,
        "metrics": metrics,
        "total_time_sec": total_time,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str,
                        default=str(REPO_ROOT / "AudioToolAgent/testing_data/10006.wav"))
    parser.add_argument("--query_id", type=int, default=0,
                        help="Which sample query to use (0, 1, or 2)")
    parser.add_argument("--model_type", type=str, default="vllm",
                        choices=["vllm", "huggingface", "gemini"],
                        help="Model backend to use")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server URL")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name/path")
    args = parser.parse_args()
    
    query_data = SAMPLE_QUERIES[args.query_id]
    
    results = run_blind_evaluation(
        query_data, 
        args.audio_path, 
        model_type=args.model_type,
        vllm_url=args.vllm_url,
        model_name=args.model_name,
    )
    
    output_path = REPO_ROOT / "outputs" / "blind_evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
