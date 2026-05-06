import json
import random
import datetime

# --- Configuration ---
NUM_TASKS = 500
OUTPUT_FILE = "benchmark_dataset.jsonl"
TOOL_REGISTRY_PATH = "docs/tool_registry.json"

# --- Templates for Categories ---
CATEGORIES = {
    "C1_Deepfake": {
        "queries": ["Is this voice real?", "Check for deepfake traces.", "Verify audio authenticity."],
        "tools": ["deepfake_audio", "audioseal"],
        "answer": "Bona fide or Synthetic"
    },
    "C2_Attribution": {
        "queries": ["Who is speaking?", "Count speakers.", "Verify identity."],
        "tools": ["diarizen", "speaker_verification"],
        "answer": "Speaker labels or Match/Mismatch"
    },
    "C3_Content": {
        "queries": ["Transcribe the audio.", "What is the language?", "Write the text."],
        "tools": ["whisper", "language_id"],
        "answer": "Transcript text"
    },
    "C4_Forensics": {
        "queries": ["Check for splicing.", "Find manipulation artifacts.", "Is this a technical duplicate?"],
        "tools": ["chromaprint", "metadata_parser", "pitch_tracker"],
        "answer": "Diagnostic report"
    },
    "C5_Separation": {
        "queries": ["Separate the vocals.", "Isolate the speakers.", "Remove background music."],
        "tools": ["demucs", "sepformer_wham"],
        "answer": "Paths to isolated files"
    },
    "C6_Quality": {
        "queries": ["Rate the quality.", "Check MOS score.", "Evaluate noisiness."],
        "tools": ["nisqa", "torchaudio_squim"],
        "answer": "Numerical quality scores"
    }
}

# --- Complex Combinations ---
# {query_format, tools, answers, logic_template}
COMPLEX_SCENARIOS = [
    {
        "query": "Separate the voices in this clip, check if either is a deepfake, and transcribe the real one.",
        "tools": ["sepformer_wham", "deepfake_audio", "whisper"],
        "logic": ["Separate speakers", "Verify Deepfake (Spk1)", "Verify Deepfake (Spk2)", "Transcribe Speech"],
        "difficulty": "Hard"
    },
    {
        "query": "Detect the language, transcribe it, and then check if the file has any watermarks.",
        "tools": ["language_id", "whisper", "audioseal"],
        "logic": ["Detect Language", "Apply ASR", "Scan Watermark"],
        "difficulty": "Medium"
    },
    {
        "query": "I am being harassed by these voices. Split the speakers, identify the main voice against my sample, and check if it is fake.",
        "tools": ["sepformer_wham", "speaker_verification", "deepfake_audio", "nisqa"],
        "logic": ["Voice Separation", "Speaker Matching", "Synthetic Check", "Quality Audit"],
        "difficulty": "Hard"
    }
]

def format_trace(logic_steps, tools):
    trace = []
    for i, step in enumerate(logic_steps):
        tool = tools[i % len(tools)]
        trace.append({"Thought": f"Step {i+1}: {step} using {tool}."})
        trace.append({"Action": {"tool_name": tool, "args": {"audio_path": "input.wav"}}})
    return trace

def generate_task(task_id):
    rand = random.random()
    
    if rand < 0.4: # 40% Complex
        scenario = random.choice(COMPLEX_SCENARIOS)
        category = "C7_Complex"
        difficulty = scenario["difficulty"]
        tools = scenario["tools"]
        logic = scenario.get("logic", ["Process audio"])
        user_query = scenario["query"]
        min_tools = len(tools)
    else:
        cat_id = random.choice(list(CATEGORIES.keys()))
        data = CATEGORIES[cat_id]
        category = cat_id
        difficulty = "Easy"
        user_query = random.choice(data["queries"])
        tools = [random.choice(data["tools"])]
        logic = [f"Analyze {cat_id}"]
        min_tools = 1

    trace = format_trace(logic, tools)
    
    return {
        "id": f"TASK_{task_id:04d}",
        "category": category,
        "difficulty": difficulty,
        "audio_asset_id": "asset_" + str(random.randint(1000, 9999)),
        "user_query": user_query,
        "expected_answer_format": "JSON object with forensic findings",
        "gold_answer": "Final analysis based on tool outputs.",
        "required_tools_min_count": min_tools,
        "reference_tool_trace": trace,
        "grading_rubric": "Score 1.0 if all tools called in order with correct args."
    }

# --- Generate ---
print(f"Generating {NUM_TASKS} tasks...")
with open(OUTPUT_FILE, "w") as f:
    for i in range(NUM_TASKS):
        task = generate_task(i)
        f.write(json.dumps(task) + "\n")

print(f"Done. Wrote to {OUTPUT_FILE}")
