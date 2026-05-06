
import json
import os

REQUIRED_TOOLS = [
    "nisqa", "speechmos", "deepfake_audio", "desta25", "audioseal", 
    "silero_vad", "language_id", "gender_detection", "whisper", "funasr", 
    "audio_caption", "sb_sgmse", "deepfilternet", "espnet_enhance", "demucs", 
    "sepformer_wham", "nemo_diarizer", "diarizen", "resemblyzer", 
    "speaker_verification", "clap_embed", "muq", "r1_aqa", "chromaprint"
]

PATH = "data/audio_dataset/toolmeta.json"
BACKUP = "data/audio_dataset/toolmeta.json.bak"

def filter_tools():
    with open(PATH, "r") as f:
        data = json.load(f)
    
    # Save backup
    if not os.path.exists(BACKUP):
        with open(BACKUP, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Backup saved to {BACKUP}")
    
    new_data = {}
    missing = []
    
    for tool in REQUIRED_TOOLS:
        if tool in data:
            new_data[tool] = data[tool]
        else:
            missing.append(tool)
            
    print(f"Kept {len(new_data)} tools.")
    if missing:
        print(f"Warning: The following requested tools were NOT found in toolmeta.json: {missing}")
        
    # Write back
    with open(PATH, "w") as f:
        json.dump(new_data, f, indent=4)
    print(f"Updated {PATH}")

if __name__ == "__main__":
    filter_tools()
