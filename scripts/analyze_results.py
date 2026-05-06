
import json
import sys

try:
    with open("outputs/results_24_tools.json", "r") as f:
        results = json.load(f)

    print(f"Total Tasks: {len(results)}")
    successes = [r for r in results.values() if r.get("success")]
    failures = [r for r in results.values() if not r.get("success")]
    
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")
    
    print("\n--- Failed Tasks ---")
    for r in failures:
        print(f"ID: {r.get('task_id')}")
        print(f"Error: {r.get('error') or r.get('trace', {}).get('error')}")
        print("-" * 20)

    print("\n--- Success Examples ---")
    for i, r in enumerate(successes[:3]):
        print(f"ID: {r.get('task_id')}")
        print(f"Tools Called: {r.get('tools_called')}")
        print(f"Predicted: {r.get('predicted_answer')}")
        print("-" * 20)
        
except Exception as e:
    print(f"Error analyzing results: {e}")
