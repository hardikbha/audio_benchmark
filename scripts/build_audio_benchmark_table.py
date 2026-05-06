#!/usr/bin/env python3
"""Build benchmark tables with accuracy, orchestration, and token/latency metrics.

Supports run_inference-style results JSON files (task_id -> result dict).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate import evaluate_end_to_end, check_answer_match  # noqa: E402


DEFAULT_PRICING = {
    # USD per 1M tokens
    "default": {"input": 0.0, "output": 0.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
}


def parse_run_spec(spec: str) -> Tuple[str, Path]:
    """Parse run spec in form model_tag=path/to/results.json."""
    if "=" not in spec:
        path = Path(spec)
        return path.stem, path
    model_tag, raw_path = spec.split("=", 1)
    return model_tag.strip(), Path(raw_path.strip())


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def sanitize_tag(tag: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", tag)


def load_pricing(path: str | None) -> Dict[str, Dict[str, float]]:
    if not path:
        return DEFAULT_PRICING
    with open(path) as f:
        user_pricing = json.load(f)
    merged = dict(DEFAULT_PRICING)
    merged.update(user_pricing)
    return merged


def estimate_cost_usd(
    pricing: Dict[str, Dict[str, float]],
    model_tag: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    price = pricing.get(model_tag, pricing.get("default", {"input": 0.0, "output": 0.0}))
    input_cost = (prompt_tokens / 1_000_000.0) * float(price.get("input", 0.0))
    output_cost = (completion_tokens / 1_000_000.0) * float(price.get("output", 0.0))
    return round(input_cost + output_cost, 6)


def flatten_results(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Return (results_dict, source_kind)."""
    # run_inference output: {task_id: {...}, ...}
    if raw and all(isinstance(v, dict) for v in raw.values()):
        return raw, "run_inference"

    # run_batch_inference evaluation summary output
    if isinstance(raw, dict) and isinstance(raw.get("per_query_metrics"), list):
        converted: Dict[str, Any] = {}
        for item in raw["per_query_metrics"]:
            task_id = str(item.get("id", len(converted)))
            converted[task_id] = {
                "task_id": task_id,
                "question": "",
                "trace": {"steps": []},
                "predicted_answer": "",
                "ground_truth": None,
                "tools_called": item.get("tools_called", []),
                "expected_tools": item.get("expected_tools", []),
                "success": bool(item.get("success", False)),
                "llm_calls": 0,
                "llm_prompt_tokens": 0,
                "llm_completion_tokens": 0,
                "llm_total_tokens": 0,
                "llm_time_sec": 0.0,
                "tool_time_sec": 0.0,
                "total_time_sec": float(item.get("time", 0.0) or 0.0),
            }
        return converted, "per_query_metrics"

    raise ValueError("Unsupported results format. Expected run_inference dict output.")


def build_per_query_rows(
    model_tag: str,
    results: Dict[str, Any],
    valid_tools: set[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task_id, item in results.items():
        tools_called = [str(t) for t in item.get("tools_called", []) if t]
        expected_tools = [str(t) for t in item.get("expected_tools", []) if t]
        valid_called = [t for t in tools_called if t in valid_tools]
        invalid_called = [t for t in tools_called if t not in valid_tools]
        answer_match = check_answer_match(item.get("predicted_answer"), item.get("ground_truth"))
        step_count = len(item.get("trace", {}).get("steps", []))

        prompt_tokens = int(item.get("llm_prompt_tokens", 0) or 0)
        completion_tokens = int(item.get("llm_completion_tokens", 0) or 0)
        total_tokens = int(item.get("llm_total_tokens", prompt_tokens + completion_tokens) or 0)
        llm_calls = int(item.get("llm_calls", 0) or 0)
        llm_time_sec = float(item.get("llm_time_sec", 0.0) or 0.0)
        tool_time_sec = float(item.get("tool_time_sec", 0.0) or 0.0)
        total_time_sec = float(item.get("total_time_sec", 0.0) or 0.0)

        rows.append(
            {
                "model_tag": model_tag,
                "task_id": task_id,
                "success": bool(item.get("success", False)),
                "answer_match": bool(answer_match),
                "question": item.get("question", ""),
                "audio_files": "|".join(item.get("audio_files", [])),
                "tools_called": "|".join(tools_called),
                "expected_tools": "|".join(expected_tools),
                "valid_tools_called": "|".join(valid_called),
                "invalid_tools_called": "|".join(invalid_called),
                "tool_calls_count": len(tools_called),
                "valid_tool_calls_count": len(valid_called),
                "invalid_tool_calls_count": len(invalid_called),
                "has_valid_tool_call": bool(valid_called),
                "step_count": step_count,
                "llm_calls": llm_calls,
                "llm_prompt_tokens": prompt_tokens,
                "llm_completion_tokens": completion_tokens,
                "llm_total_tokens": total_tokens,
                "llm_time_sec": round(llm_time_sec, 4),
                "tool_time_sec": round(tool_time_sec, 4),
                "total_time_sec": round(total_time_sec, 4),
            }
        )
    return rows


def aggregate_summary(
    model_tag: str,
    results: Dict[str, Any],
    rows: List[Dict[str, Any]],
    pricing: Dict[str, Dict[str, float]],
    source_kind: str = "run_inference",
    raw_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "Model": model_tag,
            "Queries": 0,
        }

    if source_kind == "run_inference":
        eval_metrics = evaluate_end_to_end(results).to_dict()
    else:
        eval_metrics = {
            "answer_accuracy": 0.0,
            "perception_f1": 0.0,
            "analysis_f1": 0.0,
            "transformation_f1": 0.0,
            "detection_f1": 0.0,
        }

    valid_tool_call_rate = 100.0 * safe_div(sum(1 for r in rows if r["has_valid_tool_call"]), total)
    invalid_only_rate = 100.0 * safe_div(
        sum(1 for r in rows if r["tool_calls_count"] > 0 and r["valid_tool_calls_count"] == 0),
        total,
    )
    no_tool_rate = 100.0 * safe_div(sum(1 for r in rows if r["tool_calls_count"] == 0), total)
    success_rate = 100.0 * safe_div(sum(1 for r in rows if r["success"]), total)

    total_prompt_tokens = sum(r["llm_prompt_tokens"] for r in rows)
    total_completion_tokens = sum(r["llm_completion_tokens"] for r in rows)
    total_tokens = sum(r["llm_total_tokens"] for r in rows)
    total_cost = estimate_cost_usd(pricing, model_tag, total_prompt_tokens, total_completion_tokens)

    correct_answers = sum(1 for r in rows if r["answer_match"])
    tokens_per_correct = safe_div(total_tokens, correct_answers)

    summary = {
        "Model": model_tag,
        "Queries": total,
        "AnsAcc": round(eval_metrics.get("answer_accuracy", 0.0), 2),
        "PerceptionF1": round(eval_metrics.get("perception_f1", 0.0), 2),
        "AnalysisF1": round(eval_metrics.get("analysis_f1", 0.0), 2),
        "TransformationF1": round(eval_metrics.get("transformation_f1", 0.0), 2),
        "DetectionF1": round(eval_metrics.get("detection_f1", 0.0), 2),
        "ValidToolCallRate": round(valid_tool_call_rate, 2),
        "InvalidOnlyRate": round(invalid_only_rate, 2),
        "NoToolRate": round(no_tool_rate, 2),
        "SuccessRate": round(success_rate, 2),
        "AvgToolCalls": round(safe_div(sum(r["tool_calls_count"] for r in rows), total), 3),
        "AvgValidToolCalls": round(safe_div(sum(r["valid_tool_calls_count"] for r in rows), total), 3),
        "AvgSteps": round(safe_div(sum(r["step_count"] for r in rows), total), 3),
        "AvgLLMCalls": round(safe_div(sum(r["llm_calls"] for r in rows), total), 3),
        "AvgPromptTok": round(safe_div(total_prompt_tokens, total), 2),
        "AvgCompletionTok": round(safe_div(total_completion_tokens, total), 2),
        "AvgTotalTok": round(safe_div(total_tokens, total), 2),
        "TotalTokens": int(total_tokens),
        "AvgLLMTimeSec": round(safe_div(sum(r["llm_time_sec"] for r in rows), total), 4),
        "AvgToolTimeSec": round(safe_div(sum(r["tool_time_sec"] for r in rows), total), 4),
        "AvgTotalTimeSec": round(safe_div(sum(r["total_time_sec"] for r in rows), total), 4),
        "TokensPerCorrectAnswer": round(tokens_per_correct, 2),
        "EstCostUSD": round(total_cost, 6),
    }
    if source_kind == "per_query_metrics" and raw_payload:
        summary["ToolSelAcc"] = round(float(raw_payload.get("tool_selection_accuracy", 0.0)) * 100.0, 2)
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("# Audio Benchmark Summary\n\nNo rows.\n")
        return

    cols = list(rows[0].keys())
    lines = ["# Audio Benchmark Summary", "", f"Generated: {datetime.now().isoformat(timespec='seconds')}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        vals = [str(row.get(c, "")) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build audio benchmark summary tables.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec: model_tag=path/to/results.json (can repeat).",
    )
    parser.add_argument("--toolmeta", default="data/audio_dataset/toolmeta.json")
    parser.add_argument("--pricing", default=None, help="Optional pricing JSON mapping model_tag -> {input, output}.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.toolmeta) as f:
        valid_tools = set(json.load(f).keys())

    pricing = load_pricing(args.pricing)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"outputs/audio_benchmark_tables/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    for spec in args.run:
        model_tag, path = parse_run_spec(spec)
        model_tag = sanitize_tag(model_tag)
        with path.open() as f:
            raw = json.load(f)
        results, source_kind = flatten_results(raw)

        per_query = build_per_query_rows(model_tag, results, valid_tools)
        summary = aggregate_summary(
            model_tag=model_tag,
            results=results,
            rows=per_query,
            pricing=pricing,
            source_kind=source_kind,
            raw_payload=raw,
        )
        summary_rows.append(summary)

        per_query_path = output_dir / f"{model_tag}_per_query.csv"
        write_csv(per_query_path, per_query)

    # Sort by AnsAcc desc, then ValidToolCallRate desc.
    summary_rows.sort(key=lambda r: (float(r.get("AnsAcc", 0.0)), float(r.get("ValidToolCallRate", 0.0))), reverse=True)

    summary_csv = output_dir / "benchmark_summary.csv"
    summary_md = output_dir / "benchmark_summary.md"
    write_csv(summary_csv, summary_rows)
    write_markdown_table(summary_md, summary_rows)

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary MD:  {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
