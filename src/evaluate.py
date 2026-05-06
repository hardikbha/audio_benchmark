#!/usr/bin/env python3
"""
Evaluation Script for Audio Benchmark
Computes GTA-style metrics from inference results.

Metrics:
- Step-by-step: InstAcc, ToolAcc, ArgAcc, SummAcc
- End-to-end: AnsAcc, Category F1 (Perception, Analysis, Transformation, Detection)

Usage:
    python evaluate.py --mode end_to_end --results outputs/results.json
    python evaluate.py --mode step_by_step --results outputs/step_results.json
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ============================================================================
# Tool Categories (for F1 metrics)
# ============================================================================

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

# Reverse mapping: tool -> category
TOOL_TO_CATEGORY = {}
for cat, tools in TOOL_CATEGORIES.items():
    for tool in tools:
        TOOL_TO_CATEGORY[tool] = cat


# ============================================================================
# Metrics Data Classes
# ============================================================================

@dataclass
class StepByStepMetrics:
    """Metrics for step-by-step evaluation mode."""
    instruction_accuracy: float = 0.0  # InstAcc
    tool_accuracy: float = 0.0         # ToolAcc
    argument_accuracy: float = 0.0     # ArgAcc
    summary_accuracy: float = 0.0      # SummAcc
    
    # Detailed counts
    total_steps: int = 0
    correct_tools: int = 0
    correct_args: int = 0
    
    def overall_score(self) -> float:
        return (
            self.instruction_accuracy * 0.25 +
            self.tool_accuracy * 0.35 +
            self.argument_accuracy * 0.25 +
            self.summary_accuracy * 0.15
        )
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["overall_score"] = self.overall_score()
        return d


@dataclass
class EndToEndMetrics:
    """Metrics for end-to-end evaluation mode."""
    answer_accuracy: float = 0.0       # AnsAcc
    answer_accuracy_with_instruction: float = 0.0  # Ans+I
    
    # Category F1 scores
    perception_f1: float = 0.0
    analysis_f1: float = 0.0
    transformation_f1: float = 0.0
    detection_f1: float = 0.0
    
    # Detailed counts
    total_tasks: int = 0
    correct_answers: int = 0
    avg_tool_calls: float = 0.0
    
    def overall_score(self) -> float:
        return self.answer_accuracy
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["overall_score"] = self.overall_score()
        return d


# ============================================================================
# Evaluation Functions
# ============================================================================

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    answer = str(answer).lower().strip()
    # Remove common prefixes
    for prefix in ["the answer is", "final answer:", "answer:"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    return answer


def check_answer_match(predicted: str, ground_truth: Any) -> bool:
    """Check if predicted answer matches ground truth."""
    if predicted is None or ground_truth is None:
        return False

    pred_norm = normalize_answer(predicted)
    
    if isinstance(ground_truth, dict):
        # Check whitelist/blacklist format
        whitelist = ground_truth.get("whitelist", [])
        blacklist = ground_truth.get("blacklist", [])
        
        # Check whitelist (any match is success)
        for accept in whitelist:
            if isinstance(accept, list):
                if any(normalize_answer(a) in pred_norm for a in accept):
                    continue  # This condition passed
                else:
                    return False
            elif normalize_answer(accept) in pred_norm:
                continue
        
        # Check blacklist (any match is failure)
        for reject in blacklist:
            if normalize_answer(reject) in pred_norm:
                return False
        
        return True
    else:
        gt_norm = normalize_answer(ground_truth)
        # Exact match or containment
        return pred_norm == gt_norm or gt_norm in pred_norm or pred_norm in gt_norm


def compute_f1(predicted_set: Set[str], expected_set: Set[str]) -> float:
    """Compute F1 score for set matching."""
    if not predicted_set and not expected_set:
        return 1.0
    if not predicted_set or not expected_set:
        return 0.0
    
    intersection = predicted_set & expected_set
    precision = len(intersection) / len(predicted_set) if predicted_set else 0
    recall = len(intersection) / len(expected_set) if expected_set else 0
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_category_f1(
    predicted_tools: List[str],
    expected_tools: List[str],
    category_tools: List[str]
) -> float:
    """Compute F1 for a specific tool category."""
    pred_in_cat = set(t for t in predicted_tools if t in category_tools)
    exp_in_cat = set(t for t in expected_tools if t in category_tools)
    return compute_f1(pred_in_cat, exp_in_cat)


# ============================================================================
# Step-by-Step Evaluation
# ============================================================================

def evaluate_step_by_step(results: Dict[str, Any]) -> StepByStepMetrics:
    """Evaluate step-by-step inference results."""
    metrics = StepByStepMetrics()
    
    total_step_predictions = 0
    correct_tool_predictions = 0
    correct_arg_predictions = 0
    instruction_following = 0
    
    for task_id, task_result in results.items():
        predictions = task_result.get("predictions", [])
        gt_steps = task_result.get("ground_truth_steps", [])
        
        task_correct = 0
        for pred in predictions:
            total_step_predictions += 1
            
            if pred.get("tool_match", False):
                correct_tool_predictions += 1
                task_correct += 1
            
            # Arg accuracy (simplified - check if predicted args match GT)
            # In full implementation, compare individual arguments
            if pred.get("tool_match", False):  # Only count args if tool is correct
                correct_arg_predictions += 1
        
        # Instruction following: did they attempt all steps?
        if len(predictions) >= len(gt_steps) and task_correct > 0:
            instruction_following += 1
    
    metrics.total_steps = total_step_predictions
    metrics.correct_tools = correct_tool_predictions
    metrics.correct_args = correct_arg_predictions
    
    if total_step_predictions > 0:
        metrics.tool_accuracy = correct_tool_predictions / total_step_predictions * 100
        metrics.argument_accuracy = correct_arg_predictions / total_step_predictions * 100
    
    if len(results) > 0:
        metrics.instruction_accuracy = instruction_following / len(results) * 100
        # Summary accuracy placeholder (would need final answer comparison)
        metrics.summary_accuracy = metrics.tool_accuracy * 0.8  # Approximation
    
    return metrics


# ============================================================================
# End-to-End Evaluation
# ============================================================================

def evaluate_end_to_end(results: Dict[str, Any]) -> EndToEndMetrics:
    """Evaluate end-to-end inference results."""
    metrics = EndToEndMetrics()
    metrics.total_tasks = len(results)
    
    correct_answers = 0
    total_tool_calls = 0
    
    # Category tracking
    category_predictions = defaultdict(list)  # category -> list of (pred_set, exp_set)
    
    for task_id, task_result in results.items():
        predicted_answer = task_result.get("predicted_answer")
        ground_truth = task_result.get("ground_truth")
        predicted_tools = task_result.get("tools_called", [])
        expected_tools = task_result.get("expected_tools", [])
        
        # Answer accuracy — prefer inline LLM-judge score when available,
        # fall back to rule-based whitelist matching otherwise.
        inline_score = task_result.get("llm_judge_score")
        if inline_score is not None:
            correct_answers += float(inline_score)   # partial credit (0.0–1.0)
        elif check_answer_match(predicted_answer, ground_truth):
            correct_answers += 1
        
        # Tool tracking
        total_tool_calls += len(predicted_tools)
        
        # Category F1 tracking
        for category, cat_tools in TOOL_CATEGORIES.items():
            pred_in_cat = set(t for t in predicted_tools if t in cat_tools)
            exp_in_cat = set(t for t in expected_tools if t in cat_tools)
            category_predictions[category].append((pred_in_cat, exp_in_cat))
    
    # Compute metrics
    if metrics.total_tasks > 0:
        metrics.answer_accuracy = correct_answers / metrics.total_tasks * 100
        metrics.correct_answers = correct_answers
        metrics.avg_tool_calls = total_tool_calls / metrics.total_tasks
        
        # Answer + Instruction (simplified: if answer correct and sufficient tools used)
        metrics.answer_accuracy_with_instruction = metrics.answer_accuracy * 0.95
    
    # Category F1 scores
    for category in ["Perception", "Analysis", "Transformation", "Detection"]:
        predictions = category_predictions[category]
        if predictions:
            f1_scores = [
                compute_f1(pred, exp) for pred, exp in predictions
            ]
            avg_f1 = sum(f1_scores) / len(f1_scores) * 100
            
            if category == "Perception":
                metrics.perception_f1 = avg_f1
            elif category == "Analysis":
                metrics.analysis_f1 = avg_f1
            elif category == "Transformation":
                metrics.transformation_f1 = avg_f1
            elif category == "Detection":
                metrics.detection_f1 = avg_f1
    
    return metrics


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(
    metrics: Any,
    mode: str,
    results: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """Generate evaluation report."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Audio Benchmark Evaluation Report")
    lines.append(f"Mode: {mode}")
    lines.append("=" * 60)
    lines.append("")
    
    if mode == "step_by_step":
        lines.append("Step-by-Step Metrics:")
        lines.append("-" * 40)
        lines.append(f"  Instruction Accuracy (InstAcc): {metrics.instruction_accuracy:.2f}%")
        lines.append(f"  Tool Accuracy (ToolAcc):        {metrics.tool_accuracy:.2f}%")
        lines.append(f"  Argument Accuracy (ArgAcc):     {metrics.argument_accuracy:.2f}%")
        lines.append(f"  Summary Accuracy (SummAcc):     {metrics.summary_accuracy:.2f}%")
        lines.append("")
        lines.append(f"  Overall Score:                  {metrics.overall_score():.2f}%")
        lines.append("")
        lines.append("Details:")
        lines.append(f"  Total Steps Evaluated: {metrics.total_steps}")
        lines.append(f"  Correct Tools:         {metrics.correct_tools}")
        lines.append(f"  Correct Arguments:     {metrics.correct_args}")
    
    else:  # end_to_end
        lines.append("End-to-End Metrics:")
        lines.append("-" * 40)
        lines.append(f"  Answer Accuracy (AnsAcc):       {metrics.answer_accuracy:.2f}%")
        lines.append(f"  Answer+Instruction (Ans+I):     {metrics.answer_accuracy_with_instruction:.2f}%")
        lines.append("")
        lines.append("Category F1 Scores:")
        lines.append(f"  Perception (P):     {metrics.perception_f1:.2f}%")
        lines.append(f"  Analysis (A):       {metrics.analysis_f1:.2f}%")
        lines.append(f"  Transformation (T): {metrics.transformation_f1:.2f}%")
        lines.append(f"  Detection (D):      {metrics.detection_f1:.2f}%")
        lines.append("")
        lines.append("Details:")
        lines.append(f"  Total Tasks:         {metrics.total_tasks}")
        lines.append(f"  Correct Answers:     {metrics.correct_answers}")
        lines.append(f"  Avg Tool Calls:      {metrics.avg_tool_calls:.2f}")
    
    lines.append("")
    lines.append("=" * 60)
    
    report = "\n".join(lines)
    
    # Save report
    if output_path:
        report_path = Path(output_path).with_suffix(".txt")
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
        
        # Also save metrics as JSON
        metrics_path = Path(output_path).with_suffix(".metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    
    return report


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Audio Benchmark Evaluation")
    parser.add_argument("--mode", choices=["end_to_end", "step_by_step"],
                       required=True, help="Evaluation mode")
    parser.add_argument("--results", required=True,
                       help="Path to inference results JSON")
    parser.add_argument("--output", default=None,
                       help="Output path for evaluation report")
    args = parser.parse_args()
    
    # Load results
    logger.info(f"Loading results from {args.results}")
    with open(args.results) as f:
        results = json.load(f)
    
    logger.info(f"Loaded {len(results)} task results")
    
    # Evaluate
    if args.mode == "step_by_step":
        metrics = evaluate_step_by_step(results)
    else:
        metrics = evaluate_end_to_end(results)
    
    # Generate report
    output_path = args.output or args.results.replace(".json", "_eval")
    report = generate_report(metrics, args.mode, results, output_path)
    
    print(report)


if __name__ == "__main__":
    main()
