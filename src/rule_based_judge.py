#!/usr/bin/env python3
"""
Rule-Based Partial Credit Evaluation (No LLM Calls)

This script implements deterministic, mathematical scoring for audio benchmark results.
- NO LLM API calls
- Reproducible scores
- Unbiased across models
- Partial credit for numerical answers (tolerance bands from paper)
- Whitelist/blacklist validation
- Proper Ans+I calculation with tool correctness

Scoring Methodology (from paper):
1. Numerical Values:
   - Exact match: 1.0
   - Within ±10%: 0.9
   - Within ±20%: 0.7
   - Within ±30%: 0.5
   - Beyond ±30%: 0.0

2. Whitelist Matching:
   - Partial credit: score = (matched_groups / total_groups)

3. Blacklist Violation:
   - ANY blacklist term found → Score = 0.0

4. Tool Correctness:
   - F1 score between predicted and expected tools

5. Ans+I (Answer + Instruction):
   - Answer score × Tool F1 score
"""

import json
import re
import argparse
from typing import Dict, List, Tuple, Any, Set
from pathlib import Path
from collections import defaultdict
import csv


class RuleBasedJudge:
    """Deterministic rule-based evaluation with partial credit."""

    def __init__(self, case_sensitive: bool = False, strict_whitelist: bool = False):
        """
        Args:
            case_sensitive: Whether to use case-sensitive matching (default: False)
            strict_whitelist: If True, ALL whitelist groups required (default: False for partial credit)
        """
        self.case_sensitive = case_sensitive
        self.strict_whitelist = strict_whitelist

    def normalize_text(self, text: str) -> str:
        """Normalize text for matching (case-insensitive, whitespace normalized)."""
        if not self.case_sensitive:
            text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def normalize_number(self, num_str: str) -> float:
        """
        Normalize numerical string with unit handling.

        Examples:
            "3.93" -> 3.93
            "3.93%" -> 3.93 (percentage as-is)
            "0.0393" -> 0.0393
            "393 basis points" -> 3.93 (if context suggests %)
        """
        num_str = str(num_str).strip()

        # Remove percentage sign but keep the value
        if '%' in num_str:
            num_str = num_str.replace('%', '')

        try:
            return float(num_str)
        except ValueError:
            return None

    def extract_numbers(self, text: str) -> List[float]:
        """
        Extract all numerical values from text with unit normalization.

        Examples:
            "3.93" -> [3.93]
            "quality is 3.9 out of 5" -> [3.9, 5.0]
            "1.007%" -> [1.007]
            "increased by 25%" -> [25.0]
            "0.8578 confidence" -> [0.8578]
        """
        if not text:
            return []

        # Pattern for numbers: optional sign, digits, optional decimal, optional percentage
        # Handles: -3.14, 42, 0.5, 99.9%, 1e-5
        pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?%?'
        matches = re.findall(pattern, str(text))

        numbers = []
        for match in matches:
            num = self.normalize_number(match)
            if num is not None:
                numbers.append(num)

        return numbers

    def calculate_numerical_score(self, gt_number: float, pred_numbers: List[float]) -> Tuple[float, str, float]:
        """
        Calculate score for numerical match with tolerance bands.

        Args:
            gt_number: Ground truth number
            pred_numbers: All numbers found in prediction

        Returns:
            (score, reasoning, best_error_pct)
        """
        if not pred_numbers:
            return 0.0, "No numerical values found in prediction", 100.0

        # Find closest match
        best_score = 0.0
        best_error = float('inf')
        best_pred = None

        for pred_num in pred_numbers:
            # Calculate relative error: |pred - gt| / |gt|
            if abs(gt_number) < 1e-10:  # Avoid division by zero
                error_pct = abs(pred_num - gt_number) * 100
            else:
                error_pct = abs(pred_num - gt_number) / abs(gt_number) * 100

            # Assign score based on tolerance bands
            if error_pct < 0.01:  # Essentially exact match
                score = 1.0
            elif error_pct <= 10.0:
                score = 0.9
            elif error_pct <= 20.0:
                score = 0.7
            elif error_pct <= 30.0:
                score = 0.5
            else:
                score = 0.0

            if score > best_score or (score == best_score and error_pct < best_error):
                best_score = score
                best_error = error_pct
                best_pred = pred_num

        # Generate reasoning
        if best_score == 1.0:
            reason = f"Exact match: GT={gt_number}, Pred={best_pred}"
        elif best_score >= 0.5:
            reason = f"Within ±{int(best_error)}% tolerance: GT={gt_number}, Pred={best_pred} (error={best_error:.2f}%)"
        else:
            reason = f"Beyond ±30% tolerance: GT={gt_number}, Pred={best_pred} (error={best_error:.2f}%)"

        return best_score, reason, best_error

    def check_whitelist_match(self, prediction: str, whitelist_groups: List[List[str]]) -> Tuple[float, List[bool], str]:
        """
        Check whitelist matching with partial credit.

        Each whitelist group is an OR-group: at least ONE term must appear.
        Score = (matched_groups / total_groups)

        Args:
            prediction: Prediction text
            whitelist_groups: List of OR-groups, e.g., [["real"], ["3.93"]]

        Returns:
            (score, matched_flags, reasoning)
        """
        if not whitelist_groups:
            return 1.0, [], "No whitelist requirements"

        pred_norm = self.normalize_text(prediction)
        matched_groups = []

        for group in whitelist_groups:
            group_matched = False
            for term in group:
                term_norm = self.normalize_text(str(term))
                if term_norm in pred_norm:
                    group_matched = True
                    break
            matched_groups.append(group_matched)

        num_matched = sum(matched_groups)
        total_groups = len(whitelist_groups)

        if self.strict_whitelist:
            # All groups must match
            score = 1.0 if num_matched == total_groups else 0.0
            reason = f"Whitelist: {num_matched}/{total_groups} groups matched (strict mode)"
        else:
            # Partial credit
            score = num_matched / total_groups if total_groups > 0 else 1.0
            reason = f"Whitelist: {num_matched}/{total_groups} groups matched"

        return score, matched_groups, reason

    def check_blacklist_violation(self, prediction: str, blacklist_groups: List[List[str]]) -> Tuple[bool, str]:
        """
        Check for blacklist violations.

        ANY blacklist term found → immediate failure (score = 0.0)

        Args:
            prediction: Prediction text
            blacklist_groups: List of forbidden terms

        Returns:
            (violated, reasoning)
        """
        if not blacklist_groups:
            return False, "No blacklist restrictions"

        pred_norm = self.normalize_text(prediction)

        for group in blacklist_groups:
            for term in group:
                term_norm = self.normalize_text(str(term))
                if term_norm in pred_norm:
                    return True, f"Blacklist violation: contains forbidden term '{term}'"

        return False, "No blacklist violations"

    def compute_tool_f1(self, predicted_tools: List[str], expected_tools: List[str]) -> Tuple[float, str]:
        """
        Compute F1 score for tool correctness.

        Args:
            predicted_tools: List of tools called by model
            expected_tools: List of ground truth tools

        Returns:
            (f1_score, reasoning)
        """
        if not predicted_tools and not expected_tools:
            return 1.0, "No tools required or called"

        if not predicted_tools:
            return 0.0, f"No tools called (expected {len(expected_tools)})"

        if not expected_tools:
            return 0.0, f"Tools called but none expected (called {len(predicted_tools)})"

        pred_set = set(predicted_tools)
        exp_set = set(expected_tools)

        intersection = pred_set & exp_set

        precision = len(intersection) / len(pred_set) if pred_set else 0.0
        recall = len(intersection) / len(exp_set) if exp_set else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        reason = f"Tool F1={f1:.2f} (P={precision:.2f}, R={recall:.2f}): {len(intersection)}/{len(exp_set)} correct"

        return f1, reason

    def evaluate_answer(self, prediction: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main evaluation function with partial credit scoring.

        Args:
            prediction: Model's final answer text
            ground_truth: Dict with 'whitelist' and 'blacklist' keys

        Returns:
            {
                'score': float (0.0-1.0),
                'numerical_score': float or None,
                'whitelist_score': float or None,
                'blacklist_violated': bool,
                'reasoning': str,
                'matched_whitelist_groups': list,
                'numerical_error_pct': float or None,
                'category': str  # 'numerical', 'whitelist', 'combined', 'blacklist_fail'
            }
        """
        if not prediction:
            return {
                'score': 0.0,
                'numerical_score': None,
                'whitelist_score': None,
                'blacklist_violated': False,
                'reasoning': 'Empty prediction',
                'matched_whitelist_groups': [],
                'numerical_error_pct': None,
                'category': 'empty'
            }

        whitelist_groups = ground_truth.get('whitelist', [])
        blacklist_groups = ground_truth.get('blacklist', [])

        # Step 1: Check blacklist (immediate failure)
        blacklist_violated, blacklist_reason = self.check_blacklist_violation(prediction, blacklist_groups)
        if blacklist_violated:
            return {
                'score': 0.0,
                'numerical_score': None,
                'whitelist_score': None,
                'blacklist_violated': True,
                'reasoning': blacklist_reason,
                'matched_whitelist_groups': [],
                'numerical_error_pct': None,
                'category': 'blacklist_fail'
            }

        # Step 2: Extract numerical values from both GT and prediction
        pred_numbers = self.extract_numbers(prediction)

        # Find numerical groups in whitelist
        numerical_groups = []
        non_numerical_groups = []
        for group in whitelist_groups:
            # Check if group contains numbers
            group_numbers = []
            for term in group:
                nums = self.extract_numbers(str(term))
                group_numbers.extend(nums)

            if group_numbers:
                numerical_groups.append(group_numbers[0])  # Take first number
            else:
                non_numerical_groups.append(group)

        # Step 3: Score numerical matches
        numerical_score = None
        numerical_error = None
        numerical_reasons = []

        if numerical_groups:
            scores = []
            errors = []
            for gt_num in numerical_groups:
                score, reason, error = self.calculate_numerical_score(gt_num, pred_numbers)
                scores.append(score)
                errors.append(error)
                numerical_reasons.append(reason)

            # Average numerical scores
            numerical_score = sum(scores) / len(scores) if scores else 0.0
            numerical_error = sum(errors) / len(errors) if errors else None

        # Step 4: Score whitelist text matches (non-numerical)
        whitelist_score = None
        whitelist_reason = ""
        matched_groups = []

        if non_numerical_groups:
            whitelist_score, matched_groups, whitelist_reason = self.check_whitelist_match(
                prediction, non_numerical_groups
            )

        # Step 5: Combine scores
        if numerical_score is not None and whitelist_score is not None:
            # Both numerical and whitelist: multiply
            final_score = numerical_score * whitelist_score
            reasoning = f"Combined: numerical={numerical_score:.2f} × whitelist={whitelist_score:.2f} = {final_score:.2f}. {' | '.join(numerical_reasons)} | {whitelist_reason}"
            category = 'combined'
        elif numerical_score is not None:
            # Only numerical
            final_score = numerical_score
            reasoning = ' | '.join(numerical_reasons)
            category = 'numerical'
        elif whitelist_score is not None:
            # Only whitelist
            final_score = whitelist_score
            reasoning = whitelist_reason
            category = 'whitelist'
        else:
            # No scoring criteria (edge case)
            final_score = 1.0
            reasoning = "No whitelist or numerical requirements"
            category = 'no_criteria'

        return {
            'score': final_score,
            'numerical_score': numerical_score,
            'whitelist_score': whitelist_score,
            'blacklist_violated': False,
            'reasoning': reasoning,
            'matched_whitelist_groups': matched_groups,
            'numerical_error_pct': numerical_error,
            'category': category
        }


def load_ground_truth(dataset_path: str) -> Dict[str, Dict]:
    """Load ground truth from dataset JSON."""
    with open(dataset_path) as f:
        data = json.load(f)

    gt_dict = {}
    for task in data:
        task_id = task['id']

        # Extract tool names from tool dicts
        tools = task.get('tools', [])
        if tools and isinstance(tools[0], dict):
            tool_names = [t['name'] for t in tools]
        else:
            tool_names = tools

        gt_dict[task_id] = {
            'answer': task['groundtruth_answer'],
            'tools': tool_names
        }

    return gt_dict


def extract_final_answer(result: Dict) -> str:
    """Extract final answer from result trace."""
    # Try to get final answer from trace
    if 'trace' in result and result['trace']:
        steps = result['trace'].get('steps', [])
        for step in reversed(steps):
            if step.get('is_final') and step.get('final_answer'):
                return str(step['final_answer'])

    # Fallback: empty string if not found
    return ""


def extract_tools_called(result: Dict) -> List[str]:
    """Extract tools called from result trace."""
    tools_called = []

    if 'trace' in result and result['trace']:
        steps = result['trace'].get('steps', [])
        for step in steps:
            action = step.get('action')
            if action and action not in tools_called:
                tools_called.append(action)

    return tools_called


def evaluate_results(results_path: str, dataset_path: str,
                     case_sensitive: bool = False,
                     strict_whitelist: bool = False,
                     output_path: str = None) -> Dict:
    """
    Evaluate results with rule-based partial credit scoring.

    Args:
        results_path: Path to results.json
        dataset_path: Path to dataset.json with ground truth
        case_sensitive: Use case-sensitive matching
        strict_whitelist: Require ALL whitelist groups to match
        output_path: Path to save output JSON (optional)

    Returns:
        Evaluation metrics dictionary
    """
    # Load data
    with open(results_path) as f:
        results = json.load(f)

    ground_truth = load_ground_truth(dataset_path)

    # Initialize judge
    judge = RuleBasedJudge(case_sensitive=case_sensitive, strict_whitelist=strict_whitelist)

    # Evaluate each task
    task_results = []
    total_answer_score = 0.0
    total_tool_f1 = 0.0
    total_ans_i_score = 0.0

    perfect_count = 0  # score >= 0.95
    partial_count = 0  # 0.70 <= score < 0.95
    incorrect_count = 0  # score < 0.70

    category_stats = defaultdict(int)
    numerical_errors = []

    for task_id, result in results.items():
        # Get ground truth
        if task_id not in ground_truth:
            print(f"Warning: No ground truth for task {task_id}")
            continue

        gt = ground_truth[task_id]

        # Extract final answer and tools
        prediction = extract_final_answer(result)
        predicted_tools = extract_tools_called(result)
        expected_tools = gt['tools']

        # Evaluate answer
        eval_result = judge.evaluate_answer(prediction, gt['answer'])

        # Evaluate tools
        tool_f1, tool_reason = judge.compute_tool_f1(predicted_tools, expected_tools)

        # Calculate Ans+I (Answer × Tools)
        ans_i_score = eval_result['score'] * tool_f1

        # Aggregate stats
        answer_score = eval_result['score']
        total_answer_score += answer_score
        total_tool_f1 += tool_f1
        total_ans_i_score += ans_i_score

        if answer_score >= 0.95:
            perfect_count += 1
        elif answer_score >= 0.70:
            partial_count += 1
        else:
            incorrect_count += 1

        category_stats[eval_result['category']] += 1

        if eval_result['numerical_error_pct'] is not None:
            numerical_errors.append(eval_result['numerical_error_pct'])

        # Store result
        task_results.append({
            'task_id': task_id,
            'answer_score': answer_score,
            'tool_f1': tool_f1,
            'ans_i_score': ans_i_score,
            'reasoning': eval_result['reasoning'],
            'tool_reasoning': tool_reason,
            'category': eval_result['category'],
            'numerical_score': eval_result['numerical_score'],
            'whitelist_score': eval_result['whitelist_score'],
            'blacklist_violated': eval_result['blacklist_violated'],
            'numerical_error_pct': eval_result['numerical_error_pct'],
            'predicted_tools': predicted_tools,
            'expected_tools': expected_tools,
            'prediction': prediction[:200],  # Truncate for readability
            'ground_truth': gt['answer']
        })

    # Calculate metrics
    total_tasks = len(task_results)
    answer_accuracy = (total_answer_score / total_tasks * 100) if total_tasks > 0 else 0.0
    tool_f1_avg = (total_tool_f1 / total_tasks * 100) if total_tasks > 0 else 0.0
    ans_i_accuracy = (total_ans_i_score / total_tasks * 100) if total_tasks > 0 else 0.0

    answer_accuracy_binary = (perfect_count / total_tasks * 100) if total_tasks > 0 else 0.0
    answer_accuracy_threshold_70 = ((perfect_count + partial_count) / total_tasks * 100) if total_tasks > 0 else 0.0

    avg_numerical_error = sum(numerical_errors) / len(numerical_errors) if numerical_errors else 0.0

    # Compile final metrics
    metrics = {
        'answer_accuracy': round(answer_accuracy, 2),
        'answer_accuracy_binary': round(answer_accuracy_binary, 2),
        'answer_accuracy_threshold_70': round(answer_accuracy_threshold_70, 2),
        'tool_f1': round(tool_f1_avg, 2),
        'ans_i_accuracy': round(ans_i_accuracy, 2),
        'total_tasks': total_tasks,
        'perfect_answers': perfect_count,
        'partial_credit_answers': partial_count,
        'incorrect_answers': incorrect_count,
        'avg_numerical_error_pct': round(avg_numerical_error, 2),
        'category_distribution': dict(category_stats),
        'task_results': task_results
    }

    # Save output
    if output_path:
        output_path = Path(output_path)

        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save TXT summary
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("Rule-Based Evaluation Report (Partial Credit Scoring)\n")
            f.write("=" * 70 + "\n\n")

            f.write("CORE METRICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Answer Accuracy (AnsAcc):        {metrics['answer_accuracy']:.2f}%\n")
            f.write(f"  Tool F1 Score:                   {metrics['tool_f1']:.2f}%\n")
            f.write(f"  Answer+Instruction (Ans+I):      {metrics['ans_i_accuracy']:.2f}%\n\n")

            f.write("ANSWER QUALITY BREAKDOWN:\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Perfect Answers (≥0.95):         {perfect_count}/{total_tasks} ({perfect_count/total_tasks*100:.1f}%)\n")
            f.write(f"  Partial Credit (0.70-0.94):      {partial_count}/{total_tasks} ({partial_count/total_tasks*100:.1f}%)\n")
            f.write(f"  Incorrect (<0.70):                {incorrect_count}/{total_tasks} ({incorrect_count/total_tasks*100:.1f}%)\n\n")

            f.write("ALTERNATIVE METRICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Binary Accuracy (≥0.95):         {metrics['answer_accuracy_binary']:.2f}%\n")
            f.write(f"  Threshold Accuracy (≥0.70):      {metrics['answer_accuracy_threshold_70']:.2f}%\n\n")

            if numerical_errors:
                f.write("NUMERICAL ERROR STATISTICS:\n")
                f.write("-" * 50 + "\n")
                f.write(f"  Avg Numerical Error:              {avg_numerical_error:.2f}%\n")
                f.write(f"  Tasks with numerical answers:     {len(numerical_errors)}/{total_tasks}\n\n")

            f.write("CATEGORY DISTRIBUTION:\n")
            f.write("-" * 50 + "\n")
            for cat, count in sorted(category_stats.items()):
                f.write(f"  {cat:20s}: {count:4d} ({count/total_tasks*100:.1f}%)\n")

            f.write("\n" + "=" * 70 + "\n")

        # Save CSV
        csv_path = output_path.with_suffix('.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'task_id', 'answer_score', 'tool_f1', 'ans_i_score', 'category',
                'numerical_score', 'whitelist_score', 'blacklist_violated',
                'numerical_error_pct', 'reasoning', 'tool_reasoning'
            ])
            writer.writeheader()
            for task in task_results:
                writer.writerow({
                    'task_id': task['task_id'],
                    'answer_score': task['answer_score'],
                    'tool_f1': task['tool_f1'],
                    'ans_i_score': task['ans_i_score'],
                    'category': task['category'],
                    'numerical_score': task['numerical_score'],
                    'whitelist_score': task['whitelist_score'],
                    'blacklist_violated': task['blacklist_violated'],
                    'numerical_error_pct': task['numerical_error_pct'],
                    'reasoning': task['reasoning'],
                    'tool_reasoning': task['tool_reasoning']
                })

        print(f"\nResults saved to:")
        print(f"  JSON: {output_path}")
        print(f"  TXT:  {txt_path}")
        print(f"  CSV:  {csv_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Rule-based partial credit evaluation')
    parser.add_argument('--results', required=True, help='Path to results.json')
    parser.add_argument('--dataset', required=True, help='Path to dataset.json with ground truth')
    parser.add_argument('--output', required=True, help='Path to save evaluation results')
    parser.add_argument('--case-sensitive', action='store_true', help='Use case-sensitive matching')
    parser.add_argument('--strict-whitelist', action='store_true',
                       help='Require ALL whitelist groups (no partial credit)')

    args = parser.parse_args()

    # Run evaluation
    metrics = evaluate_results(
        results_path=args.results,
        dataset_path=args.dataset,
        case_sensitive=args.case_sensitive,
        strict_whitelist=args.strict_whitelist,
        output_path=args.output
    )

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Answer Accuracy (AnsAcc):        {metrics['answer_accuracy']:.2f}%")
    print(f"Tool F1 Score:                   {metrics['tool_f1']:.2f}%")
    print(f"Answer+Instruction (Ans+I):      {metrics['ans_i_accuracy']:.2f}%")
    print(f"Perfect / Partial / Incorrect:   {metrics['perfect_answers']} / {metrics['partial_credit_answers']} / {metrics['incorrect_answers']}")
    print("=" * 70)


if __name__ == '__main__':
    main()
