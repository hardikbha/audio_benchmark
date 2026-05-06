#!/usr/bin/env python3
"""
LLM-as-Judge Evaluation Script with Partial Scoring
Implements tolerance-based numerical scoring, whitelist/blacklist validation,
and semantic equivalence checking for audio benchmark results.

Based on the methodology from AudioToolAgent paper:
- Numerical tolerance: ±10% (0.9), ±20% (0.7), ±30% (0.5)
- Whitelist validation with partial credit
- Blacklist immediate failure (0.0)
- Counting answers require exact match

Usage:
    python llm_judge_eval.py \
        --results outputs/voxtral_mini_500_*/results.json \
        --output outputs/voxtral_mini_500_*/llm_judge_eval.json \
        --llm_provider openai \
        --llm_model claude-sonnet-4.5
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract first balanced JSON object from arbitrary model text output."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


# ============================================================================
# LLM Client Setup
# ============================================================================

def create_llm_judge_client(provider: str = "openai", model: str = "claude-sonnet-4.5", api_key: Optional[str] = None):
    """Create LLM client for judge evaluation."""
    import os

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
            base_url=os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
        )
        return client, model

    elif provider == "anthropic":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            return client, model or "claude-sonnet-4-5-20250929"
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    elif provider == "local":
        from run_inference import create_local_llm_client

        client = create_local_llm_client(model)
        return client, model

    else:
        raise ValueError(f"Unsupported provider: {provider}")


# ============================================================================
# LLM Judge System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are an expert evaluator for audio analysis tasks. Your role is to assess whether a predicted answer matches the ground truth with nuanced, partial-credit scoring.

You will receive:
1. GROUND TRUTH answer (with optional whitelist/blacklist requirements)
2. PREDICTED answer from a model
3. QUESTION context

Your task: Determine correctness score (0.0-1.0) using the scoring rubric below.

## SCORING RUBRIC

### A. Numerical Values (percentages, decimals, measurements)
For answers with numerical precision requirements:

| Relative Error | Score | Example |
|----------------|-------|---------|
| Exact match | 1.0 | GT: 1.007%, Pred: "1.007%" |
| Within ±10% | 0.9 | GT: 1.007%, Pred: "1.1%" |
| Within ±20% | 0.7 | GT: 450px², Pred: "420px²" |
| Within ±30% | 0.5 | GT: 1.007%, Pred: "1.3%" |
| Beyond ±30% | 0.0 | GT: 1.007%, Pred: "2.0%" |

**Calculation:** Relative error = |predicted - ground_truth| / ground_truth

### B. Counting Answers (discrete values)
For counting questions (e.g., "How many speakers?"):
- Exact match → 1.0
- Any deviation → 0.0 (no partial credit)

### C. Whitelist Validation
**Rules:**
- ALL required terms/values from whitelist MUST appear in prediction
- Terms can appear in different forms (e.g., "1.007" = "1.007%" = "~1%")
- If k out of n whitelist items present → Base score = k/n
- Missing ANY item → Maximum score capped at 0.5

**Example:**
- Whitelist: ["signature", "timestamp", "1.007%"]
- Prediction: "The signature region has 1.1% alteration"
- Whitelist score: 2/3 = 0.67 (missing "timestamp")
- Numerical score: 0.9 (within ±10%)
- Final: 0.67 × 0.9 = 0.60

### D. Blacklist Validation
**Rules:**
- If ANY blacklist term appears in prediction → Score = 0.0 (immediate failure)

**Example:**
- Blacklist: ["authentic", "unaltered"]
- Prediction: "The audio appears authentic with 1% modification"
- Score: 0.0 (contains blacklist term "authentic")

### E. Semantic Equivalence
Accept answers that:
- Paraphrase the ground truth while preserving core meaning
- Use different units if mathematically equivalent ("1%" = "0.01")
- Approximate within tolerance bands ("~1%" for GT: 1.007%)

### F. Combined Scoring
For answers with both whitelist and numerical values:
```
Final Score = Whitelist_Score × Numerical_Accuracy_Score
```

### G. Empty/No Answer
- If prediction is empty or "I don't know" → Score = 0.0

## OUTPUT FORMAT
You must respond with ONLY a valid JSON object (no markdown, no code blocks):

{
  "score": <float 0.0-1.0>,
  "reasoning": "<brief explanation>",
  "category": "<numerical|counting|semantic|whitelist|blacklist|combined>",
  "numerical_error_pct": <float or null>,
  "whitelist_matches": <int or null>,
  "whitelist_total": <int or null>,
  "blacklist_violated": <bool>
}

## EXAMPLES

**Example 1: Perfect Numerical Match**
Ground Truth: {"whitelist": [["1.007"], ["fake"]], "blacklist": []}
Prediction: "The audio is fake with 1.007% confidence."
Output:
{
  "score": 1.0,
  "reasoning": "Exact numerical match (1.007%) and all whitelist terms present ('fake').",
  "category": "combined",
  "numerical_error_pct": 0.0,
  "whitelist_matches": 2,
  "whitelist_total": 2,
  "blacklist_violated": false
}

**Example 2: Partial Credit (Within ±10%)**
Ground Truth: {"whitelist": [["2.08%"]], "blacklist": []}
Prediction: "Deepfake manipulation detected in facial area is 1.99%."
Output:
{
  "score": 0.9,
  "reasoning": "Numerical value 1.99% is within ±10% of ground truth 2.08% (4.3% error).",
  "category": "numerical",
  "numerical_error_pct": 4.3,
  "whitelist_matches": 1,
  "whitelist_total": 1,
  "blacklist_violated": false
}

**Example 3: Blacklist Violation**
Ground Truth: {"whitelist": [["manipulated"]], "blacklist": [["authentic"]]}
Prediction: "The image appears authentic with some compression artifacts."
Output:
{
  "score": 0.0,
  "reasoning": "Blacklist violation: contains forbidden term 'authentic'. Immediate failure.",
  "category": "blacklist",
  "numerical_error_pct": null,
  "whitelist_matches": 0,
  "whitelist_total": 1,
  "blacklist_violated": true
}

**Example 4: Counting Answer (Exact Match Required)**
Ground Truth: "6"
Prediction: "I detected 5 manipulations in the audio."
Output:
{
  "score": 0.0,
  "reasoning": "Counting answer requires exact match. Predicted 5 but ground truth is 6.",
  "category": "counting",
  "numerical_error_pct": null,
  "whitelist_matches": null,
  "whitelist_total": null,
  "blacklist_violated": false
}

Be strict but fair. Apply partial credit appropriately according to the rubric.
"""


# ============================================================================
# Judge Evaluation Functions
# ============================================================================

@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    score: float  # 0.0 - 1.0
    reasoning: str
    category: str
    numerical_error_pct: Optional[float] = None
    whitelist_matches: Optional[int] = None
    whitelist_total: Optional[int] = None
    blacklist_violated: bool = False
    raw_response: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text."""
    if text is None:
        return []
    # Match percentages, decimals, integers
    matches = re.findall(r'(\d+\.?\d*)\s*%?', str(text))
    return [float(m) for m in matches if m]


def is_counting_question(question: str) -> bool:
    """Detect if question requires counting answer."""
    counting_keywords = [
        "how many", "count", "number of", "enumerate",
        "list all", "identify all", "total"
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in counting_keywords)


def call_llm_judge(
    client,
    model: str,
    provider: str,
    question: str,
    ground_truth: Any,
    prediction: str
) -> JudgeResult:
    """Call LLM to judge answer correctness."""

    # Prepare ground truth format
    if isinstance(ground_truth, dict):
        gt_str = json.dumps(ground_truth, indent=2)
    else:
        gt_str = str(ground_truth)

    user_prompt = f"""**QUESTION:**
{question}

**GROUND TRUTH:**
{gt_str}

**PREDICTED ANSWER:**
{prediction if prediction else "(empty/no answer)"}

Evaluate the prediction and provide your judgment as JSON."""

    try:
        if provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0.0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            content = response.content[0].text
        elif provider == "openai":  # OpenAI-compatible
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=1024
            )
            content = response.choices[0].message.content
        else:  # local adapter with OpenAI-like interface
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=1024
            )
            content = response.choices[0].message.content

        # Parse JSON response
        # Remove markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\n', '', content)
            content = re.sub(r'\n```$', '', content)

        # Harmony-style wrapper fallback: analysis...assistantfinal{...}
        if "assistantfinal" in content.lower():
            content = re.split(r'assistantfinal\s*:?', content, flags=re.IGNORECASE, maxsplit=1)[-1].strip()

        # Extract JSON object from noisy model output.
        json_blob = _extract_first_json_object(content)
        if json_blob:
            content = json_blob

        result_dict = json.loads(content)

        return JudgeResult(
            score=float(result_dict.get("score", 0.0)),
            reasoning=result_dict.get("reasoning", ""),
            category=result_dict.get("category", "unknown"),
            numerical_error_pct=result_dict.get("numerical_error_pct"),
            whitelist_matches=result_dict.get("whitelist_matches"),
            whitelist_total=result_dict.get("whitelist_total"),
            blacklist_violated=result_dict.get("blacklist_violated", False),
            raw_response=content
        )

    except Exception as e:
        logger.error(f"LLM judge call failed: {e}")
        logger.error(f"Response content: {content if 'content' in locals() else 'N/A'}")
        # Fallback to 0.0 score
        return JudgeResult(
            score=0.0,
            reasoning=f"LLM judge failed: {str(e)}",
            category="error",
            raw_response=content if 'content' in locals() else ""
        )


# ============================================================================
# Evaluation Metrics
# ============================================================================

@dataclass
class LLMJudgeMetrics:
    """Comprehensive metrics from LLM-as-judge evaluation."""

    # Core accuracy metrics
    answer_accuracy: float = 0.0  # AnsAcc: Average of all scores
    answer_accuracy_with_instruction: float = 0.0  # Ans+I compatibility
    answer_accuracy_binary: float = 0.0  # Binary (score >= 0.95 → 1.0, else 0.0)
    answer_accuracy_threshold_70: float = 0.0  # Scores >= 0.70 contribute

    # Detailed statistics
    total_tasks: int = 0
    perfect_answers: int = 0  # score >= 0.95
    partial_credit_answers: int = 0  # 0.70 <= score < 0.95
    incorrect_answers: int = 0  # score < 0.70

    # Category breakdown
    numerical_tasks: int = 0
    counting_tasks: int = 0
    whitelist_tasks: int = 0
    blacklist_violations: int = 0

    # Average scores by category
    avg_score_numerical: float = 0.0
    avg_score_counting: float = 0.0
    avg_score_combined: float = 0.0

    # Error distribution
    avg_numerical_error_pct: float = 0.0
    within_10pct: int = 0
    within_20pct: int = 0
    within_30pct: int = 0
    beyond_30pct: int = 0

    # Detailed results per task
    task_results: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    def summary(self) -> str:
        """Generate summary report."""
        lines = []
        lines.append("=" * 70)
        lines.append("LLM-as-Judge Evaluation Report (Partial Credit Scoring)")
        lines.append("=" * 70)
        lines.append("")

        lines.append("CORE METRICS:")
        lines.append("-" * 50)
        lines.append(f"  Answer Accuracy (AnsAcc):        {self.answer_accuracy:.2f}%")
        lines.append(f"  Answer+Instruction (Ans+I):      {self.answer_accuracy_with_instruction:.2f}%")
        lines.append(f"  Answer Acc (Binary ≥0.95):       {self.answer_accuracy_binary:.2f}%")
        lines.append(f"  Answer Acc (Threshold ≥0.70):    {self.answer_accuracy_threshold_70:.2f}%")
        lines.append("")

        lines.append("ANSWER QUALITY BREAKDOWN:")
        lines.append("-" * 50)
        lines.append(f"  Perfect Answers (≥0.95):         {self.perfect_answers}/{self.total_tasks} ({self.perfect_answers/max(self.total_tasks,1)*100:.1f}%)")
        lines.append(f"  Partial Credit (0.70-0.94):      {self.partial_credit_answers}/{self.total_tasks} ({self.partial_credit_answers/max(self.total_tasks,1)*100:.1f}%)")
        lines.append(f"  Incorrect (<0.70):                {self.incorrect_answers}/{self.total_tasks} ({self.incorrect_answers/max(self.total_tasks,1)*100:.1f}%)")
        lines.append("")

        lines.append("CATEGORY BREAKDOWN:")
        lines.append("-" * 50)
        lines.append(f"  Numerical Tasks:                  {self.numerical_tasks} (avg score: {self.avg_score_numerical:.2f})")
        lines.append(f"  Counting Tasks:                   {self.counting_tasks} (avg score: {self.avg_score_counting:.2f})")
        lines.append(f"  Combined (Whitelist+Numerical):   {self.whitelist_tasks} (avg score: {self.avg_score_combined:.2f})")
        lines.append(f"  Blacklist Violations:             {self.blacklist_violations}")
        lines.append("")

        if self.numerical_tasks > 0:
            lines.append("NUMERICAL ERROR DISTRIBUTION:")
            lines.append("-" * 50)
            lines.append(f"  Avg Numerical Error:              {self.avg_numerical_error_pct:.2f}%")
            lines.append(f"  Within ±10% (score 0.9):          {self.within_10pct}/{self.numerical_tasks}")
            lines.append(f"  Within ±20% (score 0.7):          {self.within_20pct}/{self.numerical_tasks}")
            lines.append(f"  Within ±30% (score 0.5):          {self.within_30pct}/{self.numerical_tasks}")
            lines.append(f"  Beyond ±30% (score 0.0):          {self.beyond_30pct}/{self.numerical_tasks}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


def evaluate_with_llm_judge(
    results: Dict[str, Any],
    llm_client,
    llm_model: str,
    provider: str
) -> LLMJudgeMetrics:
    """Evaluate all results using LLM judge."""

    metrics = LLMJudgeMetrics()
    metrics.total_tasks = len(results)

    all_scores = []
    category_scores = defaultdict(list)
    numerical_errors = []

    for task_id, task_result in results.items():
        question = task_result.get("question", "")
        predicted_answer = task_result.get("predicted_answer", "")
        ground_truth = task_result.get("ground_truth")

        if ground_truth is None:
            logger.warning(f"Task {task_id} has no ground truth, skipping")
            continue

        logger.info(f"Evaluating task {task_id}...")

        # Call LLM judge
        judge_result = call_llm_judge(
            llm_client, llm_model, provider,
            question, ground_truth, predicted_answer
        )

        score = judge_result.score
        all_scores.append(score)
        category_scores[judge_result.category].append(score)

        # Categorize results
        if score >= 0.95:
            metrics.perfect_answers += 1
        elif score >= 0.70:
            metrics.partial_credit_answers += 1
        else:
            metrics.incorrect_answers += 1

        # Track category
        if "numerical" in judge_result.category:
            metrics.numerical_tasks += 1
            if judge_result.numerical_error_pct is not None:
                numerical_errors.append(judge_result.numerical_error_pct)

                # Error band distribution
                err = judge_result.numerical_error_pct
                if err <= 10:
                    metrics.within_10pct += 1
                elif err <= 20:
                    metrics.within_20pct += 1
                elif err <= 30:
                    metrics.within_30pct += 1
                else:
                    metrics.beyond_30pct += 1

        if judge_result.category == "counting":
            metrics.counting_tasks += 1

        if judge_result.category == "combined" or (judge_result.whitelist_total and judge_result.whitelist_total > 0):
            metrics.whitelist_tasks += 1

        if judge_result.blacklist_violated:
            metrics.blacklist_violations += 1

        # Store detailed result
        metrics.task_results.append({
            "task_id": task_id,
            "question": question[:100] + "..." if len(question) > 100 else question,
            "predicted_answer": predicted_answer[:100] + "..." if predicted_answer and len(predicted_answer) > 100 else predicted_answer,
            "ground_truth": str(ground_truth)[:100] + "..." if len(str(ground_truth)) > 100 else str(ground_truth),
            "score": score,
            "reasoning": judge_result.reasoning,
            "category": judge_result.category,
            "numerical_error_pct": judge_result.numerical_error_pct,
            "whitelist_matches": judge_result.whitelist_matches,
            "whitelist_total": judge_result.whitelist_total,
            "blacklist_violated": judge_result.blacklist_violated
        })

    # Compute final metrics
    if all_scores:
        metrics.answer_accuracy = sum(all_scores) / len(all_scores) * 100
        # Keep compatibility with evaluate.py convention.
        metrics.answer_accuracy_with_instruction = metrics.answer_accuracy * 0.95
        metrics.answer_accuracy_binary = sum(1 for s in all_scores if s >= 0.95) / len(all_scores) * 100
        metrics.answer_accuracy_threshold_70 = sum(s for s in all_scores if s >= 0.70) / len(all_scores) * 100

    if category_scores["numerical"]:
        metrics.avg_score_numerical = sum(category_scores["numerical"]) / len(category_scores["numerical"])
    if category_scores["counting"]:
        metrics.avg_score_counting = sum(category_scores["counting"]) / len(category_scores["counting"])
    if category_scores["combined"]:
        metrics.avg_score_combined = sum(category_scores["combined"]) / len(category_scores["combined"])

    if numerical_errors:
        metrics.avg_numerical_error_pct = sum(numerical_errors) / len(numerical_errors)

    return metrics


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation with Partial Scoring")
    parser.add_argument("--results", required=True, help="Path to inference results JSON")
    parser.add_argument("--output", default=None, help="Output path for evaluation results")
    parser.add_argument("--llm_provider", default="openai", choices=["openai", "anthropic", "local"],
                       help="LLM provider for judge")
    parser.add_argument("--llm_model", default="claude-sonnet-4.5",
                       help="LLM model for judge")
    parser.add_argument("--api_key", default=None, help="API key for LLM provider")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of tasks to evaluate (for testing)")

    args = parser.parse_args()

    # Load results
    logger.info(f"Loading inference results from {args.results}")
    with open(args.results) as f:
        results = json.load(f)

    if args.limit:
        results = dict(list(results.items())[:args.limit])
        logger.info(f"Limited to {args.limit} tasks for evaluation")

    logger.info(f"Loaded {len(results)} task results")

    # Create LLM judge client
    logger.info(f"Creating LLM judge client: {args.llm_provider}/{args.llm_model}")
    llm_client, llm_model = create_llm_judge_client(
        provider=args.llm_provider,
        model=args.llm_model,
        api_key=args.api_key
    )

    # Run evaluation
    logger.info("Starting LLM-as-judge evaluation...")
    metrics = evaluate_with_llm_judge(results, llm_client, llm_model, args.llm_provider)

    # Print summary
    print("\n" + metrics.summary())

    # Save results
    if args.output is None:
        args.output = Path(args.results).parent / "llm_judge_evaluation.json"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save full metrics
    with open(output_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info(f"Detailed results saved to {output_path}")

    # Save summary report
    summary_path = output_path.with_suffix(".txt")
    with open(summary_path, "w") as f:
        f.write(metrics.summary())
    logger.info(f"Summary report saved to {summary_path}")

    # Save CSV for easy analysis
    csv_path = output_path.with_suffix(".csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task_id", "score", "category", "numerical_error_pct",
            "whitelist_matches", "whitelist_total", "blacklist_violated", "reasoning"
        ], extrasaction="ignore")
        writer.writeheader()
        for result in metrics.task_results:
            writer.writerow(result)
    logger.info(f"CSV results saved to {csv_path}")


if __name__ == "__main__":
    main()
