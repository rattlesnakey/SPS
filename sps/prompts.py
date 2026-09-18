from typing import Any, Mapping


def math_user_prompt(question: str) -> str:
    return f"{question.strip()}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def benchmark_name(record: Mapping[str, Any]) -> str:
    raw = str(record.get("benchmark") or record.get("data_source") or "").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if "gpqa" in raw:
        return "gpqa_diamond"
    if "strategyqa" in raw:
        return "strategyqa"
    if "livecodebench" in raw:
        return "livecodebench"
    return "math"


def build_user_prompt(record: Mapping[str, Any]) -> str:
    benchmark = benchmark_name(record)
    question = str(record["question"]).strip()
    if benchmark == "gpqa_diamond":
        return (
            "Please solve the following multiple-choice question. Think step by "
            "step, then end your response with a JSON object whose answer field "
            "contains only the choice letter, for example {\"answer\": \"C\"}.\n\n"
            f"{question}"
        )
    if benchmark == "strategyqa":
        return (
            "Answer the following StrategyQA question using the supplied facts and "
            "your reasoning. Think step by step, then end your response with exactly "
            "one JSON object: {\"answer\": \"Yes\"} or {\"answer\": \"No\"}.\n\n"
            f"{question}"
        )
    if benchmark == "livecodebench":
        prompt = (
            "You will be given a question (problem specification) and will generate "
            "a correct Python program that matches the specification and passes all "
            f"tests.\n\nQuestion: {question}\n\n"
        )
        starter_code = str(record.get("starter_code") or "")
        if starter_code:
            prompt += (
                "Use the following starter code to write the solution and enclose "
                "your code within Python delimiters.\n"
                f"`python\n{starter_code}\n`\n\n"
            )
        else:
            prompt += (
                "Read the inputs from stdin, solve the problem, and write the answer "
                "to stdout (do not directly test on the sample inputs). Enclose the "
                "complete program within delimiters as follows.\n"
                "`python\n# YOUR CODE HERE\n`\n\n"
            )
        return prompt
    return math_user_prompt(question)


def candidate_annotation_prompt(problem: str, reasoning_prefix: str, candidate_step: str) -> str:
    return f'''Problem
{problem}

Reasoning Prefix
{reasoning_prefix}

Candidate Step
{candidate_step}

Evaluation Criteria

1. Validity

- valid: The step is mathematically and logically sound.
- minor_issue: The core reasoning is sound, but there is a small imprecision, omission, or presentation issue.
- invalid: The step contains a substantive mathematical or logical error, contradicts the prefix, is irrelevant, or is not interpretable.

2. Local Progress

- substantial: The step makes clear progress by deriving a useful result, resolving an uncertainty, introducing a productive strategy, correcting an error, or eliminating an important possibility.
- moderate: The step provides some relevant progress, but the advancement is limited or incomplete.
- minimal: The step is mostly repetitive, superficial, or does not materially advance the reasoning.
- harmful: The step moves the reasoning in an incorrect, contradictory, or clearly unproductive direction.

3. Final Label

- positive: The step has validity valid or minor_issue and makes substantial local progress.
- negative: The step is invalid, or its progress is harmful, or it is clearly repetitive, irrelevant, or unproductive.
- neutral: The step does not clearly satisfy either the positive or negative definition, including steps with only moderate or minimal but non-harmful progress.

Justification Requirements

Provide a concise, evidence-based justification of 2--4 sentences. Explain whether the candidate is valid and how it does or does not advance the reasoning. Refer to specific claims, operations, or conclusions in the candidate step. Do not provide a full solution to the problem or lengthy hidden reasoning.

Return only the following JSON object:

{{
  "validity": "valid | minor_issue | invalid",
  "local_progress": "substantial | moderate | minimal | harmful",
  "label": "positive | neutral | negative",
  "confidence": 0.0,
  "justification": "Concise evidence-based explanation."
}}'''
