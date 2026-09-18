# SPS Prompt Templates

## Default Math Generation Prompt

```text
<|im_start|>user
{input}
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
```

The implementation constructs the user content and then applies the Qwen3 chat template with thinking mode enabled.

## GPQA-Diamond Evaluation Prompt

```text
Please solve the following multiple-choice question. Think step by step, then end your response with a JSON object whose answer field contains only the choice letter, for example {"answer": "C"}.

{question}
```

## StrategyQA Evaluation Prompt

```text
Answer the following StrategyQA question using the supplied facts and your reasoning. Think step by step, then end your response with exactly one JSON object: {"answer": "Yes"} or {"answer": "No"}.

{question}
```

## LiveCodeBench Evaluation Prompt with Starter Code

```text
You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Question: {question}

Use the following starter code to write the solution and enclose your code within Python delimiters.
`python
{starter_code}
`
```

## LiveCodeBench Evaluation Prompt without Starter Code

```text
You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Question: {question}

Read the inputs from stdin, solve the problem, and write the answer to stdout (do not directly test on the sample inputs). Enclose the complete program within delimiters as follows.
`python
# YOUR CODE HERE
`
```

## LLM-as-Judge Candidate Annotation Prompt

```text
Problem
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

{
  "validity": "valid | minor_issue | invalid",
  "local_progress": "substantial | moderate | minimal | harmful",
  "label": "positive | neutral | negative",
  "confidence": 0.0,
  "justification": "Concise evidence-based explanation."
}
```

Candidate annotation uses GPT-5.5 through the `gpt-5.5` alias with reasoning effort set to `medium`.
