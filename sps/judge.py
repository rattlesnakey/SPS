import json
import os
import re
from openai import OpenAI
from .prompts import candidate_annotation_prompt


class CandidateJudge:
    def __init__(self, model: str = "gpt-5.5", reasoning_effort: str = "medium"):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.reasoning_effort = reasoning_effort

    def annotate(self, problem: str, prefix: str, candidate: str) -> dict:
        prompt = candidate_annotation_prompt(problem, prefix, candidate)
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            reasoning={"effort": self.reasoning_effort},
        )
        text = response.output_text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Judge did not return JSON: {text[:300]}")
        obj = json.loads(match.group(0))
        label = str(obj.get("label", "")).strip().lower()
        validity = str(obj.get("validity", "")).strip().lower()
        local_progress = str(obj.get("local_progress", "")).strip().lower()
        if label not in {"positive", "neutral", "negative"}:
            raise ValueError(f"Invalid judge label: {label}")
        if validity not in {"valid", "minor_issue", "invalid"}:
            raise ValueError(f"Invalid validity label: {validity}")
        if local_progress not in {"substantial", "moderate", "minimal", "harmful"}:
            raise ValueError(f"Invalid progress label: {local_progress}")
        return {
            "validity": validity,
            "local_progress": local_progress,
            "label": label,
            "confidence": float(obj.get("confidence", 0.0)),
            "justification": str(obj.get("justification", "")).strip(),
        }
