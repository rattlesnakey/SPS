import json
import re
import subprocess
import tempfile
from pathlib import Path
from .prompts import benchmark_name


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]*)\}", text, re.DOTALL)
    return matches[-1].strip() if matches else ""


def extract_json_answer(text: str) -> str:
    matches = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    for raw in reversed(matches):
        try:
            obj = json.loads(raw)
            if "answer" in obj:
                return str(obj["answer"]).strip()
        except json.JSONDecodeError:
            pass
    return ""


def extract_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"`python\s*(.*?)`", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def normalize_answer(x: str) -> str:
    return re.sub(r"\s+", "", str(x)).lower()


def run_code_tests(code: str, tests) -> bool:
    if not tests:
        return False
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "main.py"
        path.write_text(code, encoding="utf-8")
        for test in tests:
            inp = str(test.get("input", ""))
            expected = str(test.get("output", "")).strip()
            proc = subprocess.run(
                ["python", str(path)],
                input=inp,
                text=True,
                capture_output=True,
                timeout=float(test.get("timeout", 10)),
            )
            if proc.returncode != 0 or proc.stdout.strip() != expected:
                return False
    return True


def score_response(record: dict, response: str) -> bool:
    bench = benchmark_name(record)
    if bench == "gpqa_diamond":
        return normalize_answer(extract_json_answer(response)) == normalize_answer(record.get("reference_answer", ""))
    if bench == "strategyqa":
        return normalize_answer(extract_json_answer(response)) == normalize_answer(record.get("reference_answer", ""))
    if bench == "livecodebench":
        tests = record.get("tests") or record.get("test_cases") or record.get("input_output")
        return run_code_tests(extract_code(response), tests)
    return normalize_answer(extract_boxed(response)) == normalize_answer(record.get("reference_answer", ""))
