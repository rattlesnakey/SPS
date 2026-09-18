import argparse
from collections import defaultdict
from pathlib import Path
from sps.evaluation import score_response
from sps.io import read_jsonl, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rows = read_jsonl(args.predictions)
    grouped = defaultdict(list)
    details = []
    for row in rows:
        correct = bool(score_response(row, row["response"]))
        grouped[int(row["record_index"])].append(correct)
        details.append({
            "record_index": int(row["record_index"]),
            "sample_index": int(row["sample_index"]),
            "correct": correct,
        })
    num_generations = sum(len(v) for v in grouped.values())
    correct_generations = sum(sum(int(x) for x in v) for v in grouped.values())
    pass1 = correct_generations / max(num_generations, 1)
    passk = sum(int(any(v)) for v in grouped.values()) / max(len(grouped), 1)
    output = Path(args.output) if args.output else Path(args.predictions).with_suffix(".metrics.json")
    write_json(output, {
        "num_problems": len(grouped),
        "num_generations": num_generations,
        "pass_at_1": pass1,
        "pass_at_k": passk,
        "details": details,
    })


if __name__ == "__main__":
    main()
