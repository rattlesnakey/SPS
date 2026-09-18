import argparse
import csv
import random
from pathlib import Path
from sps.io import read_jsonl, write_jsonl, write_json


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--annotations", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("aggregate")
    a.add_argument("--reviews", required=True)
    a.add_argument("--output", required=True)
    return p.parse_args()


def sample_rows(args):
    rows = read_jsonl(args.annotations)
    positive = [r for r in rows if r.get("label") == "positive"]
    negative = [r for r in rows if r.get("label") == "negative"]
    rng = random.Random(args.seed)
    chosen = rng.sample(positive, 100) + rng.sample(negative, 100)
    rng.shuffle(chosen)
    output = []
    for i, row in enumerate(chosen):
        output.append({
            "validation_id": i,
            "problem": row["problem"],
            "reasoning_prefix": row["reasoning_prefix"],
            "candidate_step": row["candidate_step"],
            "judge_label": row["label"],
            "judge_justification": row["justification"],
            "reviewer_1": "",
            "reviewer_2": "",
        })
    write_jsonl(args.output, output)


def aggregate_rows(args):
    rows = read_jsonl(args.reviews)
    subsets = {"positive": [], "negative": [], "overall": rows}
    for row in rows:
        if row["judge_label"] in subsets:
            subsets[row["judge_label"]].append(row)
    summary = {}
    for name, group in subsets.items():
        n = max(len(group), 1)
        r1 = sum(str(r.get("reviewer_1", "")).lower() == "accept" for r in group) / n
        r2 = sum(str(r.get("reviewer_2", "")).lower() == "accept" for r in group) / n
        both = sum(
            str(r.get("reviewer_1", "")).lower() == "accept" and str(r.get("reviewer_2", "")).lower() == "accept"
            for r in group
        ) / n
        summary[name] = {"reviewer_1_accept": r1, "reviewer_2_accept": r2, "both_accept": both, "count": len(group)}
    write_json(args.output, summary)


def main():
    args = parse_args()
    if args.command == "sample":
        sample_rows(args)
    else:
        aggregate_rows(args)


if __name__ == "__main__":
    main()
