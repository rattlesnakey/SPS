import argparse
import gc
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from vllm import LLM, SamplingParams
from sps.config import load_config, model_config, output_dir
from sps.dedup import CandidateDeduplicator, load_thresholds
from sps.io import read_jsonl, write_json, write_jsonl
from sps.judge import CandidateJudge


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sps.yaml")
    p.add_argument("--model", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    mc = model_config(cfg, args.model)
    root = output_dir(cfg, args.model)
    prefixes = read_jsonl(root / "uncertain_prefixes.jsonl")
    llm = LLM(
        model=mc["name"],
        dtype=mc["dtype"],
        tensor_parallel_size=int(mc["tensor_parallel_size"]),
        trust_remote_code=True,
    )
    c = cfg["candidate"]
    g = cfg["generation"]
    params = SamplingParams(
        n=int(c["num_candidates"]),
        temperature=float(c["temperature"]),
        top_p=float(g["top_p"]),
        top_k=int(g["top_k"]),
        max_tokens=int(c["max_new_tokens"]),
        stop=[cfg["runtime"]["step_delimiter"]],
        seed=int(cfg["runtime"]["seed"]),
    )
    generated = llm.generate([r["full_prefix"] for r in prefixes], params)
    raw_by_prefix = []
    for prefix, result in zip(prefixes, generated):
        raw_by_prefix.append((prefix, [o.text for o in result.outputs]))
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dc = cfg["dedup"]
    if bool(dc["enabled"]):
        thresholds = load_thresholds(dc["thresholds_path"])
        deduplicator = CandidateDeduplicator(
            dc["embedding_model"],
            thresholds["lexical_threshold"],
            thresholds["semantic_threshold"],
            int(dc["batch_size"]),
        )
    else:
        thresholds = None
        deduplicator = None
    candidate_rows = []
    before_total = 0
    after_total = 0
    for prefix, raw in raw_by_prefix:
        before_total += len(raw)
        unique = deduplicator.deduplicate(raw) if deduplicator is not None else [str(x).strip() for x in raw if str(x).strip()]
        after_total += len(unique)
        for candidate_index, text in enumerate(unique):
            candidate_rows.append({
                "prefix_id": prefix["prefix_id"],
                "candidate_index": candidate_index,
                "problem": prefix["problem"],
                "rendered_prompt": prefix["rendered_prompt"],
                "reasoning_prefix": prefix["reasoning_prefix"],
                "full_prefix": prefix["full_prefix"],
                "boundary_entropy": prefix["boundary_entropy"],
                "candidate_step": text,
                "raw_candidate_count": len(raw),
                "deduplicated_candidate_count": len(unique),
            })
    write_jsonl(root / "candidate_steps_deduplicated.jsonl", candidate_rows)
    if deduplicator is not None:
        del deduplicator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    judge_cfg = cfg["judge"]
    judge_model = str(judge_cfg.get("model", "gpt-5.5"))
    reasoning_effort = str(judge_cfg.get("reasoning_effort", "medium"))
    judge = CandidateJudge(judge_model, reasoning_effort)
    def annotate(row):
        result = judge.annotate(row["problem"], row["reasoning_prefix"], row["candidate_step"])
        return {**row, **result}
    workers = int(judge_cfg["workers"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        annotations = list(pool.map(annotate, candidate_rows))
    write_jsonl(root / "candidate_annotations.jsonl", annotations)
    write_json(root / "annotation_summary.json", {
        "judge_model": judge_model,
        "reasoning_effort": reasoning_effort,
        "num_uncertain_prefixes": len(prefixes),
        "num_candidates_before_dedup": before_total,
        "num_candidates_after_dedup": after_total,
        "num_annotated_candidates": len(annotations),
        "dedup_thresholds": thresholds,
        "label_counts": {
            label: sum(1 for r in annotations if r["label"] == label)
            for label in ("positive", "neutral", "negative")
        },
    })


if __name__ == "__main__":
    main()
