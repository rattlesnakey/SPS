import argparse
import gc
import random
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sps.config import load_config, model_config, output_dir
from sps.io import write_json, write_jsonl
from sps.modeling import boundary_entropy, load_model, render_chat
from sps.prompts import math_user_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sps.yaml")
    p.add_argument("--model", required=True)
    return p.parse_args()


def extract_problem(row: dict) -> str:
    for key in ("problem", "question"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    if isinstance(prompt, list):
        for item in reversed(prompt):
            if isinstance(item, dict) and str(item.get("content", "")).strip():
                return str(item["content"]).strip()
    raise ValueError("Cannot extract problem text")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    mc = model_config(cfg, args.model)
    root = output_dir(cfg, args.model)
    root.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["runtime"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    dataset_cfg = cfg["calibration"]
    ds = load_dataset(dataset_cfg["dataset"], split=dataset_cfg["split"])
    sample_size = int(dataset_cfg["sample_size"])
    indices = random.sample(range(len(ds)), sample_size)
    tokenizer = AutoTokenizer.from_pretrained(mc["name"], trust_remote_code=True)
    calibration = []
    rendered_prompts = []
    for calibration_id, idx in enumerate(indices):
        problem = extract_problem(ds[idx])
        user_prompt = math_user_prompt(problem)
        rendered = render_chat(tokenizer, user_prompt, mc["enable_thinking"])
        calibration.append({
            "calibration_id": calibration_id,
            "source_index": int(idx),
            "problem": problem,
            "user_prompt": user_prompt,
            "rendered_prompt": rendered,
        })
        rendered_prompts.append(rendered)
    write_jsonl(root / "calibration_prompts.jsonl", calibration)
    llm = LLM(
        model=mc["name"],
        dtype=mc["dtype"],
        tensor_parallel_size=int(mc["tensor_parallel_size"]),
        trust_remote_code=True,
    )
    g = cfg["generation"]
    sampling = SamplingParams(
        temperature=float(g["temperature"]),
        top_p=float(g["top_p"]),
        top_k=int(g["top_k"]),
        max_tokens=int(g["max_new_tokens"]),
        seed=seed,
    )
    outputs = llm.generate(rendered_prompts, sampling)
    rollouts = []
    for row, result in zip(calibration, outputs):
        rollouts.append({**row, "rollout": result.outputs[0].text})
    write_jsonl(root / "calibration_rollouts.jsonl", rollouts)
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = load_model(mc["name"], mc["dtype"])
    delimiter = cfg["runtime"]["step_delimiter"]
    prefixes = []
    entropies = []
    prefix_id = 0
    for row in rollouts:
        raw_steps = row["rollout"].split(delimiter)
        steps = [s for s in raw_steps if s.strip()]
        for step_index, step in enumerate(steps):
            reasoning_prefix = delimiter.join(steps[:step_index])
            if reasoning_prefix:
                reasoning_prefix = reasoning_prefix + delimiter
            full_prefix = row["rendered_prompt"] + reasoning_prefix
            h = boundary_entropy(tokenizer, model, full_prefix)
            prefixes.append({
                "prefix_id": prefix_id,
                "calibration_id": row["calibration_id"],
                "step_index": step_index,
                "problem": row["problem"],
                "user_prompt": row["user_prompt"],
                "rendered_prompt": row["rendered_prompt"],
                "reasoning_prefix": reasoning_prefix,
                "full_prefix": full_prefix,
                "target_step": step,
                "boundary_entropy": h,
            })
            entropies.append(h)
            prefix_id += 1
    q = float(cfg["uncertainty"]["entropy_quantile"])
    threshold = float(np.quantile(np.asarray(entropies, dtype=np.float64), q))
    uncertain = [row for row in prefixes if float(row["boundary_entropy"]) > threshold]
    write_jsonl(root / "prefix_steps.jsonl", prefixes)
    write_jsonl(root / "uncertain_prefixes.jsonl", uncertain)
    np.save(root / "boundary_entropies.npy", np.asarray(entropies, dtype=np.float32))
    write_json(root / "uncertainty_gate.json", {
        "entropy_quantile": q,
        "threshold": threshold,
        "num_prefixes": len(prefixes),
        "num_uncertain_prefixes": len(uncertain),
    })


if __name__ == "__main__":
    main()
