import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from sps.bank import load_direction_bank
from sps.config import load_config, model_config, output_dir
from sps.io import read_json, read_jsonl, write_jsonl
from sps.modeling import boundary_entropy, load_model, load_tokenizer, prefix_state, render_chat
from sps.prompts import build_user_prompt
from sps.steering import LayerSteering, entropy_adaptive_strength, generate_step, sample_direction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sps.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    mc = model_config(cfg, args.model)
    root = output_dir(cfg, args.model)
    rows = read_jsonl(args.data)
    tokenizer = load_tokenizer(mc["name"])
    model = load_model(mc["name"], mc["dtype"])
    bank = load_direction_bank(root / "direction_bank.npz")
    gate = read_json(root / "uncertainty_gate.json")
    threshold = float(gate["threshold"])
    controller = LayerSteering(model, int(mc["selected_layer"]))
    rng = np.random.default_rng(int(cfg["runtime"]["seed"]))
    out_path = Path(args.output) if args.output else root / f"{Path(args.data).stem}_sps.jsonl"
    records = []
    delimiter = cfg["runtime"]["step_delimiter"]
    max_total = int(cfg["generation"]["max_new_tokens"])
    max_step = int(cfg["candidate"]["max_new_tokens"])
    alpha_min = float(cfg["steering"]["alpha_min"])
    try:
        for record_index, row in enumerate(tqdm(rows)):
            user_prompt = build_user_prompt(row)
            rendered = render_chat(tokenizer, user_prompt, mc["enable_thinking"])
            for sample_index in range(int(args.num_samples)):
                torch.manual_seed(int(cfg["runtime"]["seed"]) + sample_index)
                generated_text = ""
                trace = []
                total_generated = 0
                for step_index in range(10000):
                    full_prefix = rendered + generated_text
                    h = boundary_entropy(tokenizer, model, full_prefix)
                    intervened = h > threshold
                    if intervened:
                        state = prefix_state(tokenizer, model, full_prefix, int(mc["selected_layer"])).numpy()
                        region, direction_index, direction = sample_direction(bank, state, rng)
                        alpha, percentile, threshold_percentile = entropy_adaptive_strength(
                            bank["entropy_reference"], h, threshold, alpha_min
                        )
                        controller.set(direction, alpha)
                    else:
                        region = -1
                        direction_index = -1
                        percentile = 0.0
                        threshold_percentile = 0.0
                        alpha = 0.0
                        controller.clear()
                    input_ids = tokenizer(full_prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
                    remain = max_total - total_generated
                    if remain <= 0:
                        break
                    output_ids, step_text = generate_step(
                        tokenizer,
                        model,
                        input_ids,
                        controller,
                        cfg["generation"],
                        delimiter,
                        min(max_step, remain),
                    )
                    controller.clear()
                    step_tokens = int(output_ids.shape[1] - input_ids.shape[1])
                    total_generated += step_tokens
                    generated_text += step_text
                    trace.append({
                        "step_index": step_index,
                        "boundary_entropy": h,
                        "intervened": intervened,
                        "matched_region": region,
                        "direction_index": direction_index,
                        "entropy_percentile": percentile,
                        "threshold_percentile": threshold_percentile,
                        "steering_strength": alpha,
                        "generated_tokens": step_tokens,
                    })
                    if step_tokens == 0 or total_generated >= max_total:
                        break
                    if tokenizer.eos_token and tokenizer.eos_token in step_text:
                        break
                records.append({
                    **row,
                    "record_index": record_index,
                    "sample_index": sample_index,
                    "user_prompt": user_prompt,
                    "response": generated_text,
                    "trace": trace,
                })
    finally:
        controller.close()
    write_jsonl(out_path, records)


if __name__ == "__main__":
    main()
