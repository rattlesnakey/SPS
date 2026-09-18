import argparse
import json
from pathlib import Path
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sps.config import load_config
from sps.io import read_jsonl, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sps.yaml")
    p.add_argument("--pairs", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dc = cfg["dedup"]
    pairs_path = Path(args.pairs or dc["manual_pairs_path"])
    output_path = Path(args.output or dc["thresholds_path"])
    rows = read_jsonl(pairs_path)
    if not rows:
        raise RuntimeError(f"No manually selected similar pairs found in {pairs_path}")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    model = SentenceTransformer(dc["embedding_model"], trust_remote_code=True)
    texts = []
    lexical = []
    for row in rows:
        a = str(row["candidate_a"]).strip()
        b = str(row["candidate_b"]).strip()
        texts.extend([a, b])
        lexical.append(float(scorer.score(a, b)["rougeL"].fmeasure))
    embeddings = model.encode(
        texts,
        batch_size=int(dc["batch_size"]),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    semantic = []
    for i in range(0, len(embeddings), 2):
        semantic.append(float(np.dot(embeddings[i], embeddings[i + 1])))
    write_json(output_path, {
        "lexical_threshold": float(np.mean(lexical)),
        "semantic_threshold": float(np.mean(semantic)),
        "num_manual_pairs": len(rows),
        "embedding_model": dc["embedding_model"],
    })


if __name__ == "__main__":
    main()
