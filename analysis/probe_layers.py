import argparse
import csv
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sps.config import load_config, model_config, output_dir
from sps.io import read_jsonl, write_json
from sps.modeling import candidate_representation, load_model, load_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sps.yaml")
    p.add_argument("--model", required=True)
    return p.parse_args()


def fit_pca_with_target(x_train, x_test, base_dim, target_variance):
    max_components = min(x_train.shape[0] - 1, x_train.shape[1])
    initial = min(int(base_dim), max_components)
    pca = PCA(n_components=initial, random_state=0)
    pca.fit(x_train)
    retained = float(pca.explained_variance_ratio_.sum())
    if retained >= float(target_variance) or initial == max_components:
        return pca, pca.transform(x_train), pca.transform(x_test), retained
    full = PCA(n_components=max_components, svd_solver="full")
    full.fit(x_train)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    required = int(np.searchsorted(cumulative, float(target_variance), side="left") + 1)
    chosen = max(initial, required)
    pca = PCA(n_components=chosen, random_state=0)
    x_train_z = pca.fit_transform(x_train)
    x_test_z = pca.transform(x_test)
    retained = float(pca.explained_variance_ratio_.sum())
    return pca, x_train_z, x_test_z, retained


def main():
    args = parse_args()
    cfg = load_config(args.config)
    mc = model_config(cfg, args.model)
    root = output_dir(cfg, args.model)
    rows = [r for r in read_jsonl(root / "candidate_annotations.jsonl") if r["label"] in {"positive", "negative"}]
    positive = [r for r in rows if r["label"] == "positive"]
    negative = [r for r in rows if r["label"] == "negative"]
    n = int(cfg["probe"]["sample_size"]) // 2
    rng = random.Random(int(cfg["runtime"]["seed"]))
    sample = rng.sample(positive, n) + rng.sample(negative, n)
    rng.shuffle(sample)
    labels = np.asarray([1 if r["label"] == "positive" else 0 for r in sample], dtype=np.int64)
    indices = np.arange(len(sample))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=float(cfg["probe"]["train_fraction"]),
        random_state=int(cfg["runtime"]["seed"]),
        stratify=labels,
    )
    tokenizer = load_tokenizer(mc["name"])
    model = load_model(mc["name"], mc["dtype"])
    results = []
    for depth in cfg["probe"]["depths"]:
        layer = max(1, min(int(mc["layers"]), int(round(int(mc["layers"]) * float(depth) / 100.0))))
        x = np.stack([
            candidate_representation(tokenizer, model, r["full_prefix"], r["candidate_step"], layer).numpy()
            for r in sample
        ])
        pca, x_train, x_test, retained = fit_pca_with_target(
            x[train_idx],
            x[test_idx],
            int(cfg["probe"]["base_pca_dim"]),
            float(cfg["probe"]["min_retained_variance"]),
        )
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(x_train, labels[train_idx])
        scores = clf.decision_function(x_test)
        auc = float(roc_auc_score(labels[test_idx], scores))
        results.append({
            "depth_percent": int(depth),
            "layer": int(layer),
            "auc": auc,
            "pca_dim": int(pca.n_components_),
            "retained_variance": retained,
        })
    csv_path = root / "probe_auc.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    best = max(results, key=lambda x: x["auc"])
    write_json(root / "probe_auc.json", {
        "model": mc["name"],
        "sample_size": len(sample),
        "train_fraction": float(cfg["probe"]["train_fraction"]),
        "best": best,
        "results": results,
    })


if __name__ == "__main__":
    main()
