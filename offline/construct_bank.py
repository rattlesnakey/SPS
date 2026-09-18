import argparse
import csv
from collections import defaultdict
import joblib
import numpy as np
from sps.bank import save_direction_bank
from sps.clustering import fit_kmeans, fit_pca, marginal_reduction_curve, unit_vector
from sps.config import load_config, model_config, output_dir
from sps.io import read_jsonl, write_json
from sps.modeling import candidate_representation, load_model, load_tokenizer, prefix_state


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sps.yaml")
    p.add_argument("--model", required=True)
    return p.parse_args()


def write_curve(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    mc = model_config(cfg, args.model)
    root = output_dir(cfg, args.model)
    rows = read_jsonl(root / "candidate_annotations.jsonl")
    grouped = defaultdict(list)
    for row in rows:
        if row["label"] in {"positive", "negative"}:
            grouped[int(row["prefix_id"])].append(row)
    tokenizer = load_tokenizer(mc["name"])
    model = load_model(mc["name"], mc["dtype"])
    selected_layer = int(mc["selected_layer"])
    prefix_ids = []
    states = []
    directions = []
    direction_prefix_ids = []
    usable = 0
    for pid, items in grouped.items():
        positive = [r for r in items if r["label"] == "positive"]
        negative = [r for r in items if r["label"] == "negative"]
        if not positive or not negative:
            continue
        full_prefix = items[0]["full_prefix"]
        state = prefix_state(tokenizer, model, full_prefix, selected_layer).numpy()
        neg_repr = np.stack([
            candidate_representation(tokenizer, model, full_prefix, r["candidate_step"], selected_layer).numpy()
            for r in negative
        ])
        neg_centroid = neg_repr.mean(axis=0)
        prefix_ids.append(pid)
        states.append(state)
        for r in positive:
            pos = candidate_representation(tokenizer, model, full_prefix, r["candidate_step"], selected_layer).numpy()
            directions.append(pos - neg_centroid)
            direction_prefix_ids.append(pid)
        usable += 1
    states = np.asarray(states, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    prefix_ids = np.asarray(prefix_ids, dtype=np.int64)
    direction_prefix_ids = np.asarray(direction_prefix_ids, dtype=np.int64)
    np.savez_compressed(
        root / "prefix_directions.npz",
        prefix_ids=prefix_ids,
        prefix_states=states,
        directions=directions,
        direction_prefix_ids=direction_prefix_ids,
    )
    pca_dim = int(cfg["representation"]["pca_dim"])
    seed = int(cfg["runtime"]["seed"])
    cc = cfg["clustering"]
    num_regions = int(cc["state_regions_by_model"][args.model])
    state_pca, state_z = fit_pca(states, pca_dim, seed)
    state_curve = marginal_reduction_curve(state_z, cc["state_k_values"], seed)
    write_curve(root / "state_cluster_marginal_reduction.csv", state_curve)
    state_km, state_labels = fit_kmeans(state_z, num_regions, seed)
    joblib.dump(state_pca, root / "state_pca.joblib")
    joblib.dump(state_km, root / "state_kmeans.joblib")
    region_centroids = []
    for region in range(num_regions):
        region_centroids.append(states[state_labels == region].mean(axis=0))
    region_centroids = np.stack(region_centroids).astype(np.float32)
    pid_to_region = {int(pid): int(label) for pid, label in zip(prefix_ids, state_labels)}
    cluster_counts = [int(x) for x in cc["direction_clusters_by_model"][args.model]]
    if len(cluster_counts) != num_regions:
        raise ValueError(f"Direction cluster count list has {len(cluster_counts)} entries for {num_regions} regions")
    vectors_by_region = []
    region_summaries = []
    for region in range(num_regions):
        mask = np.asarray([pid_to_region[int(pid)] == region for pid in direction_prefix_ids], dtype=bool)
        x = directions[mask]
        if len(x) == 0:
            vectors_by_region.append(np.empty((0, directions.shape[1]), dtype=np.float32))
            region_summaries.append({"region": region, "num_directions": 0, "num_clusters": 0})
            continue
        direction_pca, z = fit_pca(x, pca_dim, seed)
        curve = marginal_reduction_curve(z, cc["direction_k_values"], seed)
        write_curve(root / f"direction_cluster_marginal_reduction_region_{region}.csv", curve)
        k = min(cluster_counts[region], len(x))
        direction_km, labels = fit_kmeans(z, k, seed)
        vectors = np.stack([
            unit_vector(x[labels == cluster].mean(axis=0))
            for cluster in range(k)
        ]).astype(np.float32)
        vectors_by_region.append(vectors)
        joblib.dump(direction_pca, root / f"direction_pca_region_{region}.joblib")
        joblib.dump(direction_km, root / f"direction_kmeans_region_{region}.joblib")
        region_summaries.append({
            "region": region,
            "num_directions": int(len(x)),
            "num_clusters": int(k),
            "pca_dim": int(z.shape[1]),
            "retained_variance": float(direction_pca.explained_variance_ratio_.sum()),
        })
    entropies = np.load(root / "boundary_entropies.npy")
    metadata = {
        "model": mc["name"],
        "selected_layer": selected_layer,
        "num_state_regions": num_regions,
        "num_usable_prefixes": usable,
        "pca_dim": pca_dim,
        "state_retained_variance": float(state_pca.explained_variance_ratio_.sum()),
        "regions": region_summaries,
    }
    save_direction_bank(root / "direction_bank.npz", region_centroids, vectors_by_region, entropies, metadata)
    write_json(root / "direction_bank_summary.json", metadata)


if __name__ == "__main__":
    main()
