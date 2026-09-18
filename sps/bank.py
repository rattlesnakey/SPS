from pathlib import Path
import json
import numpy as np


def save_direction_bank(path: str | Path, region_centroids: np.ndarray, vectors_by_region: list[np.ndarray], entropy_reference: np.ndarray, metadata: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    packed = np.empty(len(vectors_by_region), dtype=object)
    for i, vectors in enumerate(vectors_by_region):
        packed[i] = np.asarray(vectors, dtype=np.float32)
    np.savez_compressed(
        p,
        region_centroids=np.asarray(region_centroids, dtype=np.float32),
        vectors_by_region=packed,
        entropy_reference=np.asarray(entropy_reference, dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata), dtype=object),
    )


def load_direction_bank(path: str | Path) -> dict:
    obj = np.load(path, allow_pickle=True)
    return {
        "region_centroids": np.asarray(obj["state_centroids"], dtype=np.float32),
        "vectors_by_region": [np.asarray(v, dtype=np.float32) for v in obj["vectors_by_region"]],
        "entropy_reference": np.asarray(obj["entropy_reference"], dtype=np.float32),
        "metadata": json.loads(str(obj["metadata"].item())),
    }
