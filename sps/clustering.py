import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def fit_pca(x: np.ndarray, dim: int, seed: int):
    n = min(int(dim), int(x.shape[0] - 1), int(x.shape[1]))
    if n < 1:
        raise ValueError("PCA requires at least two samples")
    pca = PCA(n_components=n, random_state=seed)
    z = pca.fit_transform(x)
    return pca, z


def fit_kmeans(x: np.ndarray, k: int, seed: int):
    km = KMeans(n_clusters=int(k), random_state=seed, n_init=20)
    labels = km.fit_predict(x)
    return km, labels


def mean_within_cluster_distance(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    assigned = centers[labels]
    return float(np.linalg.norm(x - assigned, axis=1).mean())


def marginal_reduction_curve(x: np.ndarray, k_values: list[int], seed: int) -> list[dict]:
    valid = [int(k) for k in k_values if int(k) >= 1 and int(k) <= len(x)]
    rows = []
    previous_k = None
    previous_distance = None
    for k in valid:
        km, labels = fit_kmeans(x, k, seed)
        distance = mean_within_cluster_distance(x, labels, km.cluster_centers_)
        if previous_k is None:
            reduction = None
        else:
            reduction = float((previous_distance - distance) / (k - previous_k))
        rows.append({
            "k": k,
            "mean_distance": distance,
            "distance_reduction_per_added_cluster": reduction,
        })
        previous_k = k
        previous_distance = distance
    return rows


def unit_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    if n == 0.0:
        return x
    return x / n
