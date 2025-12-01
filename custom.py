import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from clustbench import load_dataset
from sklearn.datasets import load_iris
import pandas as pd


# ============================
#  IMPLEMENTACJA DBSCAN
# ============================

def region_query(X, point_idx, eps, metric):
    dists = pairwise_distances(X[point_idx].reshape(1, -1), X, metric=metric)[0]
    neighbors = np.where(dists <= eps)[0]
    return neighbors


def expand_cluster(X, labels, point_idx, cluster_id, eps, min_pts, metric):
    neighbors = region_query(X, point_idx, eps, metric)
    if len(neighbors) < min_pts:
        labels[point_idx] = -1  # szum
        return False

    labels[point_idx] = cluster_id
    i = 0

    while i < len(neighbors):
        n = neighbors[i]
        if labels[n] == -1:
            labels[n] = cluster_id
        elif labels[n] == 0:
            labels[n] = cluster_id
            n_neighbors = region_query(X, n, eps, metric)
            if len(n_neighbors) >= min_pts:
                neighbors = np.concatenate((neighbors, n_neighbors))
        i += 1

    return True


def dbscan_custom(X, eps, min_pts, metric):
    labels = np.zeros(X.shape[0], dtype=int)
    cluster_id = 0

    for point_idx in range(X.shape[0]):
        if labels[point_idx] != 0:
            continue
        if expand_cluster(X, labels, point_idx, cluster_id + 1, eps, min_pts, metric):
            cluster_id += 1

    return labels


# ============================
#  DYNAMICZNY EPS
# ============================

def compute_dynamic_eps(X, k=5, metric="euclidean"):
    dist_matrix = pairwise_distances(X, X, metric=metric)
    k_dists = np.partition(dist_matrix, kth=k, axis=1)[:, k]
    return np.mean(k_dists)


# ============================
#  EWALUACJA
# ============================

def evaluate(X, labels, y_true):
    mask = labels != -1
    if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
        return None

    return {
        "ARI": adjusted_rand_score(y_true[mask], labels[mask]),
        "NMI": normalized_mutual_info_score(y_true[mask], labels[mask]),
        "Silhouette": silhouette_score(X[mask], labels[mask]),
        "DaviesBouldin": davies_bouldin_score(X[mask], labels[mask]),
    }


# ============================
#  GŁÓWNY PROGRAM
# ============================

metrics = ["euclidean", "manhattan", "minkowski", "cosine", "mahalanobis"]
benchmark_sets = [
    ("wut", "s3"),
    ("sipu", "fuzzyx"),
    ("uci", "wine"),
]

results = []

for name, X, y_true in [
    ("iris", load_iris().data, load_iris().target)
]:

    # IRIS
    X = StandardScaler().fit_transform(X)

    for metric in metrics:
        eps = compute_dynamic_eps(X, k=5, metric=metric)
        labels = dbscan_custom(X, eps, min_pts=5, metric=metric)
        scores = evaluate(X, labels, y_true)

        if scores:
            scores.update({
                "dataset": name,
                "metric": metric
            })
            results.append(scores)


# BENCHMARKI GAGOLEWSKIEGO
for battery, dataset in benchmark_sets:
    b = load_dataset(battery, dataset)
    X = StandardScaler().fit_transform(b.data)
    y_true = b.labels[0]

    for metric in metrics:
        eps = compute_dynamic_eps(X, k=5, metric=metric)
        labels = dbscan_custom(X, eps, min_pts=5, metric=metric)
        scores = evaluate(X, labels, y_true)

        if scores:
            scores.update({
                "dataset": f"{battery}_{dataset}",
                "metric": metric
            })
            results.append(scores)


# ZAPIS WYNIKÓW
df = pd.DataFrame(results)
df.to_csv("dbscan_custom_results.csv", index=False)

print(df)
