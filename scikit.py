import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score, pairwise_distances
from clustbench import load_dataset
import pandas as pd


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


metrics = ["euclidean", "manhattan", "minkowski", "cosine", "mahalanobis"]
benchmark_sets = [
    ("wut", "s3"),
    ("sipu", "fuzzyx"),
    ("uci", "wine"),
]

results = []

# ============================
#  IRIS
# ============================

X = load_iris().data
y_true = load_iris().target
X = StandardScaler().fit_transform(X)

for metric in metrics:
    eps = compute_dynamic_eps(X, k=5, metric=metric)
    clustering = DBSCAN(eps=eps, min_samples=5, metric=metric).fit(X)
    labels = clustering.labels_
    scores = evaluate(X, labels, y_true)
    if scores:
        scores.update({"dataset": "iris", "metric": metric})
        results.append(scores)


# ============================
#  BENCHMARKI
# ============================

for battery, dataset in benchmark_sets:
    b = load_dataset(battery, dataset)
    X = StandardScaler().fit_transform(b.data)
    y_true = b.labels[0]

    for metric in metrics:
        eps = compute_dynamic_eps(X, k=5, metric=metric)
        clustering = DBSCAN(eps=eps, min_samples=5, metric=metric).fit(X)
        labels = clustering.labels_
        scores = evaluate(X, labels, y_true)
        if scores:
            scores.update({"dataset": f"{battery}_{dataset}", "metric": metric})
            results.append(scores)


df = pd.DataFrame(results)
df.to_csv("dbscan_sklearn_results.csv", index=False)

print(df)
