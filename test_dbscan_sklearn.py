import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from clustbench import load_dataset
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score
)

metrics = ["euclidean", "manhattan", "minkowski", "cosine"]
results = []

def evaluate_sklearn(X, labels, y):
    mask = labels != -1
    if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
        return None
    
    return {
        "ARI": adjusted_rand_score(y[mask], labels[mask]),
        "NMI": normalized_mutual_info_score(y[mask], labels[mask]),
        "Silhouette": silhouette_score(X[mask], labels[mask]),
        "DaviesBouldin": davies_bouldin_score(X[mask], labels[mask])
    }

# -------------------- Iris --------------------
iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target

for metric in metrics:
    dbs = DBSCAN(eps=0.5, min_samples=5, metric=metric)
    labels = dbs.fit_predict(X)
    scores = evaluate_sklearn(X, labels, y)

    if scores:
        scores.update({"dataset": "iris", "metric": metric})
        results.append(scores)

# -------------------- Benchmark Gagolewski --------------------
benchmark_sets = [
    ("wut", "s3"),
    ("sipu", "fuzzyx"),
    ("uci", "wine"),
]

for battery, dataset in benchmark_sets:
    b = load_dataset(battery, dataset)
    X = StandardScaler().fit_transform(b.data)
    y = b.labels[0]

    for metric in metrics:
        dbs = DBSCAN(eps=0.5, min_samples=5, metric=metric)
        labels = dbs.fit_predict(X)
        scores = evaluate_sklearn(X, labels, y)

        if scores:
            scores.update({"dataset": f"{battery}_{dataset}", "metric": metric})
            results.append(scores)

df = pd.DataFrame(results)
df.to_csv("sklearn_results.csv", index=False)
print(df)
