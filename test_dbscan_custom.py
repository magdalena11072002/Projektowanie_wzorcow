import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from clustbench import load_dataset

from main_custom_dbscan import (
    dbscan_custom,
    compute_dynamic_eps,
    evaluate
)

metrics = ["euclidean", "manhattan", "minkowski", "cosine", "mahalanobis"]
results = []

# -------------------- Iris --------------------
iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target

for metric in metrics:
    eps = compute_dynamic_eps(X, k=5, metric=metric)
    labels = dbscan_custom(X, eps, min_pts=5, metric=metric)
    scores = evaluate(X, labels, y)

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
        eps = compute_dynamic_eps(X, k=5, metric=metric)
        labels = dbscan_custom(X, eps, min_pts=5, metric=metric)
        scores = evaluate(X, labels, y)

        if scores:
            scores.update({"dataset": f"{battery}_{dataset}", "metric": metric})
            results.append(scores)

df = pd.DataFrame(results)
df.to_csv("custom_results.csv", index=False)
print(df)
