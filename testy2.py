#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score
)
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances



# =========================================================
# TWOJA WŁASNA IMPLEMENTACJA DBSCAN
# =========================================================

def region_query_distmatrix(D, point_idx, eps):
    """ Zapytanie sąsiedztwa korzystając Z GOTOWEJ MACIERZY ODLEGŁOŚCI """
    return np.where(D[point_idx] <= eps)[0]


def expand_cluster_distmatrix(D, labels, point_idx, cluster_id, eps, min_pts):
    neighbors = region_query_distmatrix(D, point_idx, eps)

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
            new_neighbors = region_query_distmatrix(D, n, eps)
            if len(new_neighbors) >= min_pts:
                neighbors = np.concatenate((neighbors, new_neighbors))

        i += 1

    return True


def my_dbscan(D, eps=0.5, min_pts=5):
    """ Twoja własna implementacja DBSCAN działająca na macierzy odległości """
    n = D.shape[0]
    labels = np.zeros(n, dtype=int)
    cluster_id = 0

    for point_idx in range(n):
        if labels[point_idx] != 0:
            continue

        if expand_cluster_distmatrix(D, labels, point_idx, cluster_id + 1, eps, min_pts):
            cluster_id += 1

    return labels



# =========================================================
# ŁADOWANIE BENCHMARKÓW (auto-dobór labels*.gz)
# =========================================================

def load_benchmark(dataset_name, base_path):
    """
    Ładuje zestawy Gagolewskiego.
    Wybiera automatycznie labels0.gz, labels1.gz, labels2.gz itd.
    """

    data_file = os.path.join(base_path, f"{dataset_name}.data.gz")

    # ---- znajdź labelsX.gz ----
    labels_candidates = sorted([
        f for f in os.listdir(base_path)
        if f.startswith(dataset_name + ".labels")
    ])

    if len(labels_candidates) == 0:
        raise FileNotFoundError(f"Brak plików labels* dla {dataset_name} w {base_path}")

    labels_file = os.path.join(base_path, labels_candidates[0])

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Brak pliku: {data_file}")

    X = np.loadtxt(data_file)
    y = np.loadtxt(labels_file, dtype=int)

    return X, y



# =========================================================
# MACIERZE ODLEGŁOŚCI
# =========================================================

def pairwise_distance_matrix(X, metric):
    if metric == "euclidean":
        return squareform(pdist(X, metric="euclidean"))
    if metric == "manhattan":
        return squareform(pdist(X, metric="cityblock"))
    if metric == "minkowski":
        return squareform(pdist(X, metric="minkowski", p=3))
    if metric == "cosine":
        return squareform(pdist(X, metric="cosine"))
    if metric == "correlation":
        return squareform(pdist(X, metric="correlation"))
    if metric == "canberra":
        return squareform(pdist(X, metric="canberra"))
    if metric == "braycurtis":
        return squareform(pdist(X, metric="braycurtis"))
    if metric == "hamming":
        return squareform(pdist(X, metric="hamming"))

    if metric == "mahalanobis":
        VI = np.linalg.inv(np.cov(X.T))
        return squareform(pdist(X, metric="mahalanobis", VI=VI))

    raise ValueError("Nieznana metryka odległości: " + metric)



# =========================================================
# LISTA METRYK
# =========================================================

metrics = [
    "euclidean", "manhattan", "minkowski", "cosine",
    "mahalanobis", "canberra", "hamming", "braycurtis", "correlation"
]



# =========================================================
# GŁÓWNY PROGRAM
# =========================================================

def main():

    results = []

    # ---------------------------------------------
    # Zbiory danych
    # ---------------------------------------------
    iris = load_iris()

    datasets = {
        "iris": ("iris", "sklearn"),

        "wine": ("wine", "uci"),
        "s3": ("s3", "sipu"),
        "fuzzyx": ("fuzzyx", "graves")
    }

    base_paths = {
        "uci": "clustering-data-v1-master/uci",
        "graves": "clustering-data-v1-master/graves",
        "sipu": "clustering-data-v1-master/sipu"
    }

    # ---------------------------------------------
    # Pętla po zbiorach
    # ---------------------------------------------
    for name, (dataset_name, source) in datasets.items():

        print(f"\n=== Dataset: {name} ===")

        # ---- wczytywanie danych ----
        if source == "sklearn":
            X = iris.data
            y_true = iris.target
        else:
            folder = base_paths[source]
            X, y_true = load_benchmark(dataset_name, folder)

        # standaryzacja (opcjonalna, ale pomaga DBSCANowi)
        X = StandardScaler().fit_transform(X)

        # ---- test metryk ----
        for metric in metrics:

            print(f"  > Metryka: {metric}")

            try:
                D = pairwise_distance_matrix(X, metric)
            except Exception as e:
                print(f"    Błąd odległości {metric}: {e}")
                continue

            # ---------------- MY DBSCAN ----------------
            try:
                y_my = my_dbscan(D, eps=0.5, min_pts=5)
            except Exception as e:
                print("    Błąd w my_dbscan:", e)
                y_my = np.full(len(X), -1)

            # ---------------- SKLEARN DBSCAN ---------
            try:
                sk = DBSCAN(eps=0.5, min_samples=5, metric="precomputed")
                y_sk = sk.fit_predict(D)
            except Exception as e:
                print("    Błąd sklearn:", e)
                y_sk = np.full(len(X), -1)

            # ---------------- EWALUACJA --------------
            for version, y_pred in [
                ("my_dbscan", y_my),
                ("sklearn_dbscan", y_sk)
            ]:

                if len(set(y_pred)) > 1:
                    silhouette = silhouette_score(X, y_pred)
                    dbi = davies_bouldin_score(X, y_pred)
                else:
                    silhouette = np.nan
                    dbi = np.nan

                results.append({
                    "dataset": name,
                    "metric": metric,
                    "version": version,
                    "ARI": adjusted_rand_score(y_true, y_pred),
                    "NMI": normalized_mutual_info_score(y_true, y_pred),
                    "Silhouette": silhouette,
                    "DaviesBouldin": dbi
                })

    # ----------------------------------------------------------
    # ZAPIS WYNIKÓW
    # ----------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv("wyniki_dbscan_porownanie.csv", index=False)
    print("\nZapisano wyniki do wyniki_dbscan_porownanie.csv")

    # ----------------------------------------------------------
    # WYKRESY
    # ----------------------------------------------------------
    for m in ["ARI", "NMI", "Silhouette", "DaviesBouldin"]:
        pivot = df.pivot_table(
            values=m,
            index=["dataset", "metric"],
            columns="version"
        )

        pivot.plot(kind="bar", figsize=(12, 6))
        plt.title(f"Porównanie: {m}")
        plt.tight_layout()
        plt.savefig(f"wykres_{m}.png")
        plt.close()

    print("Wygenerowano wykresy wykres_*.png")



# =========================================================
# START
# =========================================================
if __name__ == "__main__":
    main()
