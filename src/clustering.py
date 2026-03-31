"""
clustering.py
-------------
Clustering logic for the three embedding methods:

  • TF-IDF  → K-means (k=10, random_state=42, n_init=10)
  • MiniLM  → HDBSCAN (min_cluster_size=15, min_samples=5)
  • OpenAI  → HDBSCAN (same params)

Noise points (HDBSCAN label -1) are kept in the returned array so that
downstream code can handle them explicitly.
"""

from __future__ import annotations

import numpy as np

# HDBSCAN params shared across semantic methods
HDBSCAN_PARAMS = dict(min_cluster_size=30, min_samples=5)
KMEANS_K = 10


def cluster_kmeans(embeddings: np.ndarray, k: int = KMEANS_K) -> np.ndarray:
    """Cluster TF-IDF/LSA embeddings with K-means.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n, d)
        Pre-computed embedding matrix.
    k : int
        Number of clusters.

    Returns
    -------
    np.ndarray, shape (n,)
        Integer cluster labels in [0, k-1].
    """
    from sklearn.cluster import KMeans

    print(f"\n[K-means] Fitting k={k} …")
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(embeddings)
    _print_cluster_summary("K-means", labels)
    return labels.astype(np.int32)


def cluster_hdbscan(embeddings: np.ndarray, method_name: str = "HDBSCAN") -> np.ndarray:
    """Cluster semantic embeddings with HDBSCAN.

    Uses ``fast-hdbscan`` which is compatible with Apple Silicon (M-series).
    Noise points receive label ``-1``.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n, d)
        Pre-computed embedding matrix.
    method_name : str
        Label printed in progress messages.

    Returns
    -------
    np.ndarray, shape (n,)
        Integer cluster labels; ``-1`` denotes noise.
    """
    try:
        from fast_hdbscan import HDBSCAN
    except ImportError:
        raise ImportError(
            "fast-hdbscan is not installed. Run: pip install fast-hdbscan"
        )

    print(f"\n[{method_name}] Fitting HDBSCAN "
          f"(min_cluster_size={HDBSCAN_PARAMS['min_cluster_size']}, "
          f"min_samples={HDBSCAN_PARAMS['min_samples']}) …")

    hdb = HDBSCAN(**HDBSCAN_PARAMS)
    labels = hdb.fit_predict(embeddings)
    _print_cluster_summary(method_name, labels)
    return labels.astype(np.int32)


# ── internal ──────────────────────────────────────────────────────────────────

def _print_cluster_summary(name: str, labels: np.ndarray) -> None:
    """Print a compact summary of cluster sizes."""
    unique, counts = np.unique(labels, return_counts=True)
    n_noise = int(counts[unique == -1].sum()) if -1 in unique else 0
    n_clusters = int((unique >= 0).sum())
    print(f"  [{name}] {n_clusters} clusters found  |  noise points: {n_noise}")
    if n_clusters <= 20:
        for lbl, cnt in sorted(zip(unique, counts)):
            tag = " (noise)" if lbl == -1 else ""
            print(f"    cluster {lbl:>3}{tag}: {cnt} points")
