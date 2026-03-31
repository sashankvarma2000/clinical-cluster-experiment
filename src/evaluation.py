"""
evaluation.py
-------------
Evaluation metrics for the clustering comparison:

  • Adjusted Rand Index  (ARI) — measures agreement with ground-truth labels.
    Noise points (label -1) are excluded before computing.
  • Silhouette Score — measures cluster compactness/separation on the
    2-D UMAP projection (no labels required).
  • Signal Detectability — primary article metric: recovery_rate × cluster_purity.
    Captures both whether the signal was concentrated (recovery) AND whether
    the resulting cluster was meaningfully pure (purity).  A method that groups
    all injected records into one huge cluster gets penalised by low purity.

Results are printed as a formatted table and returned as a dict.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score


def compute_ari(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Adjusted Rand Index between cluster assignments and ground truth.

    Noise points (cluster label == -1) are excluded from the calculation
    because HDBSCAN noise is "undecided" rather than mis-assigned.

    Parameters
    ----------
    cluster_labels : np.ndarray, shape (n,)
        Cluster assignments (may contain -1 for noise).
    true_labels : np.ndarray, shape (n,)
        Ground-truth integer class labels.

    Returns
    -------
    float
        ARI in [-1, 1]; higher is better. 1.0 = perfect agreement.
    """
    mask = cluster_labels != -1
    if mask.sum() == 0:
        return 0.0
    return float(adjusted_rand_score(true_labels[mask], cluster_labels[mask]))


def compute_silhouette(
    embeddings_2d: np.ndarray,
    cluster_labels: np.ndarray,
    sample_size: int = 2000,
) -> float:
    """Silhouette score computed on 2-D UMAP projections.

    Noise points are excluded.  A random sub-sample is used when the dataset
    is large so the metric remains fast.

    Parameters
    ----------
    embeddings_2d : np.ndarray, shape (n, 2)
        UMAP 2-D projection of the embeddings.
    cluster_labels : np.ndarray, shape (n,)
        Cluster assignments (may contain -1).
    sample_size : int
        Maximum number of points to pass to ``silhouette_score``.

    Returns
    -------
    float
        Silhouette score in [-1, 1]; higher is better.
    """
    mask = cluster_labels != -1
    X = embeddings_2d[mask]
    y = cluster_labels[mask]

    if len(np.unique(y)) < 2:
        return float("nan")

    if len(X) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), sample_size, replace=False)
        X, y = X[idx], y[idx]

    return float(silhouette_score(X, y))


def compute_signal_detectability(recovery_rate: float, cluster_purity: float) -> float:
    """Primary article metric: recovery_rate × cluster_purity.

    A method that groups all injected records into one massive cluster scores
    high on recovery but near-zero on purity — and is penalised accordingly.
    A method that produces a tight, high-purity cluster of outbreak records
    scores near 1.0 on both axes and approaches 1.0 overall.

    Parameters
    ----------
    recovery_rate : float
        Fraction of injected records in the modal cluster.
    cluster_purity : float
        Fraction of the modal cluster that is injected (synthetic).

    Returns
    -------
    float
        Signal detectability in [0, 1].  NaN if either input is NaN.
    """
    import math
    if math.isnan(recovery_rate) or math.isnan(cluster_purity):
        return float("nan")
    return recovery_rate * cluster_purity


def print_results_table(results: Dict[str, Dict]) -> None:
    """Print a formatted comparison table of all metrics.

    Parameters
    ----------
    results : dict
        Keys are method names; values are dicts with keys
        ``ari``, ``silhouette``, ``recovery_rate``, ``cluster_purity``,
        ``signal_detectability``.
    """
    col_w = 18
    metrics = ["ari", "silhouette", "recovery_rate", "cluster_purity",
               "signal_detectability"]
    headers = ["Method", "ARI", "Silhouette", "Recovery", "Purity",
               "Detectability"]

    sep = "+" + "+".join(["-" * col_w] * len(headers)) + "+"
    header_row = "|" + "|".join(h.center(col_w) for h in headers) + "|"

    print(f"\n{'='*len(sep)}")
    print("  RESULTS SUMMARY")
    print('='*len(sep))
    print(sep)
    print(header_row)
    print(sep)

    for method, vals in results.items():
        row_vals = [method] + [
            f"{vals.get(m, float('nan')):.4f}" if not isinstance(vals.get(m, ""), str)
            else str(vals.get(m, "—"))
            for m in metrics
        ]
        print("|" + "|".join(v.center(col_w) for v in row_vals) + "|")

    print(sep)
    print()
