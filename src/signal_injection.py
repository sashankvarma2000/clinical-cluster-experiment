"""
signal_injection.py
--------------------
Simulates an epidemiological outbreak-detection scenario.

50 synthetic "acute vision disturbance after medication" records, written in
10 surface forms (5 copies each), are injected into the dataset AFTER the
baseline embeddings are computed.  Each embedding method then re-clusters the
augmented dataset and we measure:

  recovery_rate   — fraction of the 50 injected records that land in the
                    modal (most common) cluster among them.
  cluster_purity  — fraction of records in that modal cluster that are
                    injected (synthetic).  High purity → tight signal.

A recovery_rate near 1.0 means the system would detect the outbreak.
A rate near 0.1 means records are scattered across ten random clusters.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

# ── Synthetic outbreak records ────────────────────────────────────────────────

OUTBREAK_PHRASES: List[str] = [
    "sudden vision loss after taking medication",
    "patient reports blurry eyesight post-dose",
    "acute visual disturbance following drug administration",
    "can't see clearly since starting the medication",
    "eyes went blurry after first pill",
    "visual acuity decreased after medication initiation",
    "sight problems beginning after prescription started",
    "went blind in one eye after taking the drug",
    "difficulty seeing following pharmacological treatment",
    "ocular symptoms appeared post-medication",
]

RECORDS_PER_PHRASE: int = 5
N_INJECTED: int = len(OUTBREAK_PHRASES) * RECORDS_PER_PHRASE  # 50


def build_injected_texts() -> List[str]:
    """Return the list of 50 synthetic outbreak texts.

    Each of the 10 surface forms is repeated ``RECORDS_PER_PHRASE`` times
    to simulate multiple patients reporting the same event in different words.

    Returns
    -------
    list of str, length 50
    """
    return [phrase for phrase in OUTBREAK_PHRASES for _ in range(RECORDS_PER_PHRASE)]


def run_signal_injection(
    base_embeddings: np.ndarray,
    embed_fn: Callable[[List[str]], np.ndarray | None],
    cluster_fn: Callable[[np.ndarray], np.ndarray],
    method_name: str,
    cache_suffix: str,
) -> Dict[str, float]:
    """Inject synthetic records, cluster, and measure detection performance.

    The injected texts are embedded using *embed_fn* (with a unique
    cache_suffix so they are stored separately from the baseline), then
    appended to *base_embeddings*.  The full dataset is then re-clustered
    using *cluster_fn*.

    Parameters
    ----------
    base_embeddings : np.ndarray, shape (n_base, d)
        Pre-computed embeddings for the original dataset.
    embed_fn : callable
        Embedding function for new texts.  Signature:
        ``embed_fn(texts: list[str]) -> np.ndarray | None``.
    cluster_fn : callable
        Clustering function.  Signature:
        ``cluster_fn(embeddings: np.ndarray) -> np.ndarray``.
    method_name : str
        Human-readable name for logging.
    cache_suffix : str
        Suffix used to cache the injected-text embeddings.

    Returns
    -------
    dict with keys:
        ``recovery_rate``  (float in [0, 1])
        ``cluster_purity`` (float in [0, 1])
        ``modal_cluster``  (int)
        ``n_noise``        (int)
    """
    injected_texts = build_injected_texts()
    n_base = len(base_embeddings)

    print(f"\n[Signal Injection — {method_name}]")
    print(f"  Embedding {N_INJECTED} synthetic outbreak records …")

    injected_emb = embed_fn(injected_texts, cache_suffix=f"{cache_suffix}_injected")

    if injected_emb is None:
        print(f"  ⚠  Skipping signal injection for {method_name} (embed_fn returned None).")
        return {
            "recovery_rate": float("nan"),
            "cluster_purity": float("nan"),
            "modal_cluster": -1,
            "n_noise": -1,
        }

    # Combine base + injected
    combined = np.vstack([base_embeddings, injected_emb])

    # Re-cluster the combined dataset
    all_labels = cluster_fn(combined)

    # Extract the labels for the injected records only
    injected_labels = all_labels[n_base:]

    # Modal cluster (ignoring noise label -1 when possible)
    non_noise = injected_labels[injected_labels != -1]
    if len(non_noise) == 0:
        print("  ⚠  All injected records classified as noise.")
        return {
            "recovery_rate": 0.0,
            "cluster_purity": 0.0,
            "modal_cluster": -1,
            "n_noise": int((injected_labels == -1).sum()),
        }

    values, cnts = np.unique(non_noise, return_counts=True)
    modal_cluster = int(values[cnts.argmax()])
    modal_count = int(cnts.max())

    recovery_rate = modal_count / N_INJECTED

    # Purity: how many records in the modal cluster are injected?
    modal_total = int((all_labels == modal_cluster).sum())
    cluster_purity = modal_count / modal_total if modal_total > 0 else 0.0

    n_noise = int((injected_labels == -1).sum())

    print(f"  Modal cluster: {modal_cluster}  "
          f"({modal_count}/{N_INJECTED} injected records = {recovery_rate:.1%})")
    print(f"  Cluster purity: {cluster_purity:.1%}  "
          f"({modal_count} injected / {modal_total} total in cluster)")
    print(f"  Noise: {n_noise} of {N_INJECTED} injected records")

    return {
        "recovery_rate": recovery_rate,
        "cluster_purity": cluster_purity,
        "modal_cluster": modal_cluster,
        "n_noise": n_noise,
        "all_labels": all_labels,          # full label array for visualisation
        "injected_labels": injected_labels,
        "n_base": n_base,
    }


def get_injected_embeddings_for_viz(
    method_name: str,
    cache_suffix: str,
    embed_fn: Callable[[List[str]], np.ndarray | None],
) -> Tuple[List[str], np.ndarray | None]:
    """Return (texts, embeddings) for the 50 injected records.

    Convenience wrapper used by visualisation code.

    Parameters
    ----------
    method_name : str
        Human-readable label for logging.
    cache_suffix : str
        Suffix used to find the cached file.
    embed_fn : callable
        Embedding function (will only be called on cache-miss).

    Returns
    -------
    tuple (list[str], np.ndarray or None)
    """
    texts = build_injected_texts()
    embs = embed_fn(texts, cache_suffix=f"{cache_suffix}_injected")
    return texts, embs
