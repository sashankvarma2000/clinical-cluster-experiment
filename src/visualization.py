"""
visualization.py
----------------
All publication-ready plots for the experiment.

Outputs (saved to results/):
  umap_comparison.png        — three UMAP scatter plots, coloured by specialty
  signal_injection_umap.png  — three UMAP panels showing injected-record positions
  summary_metrics.png        — bar chart: ARI and signal recovery rate per method

All figures use a clean, minimal style suitable for a TDS article.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]
NOISE_COLOR = "#CCCCCC"
INJECT_COLOR = "#E63946"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "figure.dpi": 150,
})

METHOD_LABELS = {
    "tfidf": "TF-IDF + K-means",
    "minilm": "MiniLM + HDBSCAN",
    "openai": "OpenAI + HDBSCAN",
}


# ── UMAP reduction ─────────────────────────────────────────────────────────────

def compute_umap_2d(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Reduce *embeddings* to 2-D with UMAP.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n, d)
        High-dimensional embedding matrix.
    random_state : int
        Seed for UMAP.

    Returns
    -------
    np.ndarray, shape (n, 2)
    """
    from umap import UMAP

    reducer = UMAP(n_components=2, random_state=random_state, verbose=False)
    return reducer.fit_transform(embeddings).astype(np.float32)


# ── Plot 1: UMAP coloured by specialty ────────────────────────────────────────

def plot_umap_comparison(
    umap_dict: Dict[str, np.ndarray],
    specialty_ids: np.ndarray,
    specialty_names: List[str],
    save_path: Optional[Path] = None,
) -> None:
    """Three side-by-side UMAP scatter plots coloured by ground-truth specialty.

    Parameters
    ----------
    umap_dict : dict {method_key: np.ndarray shape (n, 2)}
        UMAP projections for each method.  Keys must be in METHOD_LABELS.
    specialty_ids : np.ndarray, shape (n,)
        Integer specialty labels aligned with the UMAP rows.
    specialty_names : list of str
        Name for each integer id (index = id).
    save_path : Path, optional
        Where to save the figure.  Defaults to results/umap_comparison.png.
    """
    save_path = save_path or RESULTS_DIR / "umap_comparison.png"
    methods = [k for k in METHOD_LABELS if k in umap_dict]

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5))
    if len(methods) == 1:
        axes = [axes]

    unique_ids = np.unique(specialty_ids)
    color_map = {sid: PALETTE[i % len(PALETTE)] for i, sid in enumerate(unique_ids)}

    for ax, method in zip(axes, methods):
        xy = umap_dict[method]
        for sid in unique_ids:
            mask = specialty_ids == sid
            ax.scatter(
                xy[mask, 0], xy[mask, 1],
                c=color_map[sid],
                s=8, alpha=0.6, linewidths=0,
                label=specialty_names[sid] if sid < len(specialty_names) else str(sid),
            )
        ax.set_title(METHOD_LABELS[method], fontweight="bold")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared legend below the figure
    patches = [
        mpatches.Patch(color=color_map[sid],
                       label=specialty_names[sid] if sid < len(specialty_names) else str(sid))
        for sid in unique_ids
    ]
    fig.legend(
        handles=patches, loc="lower center",
        ncol=5, fontsize=7, frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        "UMAP Projections — Coloured by Medical Specialty",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Plot 2: Signal injection UMAP ─────────────────────────────────────────────

def plot_signal_injection_umap(
    umap_dict: Dict[str, np.ndarray],
    n_base: int,
    save_path: Optional[Path] = None,
) -> None:
    """Three UMAP panels showing where injected records land.

    Background (real) records are grey; injected records are highlighted red.

    Parameters
    ----------
    umap_dict : dict {method_key: np.ndarray shape (n_base + 50, 2)}
        UMAP projections of the *combined* (base + injected) dataset.
    n_base : int
        Number of real (non-injected) records.
    save_path : Path, optional
        Save location.  Defaults to results/signal_injection_umap.png.
    """
    save_path = save_path or RESULTS_DIR / "signal_injection_umap.png"
    methods = [k for k in METHOD_LABELS if k in umap_dict]

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        xy = umap_dict[method]
        base_xy = xy[:n_base]
        inj_xy = xy[n_base:]

        ax.scatter(
            base_xy[:, 0], base_xy[:, 1],
            c=NOISE_COLOR, s=6, alpha=0.35, linewidths=0,
            label="Real records",
        )
        ax.scatter(
            inj_xy[:, 0], inj_xy[:, 1],
            c=INJECT_COLOR, s=30, alpha=0.85, linewidths=0,
            label="Injected (outbreak)",
            zorder=5,
        )
        ax.set_title(METHOD_LABELS[method], fontweight="bold")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend
    legend_patches = [
        mpatches.Patch(color=NOISE_COLOR, label="Real records"),
        mpatches.Patch(color=INJECT_COLOR, label="Injected outbreak records"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=2, fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "Signal Injection — Where Outbreak Records Land",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Plot 3: Summary metrics bar chart ─────────────────────────────────────────

def plot_summary_metrics(
    results: Dict[str, Dict],
    save_path: Optional[Path] = None,
) -> None:
    """Grouped bar chart: ARI, Silhouette, and Signal Detectability per method.

    Signal Detectability (recovery_rate × cluster_purity) is the primary red bar
    — it is the article's headline metric and appears rightmost in each group.

    Parameters
    ----------
    results : dict {method_key: {ari, silhouette, signal_detectability, ...}}
        Aggregated metrics dictionary.
    save_path : Path, optional
        Save location.  Defaults to results/summary_metrics.png.
    """
    save_path = save_path or RESULTS_DIR / "summary_metrics.png"

    methods = [k for k in METHOD_LABELS if k in results]
    labels = [METHOD_LABELS[m] for m in methods]

    ari_vals = [results[m].get("ari", float("nan")) for m in methods]
    sil_vals = [results[m].get("silhouette", float("nan")) for m in methods]
    det_vals = [results[m].get("signal_detectability", float("nan")) for m in methods]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))

    bars_ari = ax.bar(x - width, ari_vals, width, label="ARI",
                      color=PALETTE[0], alpha=0.85)
    bars_sil = ax.bar(x,         sil_vals, width, label="Silhouette",
                      color=PALETTE[2], alpha=0.85)
    bars_det = ax.bar(x + width, det_vals, width,
                      label="Signal Detectability (recovery × purity)",
                      color=INJECT_COLOR, alpha=0.90)

    def _annotate(bars):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                )

    _annotate(bars_ari)
    _annotate(bars_sil)
    _annotate(bars_det)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(
        "Clustering Quality and Signal Detection by Embedding Method",
        fontweight="bold", fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(1.0, color="#999999", linewidth=0.6, linestyle="--")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")
