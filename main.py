"""
main.py
-------
Orchestrates the full clustering-comparison experiment end-to-end.

Stages
------
1  Load data (MTSamples, top-10 specialties)
2  Compute embeddings  (TF-IDF, MiniLM, OpenAI*)
3  Cluster each embedding set
4  UMAP reduction → 2-D
5  Compute ARI + Silhouette
6  Signal injection experiment
7  Visualisations
8  Save results/summary.csv

*OpenAI is skipped gracefully if OPENAI_API_KEY is not set.

Usage
-----
    python main.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── Stage helpers ──────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    width = 60
    print(f"\n{'#'*width}")
    print(f"#  {text}")
    print(f"{'#'*width}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.perf_counter()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    _banner("Stage 1 — Load Data")
    from src.data_loader import load_mtsamples
    df = load_mtsamples()

    texts: list[str] = df["description"].tolist()
    true_labels: np.ndarray = df["specialty_id"].to_numpy()
    specialty_names: list[str] = sorted(df["specialty"].unique())
    n_specialties = len(specialty_names)
    print(f"  {len(texts):,} records  |  {n_specialties} specialties")

    # ── 2. Compute embeddings ─────────────────────────────────────────────────
    _banner("Stage 2 — Compute Embeddings")
    from src.embeddings import embed_tfidf, embed_minilm, embed_openai

    emb_tfidf  = embed_tfidf(texts)
    emb_minilm = embed_minilm(texts)
    emb_openai = embed_openai(texts)          # None if no API key

    embedding_map = {"tfidf": emb_tfidf, "minilm": emb_minilm}
    if emb_openai is not None:
        embedding_map["openai"] = emb_openai

    # ── 3. Cluster ────────────────────────────────────────────────────────────
    _banner("Stage 3 — Cluster")
    from src.clustering import cluster_kmeans, cluster_hdbscan

    cluster_labels: dict[str, np.ndarray] = {}
    cluster_labels["tfidf"]  = cluster_kmeans(emb_tfidf)
    cluster_labels["minilm"] = cluster_hdbscan(emb_minilm, method_name="MiniLM HDBSCAN")
    if emb_openai is not None:
        cluster_labels["openai"] = cluster_hdbscan(emb_openai, method_name="OpenAI HDBSCAN")

    # ── 4. UMAP reduction ─────────────────────────────────────────────────────
    _banner("Stage 4 — UMAP Reduction (2-D)")
    from src.visualization import compute_umap_2d

    umap_2d: dict[str, np.ndarray] = {}
    for key, emb in embedding_map.items():
        print(f"  [{key}] UMAP …")
        umap_2d[key] = compute_umap_2d(emb)

    # ── 5. ARI + Silhouette ───────────────────────────────────────────────────
    _banner("Stage 5 — Evaluation (ARI + Silhouette)")
    from src.evaluation import compute_ari, compute_silhouette

    results: dict[str, dict] = {}
    for key in embedding_map:
        ari = compute_ari(cluster_labels[key], true_labels)
        sil = compute_silhouette(umap_2d[key], cluster_labels[key])
        results[key] = {"ari": ari, "silhouette": sil}
        print(f"  [{key}]  ARI={ari:.4f}  Silhouette={sil:.4f}")

    # ── 6. Signal injection ───────────────────────────────────────────────────
    _banner("Stage 6 — Signal Injection Experiment")
    from src.signal_injection import run_signal_injection, build_injected_texts
    from src.clustering import cluster_kmeans, cluster_hdbscan
    from src.embeddings import embed_tfidf, embed_minilm, embed_openai, transform_tfidf
    from src.evaluation import compute_signal_detectability

    # embed_fn for injection:
    #   TF-IDF  → transform_tfidf (reuse fitted pipeline, same feature space)
    #   Semantic → normal encode function (model is stateless w.r.t. vocabulary)
    injection_configs = [
        ("tfidf",   emb_tfidf,   transform_tfidf, lambda e: cluster_kmeans(e)),
        ("minilm",  emb_minilm,  embed_minilm,    lambda e: cluster_hdbscan(e, "MiniLM HDBSCAN")),
    ]
    if emb_openai is not None:
        injection_configs.append(
            ("openai", emb_openai, embed_openai,
             lambda e: cluster_hdbscan(e, "OpenAI HDBSCAN"))
        )

    # umap_injected: one 2-D projection per method, fitted on real+injected TOGETHER.
    # We stack [base_emb | injected_emb] and call compute_umap_2d on the combined
    # matrix so both sets of points share a single coordinate space.
    umap_injected: dict[str, np.ndarray] = {}
    n_base = len(texts)
    injected_texts = build_injected_texts()   # the 50 synthetic outbreak phrases

    for key, base_emb, efn, cfn in injection_configs:
        inj_result = run_signal_injection(
            base_embeddings=base_emb,
            embed_fn=efn,
            cluster_fn=cfn,
            method_name=key,
            cache_suffix=key,
        )
        results[key]["recovery_rate"]  = inj_result["recovery_rate"]
        results[key]["cluster_purity"] = inj_result["cluster_purity"]
        results[key]["signal_detectability"] = compute_signal_detectability(
            inj_result["recovery_rate"], inj_result["cluster_purity"]
        )

        # Combined UMAP: embed injected texts with the same function, then fit
        # UMAP on the concatenated [base | injected] matrix in one call.
        if "all_labels" in inj_result:
            inj_cache_path = RESULTS_DIR / f"embeddings_{key}_injected.npy"
            if inj_cache_path.exists():
                inj_emb = np.load(inj_cache_path)
                combined_emb = np.vstack([base_emb, inj_emb])
                print(f"  [{key}] UMAP on combined ({n_base} real + {len(inj_emb)} injected) …")
                umap_injected[key] = compute_umap_2d(combined_emb)

    # Fill missing metrics for any method that had no injection run
    for key in embedding_map:
        results[key].setdefault("recovery_rate",        float("nan"))
        results[key].setdefault("cluster_purity",       float("nan"))
        results[key].setdefault("signal_detectability", float("nan"))

    # ── 7. Visualisations ─────────────────────────────────────────────────────
    _banner("Stage 7 — Visualisations")
    from src.evaluation import print_results_table
    from src.visualization import (
        plot_umap_comparison,
        plot_signal_injection_umap,
        plot_summary_metrics,
    )

    print_results_table(results)

    print("  Saving UMAP comparison …")
    plot_umap_comparison(umap_2d, true_labels, specialty_names)

    if umap_injected:
        print("  Saving signal injection UMAP …")
        plot_signal_injection_umap(umap_injected, n_base=n_base)

    print("  Saving summary metrics bar chart …")
    plot_summary_metrics(results)

    # ── 8. Save CSV ───────────────────────────────────────────────────────────
    _banner("Stage 8 — Save Summary CSV")
    rows = []
    for method, vals in results.items():
        row = {"method": method}
        row.update({k: v for k, v in vals.items()
                    if isinstance(v, (int, float))})
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")
    print(summary_df.to_string(index=False))

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  Experiment complete.  Total runtime: {elapsed:.1f}s")
    print(f"  Results saved to: {RESULTS_DIR.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
