# Experiment Summary

**"The Clustering Layer That Public Health Surveillance Is Missing:
How Embedding Choice Affects Signal Detectability in Clinical Free-Text"**

---

## The Question

Public health surveillance systems that ingest clinical free-text must convert
that text into vectors before they can detect patterns. The standard approach
is TF-IDF — a keyword frequency method that has been in use for decades.

The experiment asks a single, sharp question:

> **If an outbreak is described in 10 different ways by 50 different patients,
> does the embedding method determine whether a surveillance system detects it
> or misses it?**

---

## Dataset

**MTSamples** — 4,999 medical transcription records, CC0 licensed.
- Input field: `description` — 1–3 sentence clinical summaries per record
- Ground truth: `medical_specialty` — filtered to the top 10 specialties by count
- Final working set: **3,652 records across 10 specialties**

| Specialty | Records |
|---|---|
| Surgery | 1,102 |
| Consult - History and Phy. | 514 |
| Cardiovascular / Pulmonary | 372 |
| Orthopedic | 355 |
| Radiology | 273 |
| General Medicine | 259 |
| Gastroenterology | 229 |
| Neurology | 222 |
| SOAP / Chart / Progress Notes | 166 |
| Obstetrics / Gynecology | 160 |

---

## Three Methods Compared

| # | Method | Embedding | Clustering | Dimensions |
|---|---|---|---|---|
| 1 | **Baseline** | TF-IDF (max 5,000 features, bigrams) + TruncatedSVD | K-means (k=10) | 100 |
| 2 | **Open-source semantic** | sentence-transformers all-MiniLM-L6-v2 | HDBSCAN | 384 |
| 3 | **API-based semantic** | OpenAI text-embedding-3-small | HDBSCAN | 1,536 |

HDBSCAN parameters: `min_cluster_size=30`, `min_samples=5`.
All random seeds fixed at 42. Embeddings cached to `.npy` files after first run.

---

## Evaluation Metrics

### Standard clustering metrics
- **ARI (Adjusted Rand Index)** — agreement between cluster assignments and
  ground-truth specialties. Noise points (HDBSCAN label −1) excluded.
- **Silhouette Score** — intrinsic cluster quality computed on 2-D UMAP
  projections. No labels required.

### Signal injection metrics (novel)
50 synthetic "outbreak" records are injected after embedding. All 50 describe
the same clinical event — *acute vision disturbance following medication* —
written in 10 different surface forms (5 copies each):

1. "sudden vision loss after taking medication"
2. "patient reports blurry eyesight post-dose"
3. "acute visual disturbance following drug administration"
4. "can't see clearly since starting the medication"
5. "eyes went blurry after first pill"
6. "visual acuity decreased after medication initiation"
7. "sight problems beginning after prescription started"
8. "went blind in one eye after taking the drug"
9. "difficulty seeing following pharmacological treatment"
10. "ocular symptoms appeared post-medication"

- **Recovery Rate** — fraction of the 50 injected records that land in the
  same (modal) cluster.
- **Cluster Purity** — fraction of that modal cluster that is injected
  (synthetic). A high-purity cluster means the signal is clean and isolated.
- **Signal Detectability** = Recovery Rate × Cluster Purity.
  The primary article metric. Penalises methods that achieve high recovery by
  dumping the outbreak into a massive existing cluster where it would be invisible.

---

## Results

| Method | ARI | Silhouette | Recovery | Purity | **Detectability** |
|---|---|---|---|---|---|
| TF-IDF + K-means | 0.089 | −0.063 | 90% | 3.1% | **0.028** |
| MiniLM + HDBSCAN | 0.121 | 0.217 | 90% | 95.7% | **0.862** |
| OpenAI + HDBSCAN | 0.154 | 0.133 | 100% | 100% | **1.000** |

### What the numbers mean

**TF-IDF** placed 45 of 50 injected records in the same K-means cluster —
but that cluster contained **1,472 real records**. The outbreak signal is
buried at a 3.1% concentration. A surveillance analyst scanning cluster
outputs would see nothing unusual. Detectability: **0.028**.

**MiniLM** placed 45 of 50 injected records in a cluster of only 47 total
records — 95.7% synthetic. Five records were classified as noise (−1).
The cluster is a near-perfect outbreak signal, but 5 cases were missed.
Detectability: **0.862**.

**OpenAI** placed all **50 of 50** injected records in a dedicated cluster
of exactly 50 records. Zero real records mixed in. Zero noise. HDBSCAN
found the outbreak as a standalone cluster in 1,536-dimensional space with
no parameter tuning. Detectability: **1.000**.

---

## Key Finding

All three methods achieve similar **recovery rates** (~90–100%) — meaning
they all "find" most of the injected records somewhere. But the
**detectability gap is 36×** between the best and worst method (1.000 vs 0.028).

The difference is entirely explained by cluster purity. TF-IDF creates
clusters by keyword overlap: the outbreak records land in a large general
cluster because they share words like "medication" with thousands of real
records. Semantic methods encode *meaning*, not word counts, and place all
variants of "vision problem after drug" near each other — and nowhere near
unrelated records.

> The surveillance blind spot is not that TF-IDF fails to cluster.
> It is that it clusters too broadly for the signal to be visible.

---

## Project Structure

```
clinical-cluster-experiment/
├── data/
│   └── mtsamples.csv              # CC0 dataset (place here before running)
├── notebooks/
│   └── experiment.ipynb           # Colab-ready walkthrough
├── src/
│   ├── data_loader.py             # Load, filter, label MTSamples
│   ├── embeddings.py              # TF-IDF, MiniLM, OpenAI — with caching
│   ├── clustering.py              # K-means and HDBSCAN wrappers
│   ├── evaluation.py              # ARI, Silhouette, Signal Detectability
│   ├── signal_injection.py        # Inject synthetic records, measure recovery
│   └── visualization.py           # UMAP plots, bar chart
├── results/
│   ├── summary.csv                # All metrics, all methods
│   ├── umap_comparison.png        # UMAP coloured by specialty (3 panels)
│   ├── signal_injection_umap.png  # Where outbreak records land (3 panels)
│   ├── summary_metrics.png        # Bar chart: ARI, Silhouette, Detectability
│   └── embeddings_*.npy           # Cached embeddings (skip re-computation)
├── main.py                        # End-to-end experiment runner
├── requirements.txt
├── README.md                      # Setup and usage instructions
└── summary.md                     # This file
```

---

## How to Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set OpenAI API key (required for Method 3 only)
export OPENAI_API_KEY=sk-...

# 3. Run
python main.py
```

Total runtime: ~2 minutes on an M2 MacBook Air (CPU only).
On re-runs, cached `.npy` embeddings are loaded instantly — only
clustering, UMAP, and plotting are repeated (~30 seconds).

---

## Data Citation

MTSamples dataset by tboyle10, available at
https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions,
released under the CC0 1.0 Universal Public Domain Dedication.
