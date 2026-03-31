# The Clustering Layer That Public Health Surveillance Is Missing

Reproducible experiment accompanying the Towards Data Science article:

> *"The Clustering Layer That Public Health Surveillance Is Missing: How Embedding
> Choice Affects Signal Detectability in Clinical Free-Text"*

## What the experiment does

We compare three text-embedding approaches on 4,999 public medical transcription
records (MTSamples), measuring how well each method groups clinical notes by
specialty — and, more importantly, whether it can detect a simulated disease
outbreak embedded in the dataset as 50 synthetic free-text records.

| Method | Embedding | Clustering |
|---|---|---|
| Baseline | TF-IDF + LSA (100 dims) | K-means (k=10) |
| Open-source semantic | MiniLM-L6-v2 | HDBSCAN |
| API-based semantic | OpenAI text-embedding-3-small | HDBSCAN |

**Metrics evaluated:**
- Adjusted Rand Index (ARI) — alignment with ground-truth specialties
- Silhouette Score — intrinsic cluster quality on UMAP-reduced embeddings
- Signal Recovery Rate — fraction of 50 injected "outbreak" records that land
  in the same cluster (novel metric, described in the article)

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Set your OpenAI API key to enable Method 3
export OPENAI_API_KEY=sk-...

# 3. Run the full experiment
python main.py
```

The OpenAI method is skipped gracefully if `OPENAI_API_KEY` is not set.
Methods 1 and 2 run entirely offline.

## Data

Place `mtsamples.csv` (downloaded from
[Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions))
at `data/mtsamples.csv` before running.

The MTSamples dataset is released under the
[CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).

## Outputs

All outputs are written to `results/`:

| File | Description |
|---|---|
| `summary.csv` | ARI, Silhouette, recovery rate, and purity per method |
| `umap_comparison.png` | Three UMAP scatter plots coloured by specialty |
| `signal_injection_umap.png` | Where outbreak records land on each UMAP |
| `summary_metrics.png` | Bar chart comparing all metrics across methods |
| `embeddings_*.npy` | Cached embeddings (speeds up re-runs) |

## Project structure

```
clinical-cluster-experiment/
├── data/
│   └── mtsamples.csv          ← place dataset here
├── notebooks/
│   └── experiment.ipynb       ← Colab-ready walkthrough
├── src/
│   ├── data_loader.py
│   ├── embeddings.py
│   ├── clustering.py
│   ├── evaluation.py
│   ├── signal_injection.py
│   └── visualization.py
├── results/                   ← auto-generated
├── main.py
├── requirements.txt
└── README.md
```
