"""
embeddings.py
-------------
Three text-embedding methods used in the clustering comparison:

  1. TF-IDF (max 5 000 features, bigrams) → TruncatedSVD to 100 dims (LSA)
  2. sentence-transformers  all-MiniLM-L6-v2  (batch_size=64)
  3. OpenAI  text-embedding-3-small  (chunked, with retry on rate-limit)

Each public function accepts a list/array of strings and returns a
``numpy.ndarray`` of shape ``(n_samples, n_dims)``.

Embeddings are cached to ``results/<method>.npy`` so repeated runs are free.
"""

import os
import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
from tqdm import tqdm

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Path where the fitted TF-IDF pipeline is persisted so injection can reuse it
_TFIDF_PIPELINE_PATH = RESULTS_DIR / "tfidf_pipeline.joblib"


# ── helpers ───────────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    return RESULTS_DIR / f"embeddings_{name}.npy"


def _load_or_none(name: str) -> np.ndarray | None:
    p = _cache_path(name)
    if p.exists():
        print(f"  [cache] Loading {name} embeddings from {p}")
        return np.load(p)
    return None


def _save(name: str, arr: np.ndarray) -> None:
    np.save(_cache_path(name), arr)
    print(f"  [cache] Saved {name} embeddings → {_cache_path(name)}")


# ── Method 1: TF-IDF + LSA ────────────────────────────────────────────────────

def embed_tfidf(texts: List[str], cache_suffix: str = "tfidf") -> np.ndarray:
    """Compute TF-IDF bag-of-words embeddings reduced via LSA (TruncatedSVD).

    Parameters
    ----------
    texts : list of str
        Input documents.
    cache_suffix : str
        Key used for the on-disk cache file name.

    Returns
    -------
    np.ndarray, shape (n, 100)
        LSA-reduced TF-IDF vectors.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline

    def _build_and_fit_pipeline(texts):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5_000, ngram_range=(1, 2),
                                       sublinear_tf=True)),
            ("svd",   TruncatedSVD(n_components=100, random_state=42)),
        ])
        pipe.fit(texts)
        joblib.dump(pipe, _TFIDF_PIPELINE_PATH)
        return pipe

    cached = _load_or_none(cache_suffix)
    if cached is not None:
        # Ensure the fitted pipeline exists for later transform calls
        if not _TFIDF_PIPELINE_PATH.exists():
            print("[TF-IDF] Re-fitting pipeline to rebuild saved state …")
            _build_and_fit_pipeline(texts)
        return cached

    print("\n[TF-IDF] Building TF-IDF + LSA embeddings …")
    t0 = time.perf_counter()

    pipe = _build_and_fit_pipeline(texts)
    arr = pipe.transform(texts).astype(np.float32)

    elapsed = time.perf_counter() - t0
    print(f"[TF-IDF] Done in {elapsed:.1f}s  —  shape {arr.shape}")
    _save(cache_suffix, arr)
    return arr


def transform_tfidf(texts: List[str], cache_suffix: str) -> np.ndarray:
    """Transform *texts* using the already-fitted TF-IDF + LSA pipeline.

    Must be called after :func:`embed_tfidf` has been run on the base corpus
    (which saves the fitted pipeline to disk).  Used for signal injection so
    the injected records share the same feature space as the base embeddings.

    Parameters
    ----------
    texts : list of str
        New documents to embed.
    cache_suffix : str
        Cache key; if the .npy file already exists it is returned immediately.

    Returns
    -------
    np.ndarray, shape (n, 100)
    """
    cached = _load_or_none(cache_suffix)
    if cached is not None:
        return cached

    if not _TFIDF_PIPELINE_PATH.exists():
        raise RuntimeError(
            "Fitted TF-IDF pipeline not found. Run embed_tfidf() on the base "
            "corpus first."
        )

    pipe = joblib.load(_TFIDF_PIPELINE_PATH)
    arr = pipe.transform(texts).astype(np.float32)
    _save(cache_suffix, arr)
    print(f"  [TF-IDF transform] shape {arr.shape}")
    return arr


# ── Method 2: sentence-transformers MiniLM ────────────────────────────────────

def embed_minilm(texts: List[str], cache_suffix: str = "minilm") -> np.ndarray:
    """Compute dense sentence embeddings using all-MiniLM-L6-v2.

    Parameters
    ----------
    texts : list of str
        Input documents.
    cache_suffix : str
        Key used for the on-disk cache file name.

    Returns
    -------
    np.ndarray, shape (n, 384)
        Unit-normalised sentence embeddings.
    """
    cached = _load_or_none(cache_suffix)
    if cached is not None:
        return cached

    from sentence_transformers import SentenceTransformer

    print("\n[MiniLM] Loading all-MiniLM-L6-v2 …")
    t0 = time.perf_counter()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    arr = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    elapsed = time.perf_counter() - t0
    print(f"[MiniLM] Done in {elapsed:.1f}s  —  shape {arr.shape}")
    _save(cache_suffix, arr)
    return arr


# ── Method 3: OpenAI text-embedding-3-small ───────────────────────────────────

def embed_openai(
    texts: List[str],
    cache_suffix: str = "openai",
    chunk_size: int = 100,
    max_retries: int = 5,
) -> np.ndarray | None:
    """Compute embeddings via the OpenAI API (text-embedding-3-small).

    Sends texts in batches of *chunk_size* to respect rate limits.
    Retries up to *max_retries* times on RateLimitError with exponential back-off.

    If ``OPENAI_API_KEY`` is not set, prints a warning and returns ``None``
    so the rest of the experiment continues without OpenAI.

    Parameters
    ----------
    texts : list of str
        Input documents.
    cache_suffix : str
        Key used for the on-disk cache file name.
    chunk_size : int
        Number of texts per API call.
    max_retries : int
        Max retry attempts on rate-limit errors.

    Returns
    -------
    np.ndarray, shape (n, 1536)  or  None
        OpenAI embedding vectors, or None when API key is absent.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print(
            "\n[OpenAI] ⚠  OPENAI_API_KEY not set — skipping OpenAI embeddings.\n"
            "         Set the variable and re-run; cached results will be used."
        )
        return None

    cached = _load_or_none(cache_suffix)
    if cached is not None:
        return cached

    try:
        from openai import OpenAI, RateLimitError
    except ImportError:
        print("[OpenAI] openai package not installed — skipping.")
        return None

    client = OpenAI(api_key=api_key)
    model_id = "text-embedding-3-small"

    print(f"\n[OpenAI] Embedding {len(texts):,} texts with {model_id} …")
    t0 = time.perf_counter()

    all_embeddings: List[np.ndarray] = []
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

    for chunk in tqdm(chunks, desc="[OpenAI] batches"):
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(model=model_id, input=chunk)
                vecs = [np.array(item.embedding, dtype=np.float32)
                        for item in response.data]
                all_embeddings.extend(vecs)
                break
            except RateLimitError:
                wait = 2 ** attempt
                print(f"\n  Rate limit hit — waiting {wait}s …")
                time.sleep(wait)
        else:
            raise RuntimeError("Exceeded max retries on OpenAI rate limit.")

    arr = np.vstack(all_embeddings)
    elapsed = time.perf_counter() - t0
    print(f"[OpenAI] Done in {elapsed:.1f}s  —  shape {arr.shape}")
    _save(cache_suffix, arr)
    return arr
