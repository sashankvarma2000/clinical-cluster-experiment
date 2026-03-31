"""
data_loader.py
--------------
Loads MTSamples medical transcription data, filters to the top-10 specialties,
and returns a clean DataFrame ready for embedding.
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mtsamples.csv"
TOP_N_SPECIALTIES = 10


def load_mtsamples(path: Path = DATA_PATH, top_n: int = TOP_N_SPECIALTIES) -> pd.DataFrame:
    """Load and preprocess the MTSamples dataset.

    Steps:
    1. Read CSV from *path*.
    2. Keep only `transcription`, `description`, and `medical_specialty` columns.
    3. Drop rows with null *description* or *medical_specialty*.
    4. Strip whitespace from *medical_specialty*.
    5. Keep only the *top_n* specialties by record count.
    6. Assign integer IDs to each specialty (``specialty_id``).
    7. Print class-distribution summary.

    Parameters
    ----------
    path : Path
        Location of ``mtsamples.csv``.
    top_n : int
        Number of most-frequent specialties to retain.

    Returns
    -------
    pd.DataFrame
        Columns: ``description``, ``specialty``, ``specialty_id``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"MTSamples CSV not found at {path}.\n"
            "Download it from https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions "
            "and place it at data/mtsamples.csv"
        )

    df = pd.read_csv(path)

    # Normalise column names (Kaggle CSV uses lowercase with spaces)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Required columns
    required = {"description", "medical_specialty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    df = df[["description", "medical_specialty"]].copy()
    df.dropna(subset=["description", "medical_specialty"], inplace=True)

    # Clean text
    df["description"] = df["description"].str.strip()
    df["medical_specialty"] = df["medical_specialty"].str.strip()

    # Drop empty strings after strip
    df = df[df["description"].str.len() > 0]

    # Filter to top-N specialties
    top_specialties = (
        df["medical_specialty"]
        .value_counts()
        .head(top_n)
        .index.tolist()
    )
    df = df[df["medical_specialty"].isin(top_specialties)].copy()

    # Integer label map (sorted for reproducibility)
    specialty_order = sorted(top_specialties)
    label_map = {name: idx for idx, name in enumerate(specialty_order)}
    df["specialty"] = df["medical_specialty"]
    df["specialty_id"] = df["specialty"].map(label_map)
    df.drop(columns=["medical_specialty"], inplace=True)

    df.reset_index(drop=True, inplace=True)

    # ── Print class distribution ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  MTSamples loaded  —  {len(df):,} records, {top_n} specialties")
    print(f"{'='*55}")
    dist = df["specialty"].value_counts()
    for spec, cnt in dist.items():
        bar = "█" * (cnt // 20)
        print(f"  {spec:<35} {cnt:>4}  {bar}")
    print(f"{'='*55}\n")

    return df[["description", "specialty", "specialty_id"]]
