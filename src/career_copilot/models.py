"""Retriever fit + persistence: TF-IDF over posting text.

`python -m career_copilot.models` builds and saves the artefact.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import load_all
from .features import posting_text

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def fit_retriever(postings: pd.DataFrame) -> dict:
    docs = [posting_text(r) for _, r in postings.iterrows()]
    vec = TfidfVectorizer(
        token_pattern=r"[A-Za-z][A-Za-z\-+#./ ]+",
        lowercase=True,
        ngram_range=(1, 2),
        max_features=4096,
        sublinear_tf=True,
    )
    X = vec.fit_transform(docs).astype(np.float32)
    job_ids = postings["job_id"].tolist()
    return dict(
        vec=vec,
        X=X,
        job_ids=job_ids,
    )


def save(obj, name: str = "rag_index.joblib") -> Path:
    path = MODEL_DIR / name
    joblib.dump(obj, path)
    return path


def load(name: str = "rag_index.joblib"):
    return joblib.load(MODEL_DIR / name)


def main():
    postings, _ = load_all()
    art = fit_retriever(postings)
    save(art)
    print(f"Saved retriever -> models/rag_index.joblib")
    print(f"  vocab size: {len(art['vec'].vocabulary_)}")
    print(f"  X shape: {art['X'].shape}")


if __name__ == "__main__":
    main()
