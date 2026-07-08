"""T2/M2: SemDeDup on frozen-Whisper encoder embeddings.

The semantic replacement for retired F2 (PROGRESS.md it.8): embeddings must
separate content, which mel-stat signatures could not. Needs stage-1 model
scoring to have filled `encoder_embedding` (scoring/model_scores.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _embeddings(df: pd.DataFrame) -> np.ndarray:
    """Encoder embeddings, mean-centered then L2-normalized (as in fft_prune)."""
    x = np.stack(df["encoder_embedding"].to_numpy())
    x = x - x.mean(axis=0)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def select_dedup_m2(df: pd.DataFrame, budget_hours: float | None = None,
                    tau: float = 0.95, n_clusters: int | None = None,
                    seed: int = 0) -> pd.DataFrame:
    """M2: k-means blocking + within-cluster cosine dedup on embeddings.

    tau=0.95 is provisional: max distinct-pair sim measured 0.922 on the 73-clip
    pilot (whisper-tiny). Proper validation = FLEURS with model scores, where
    same-sentence/different-speaker pairs give true semantic-dup positives.
    """
    from sklearn.cluster import KMeans

    from .fft_prune import _default_clusters

    if n_clusters is None:
        n_clusters = _default_clusters(len(df))
    x = _embeddings(df)
    labels = KMeans(n_clusters=min(n_clusters, len(df)), n_init=3, random_state=seed).fit_predict(x)
    keep_mask = np.zeros(len(df), dtype=bool)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        kept: list[int] = []
        for i in idx:
            if not kept or (x[i] @ x[kept].T).max() <= tau:
                kept.append(i)
        keep_mask[kept] = True
    out = df[keep_mask]
    if budget_hours is not None and out["duration_s"].sum() > budget_hours * 3600.0:
        from .baselines import select_random_stratified
        out = select_random_stratified(out, budget_hours, seed=seed)
    return out
