"""T0 baselines. Every method must beat B1 or it doesn't matter (COWERAGE lesson)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_random_stratified(
    df: pd.DataFrame, budget_hours: float, seed: int = 0, n_strata: int = 10
) -> pd.DataFrame:
    """B1: random selection under an HOURS budget, stratified by clip duration.

    Duration-stratified so the subset's length distribution matches the corpus
    (plain random under an hours budget over-selects short clips when sampling
    is done per-sample; DDP-ASR showed length-distribution mismatch hurts).
    `df` needs columns: duration_s. Returns the selected rows.
    """
    rng = np.random.default_rng(seed)
    budget_s = budget_hours * 3600.0
    df = df.copy()
    df["stratum"] = pd.qcut(df["duration_s"], n_strata, labels=False, duplicates="drop")
    frac = budget_s / df["duration_s"].sum()
    picked = []
    for _, grp in df.groupby("stratum"):
        idx = rng.permutation(len(grp))
        cum = grp["duration_s"].to_numpy()[idx].cumsum()
        k = int(np.searchsorted(cum, frac * grp["duration_s"].sum()) + 1)
        picked.append(grp.iloc[idx[:k]])
    return pd.concat(picked).drop(columns="stratum")


def select_transcript_dedup(df: pd.DataFrame, budget_hours: float | None = None,
                            seed: int = 0) -> pd.DataFrame:
    """B2: keep one clip per unique normalized transcript, then fill to budget.

    The control that F2/M2 acoustic dedup must beat: on read corpora (Common
    Voice) most redundancy is literally the same sentence, and text dedup
    finds it for free. Normalization: lowercase, strip punctuation.
    """
    norm = (df["transcript"].fillna("").str.lower()
            .str.replace(r"[^\w\s]", "", regex=True).str.strip())
    out = df[~norm.duplicated()]
    if budget_hours is not None and out["duration_s"].sum() > budget_hours * 3600.0:
        out = select_random_stratified(out, budget_hours, seed=seed)
    return out
