"""T2/M1: difficulty-window selection from frozen-teacher scores.

Needs stage-1 model scoring (scoring/model_scores.py) to have filled column
`teacher_logprob` (frozen-Whisper avg token log-prob; higher = easier).
Windows to sweep: keep-easy / keep-hard / keep-middle — tests both the
"When Less is More" keep-middle finding and the Sorscher budget flip.

Duration control: teacher_logprob correlates +0.64 with clip duration
(PROGRESS.md it.15 — longer clip, more context, higher mean token log-prob),
so windows are applied WITHIN duration strata. Otherwise keep-easy silently
selects long clips and the subset's length distribution drifts from the
corpus (DDP-ASR showed length mismatch hurts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = ("easy", "hard", "middle")


def _window_rank(scores: pd.Series, window: str) -> pd.Series:
    """Rank within one stratum, 0 = best fit for the window."""
    if window == "easy":
        key = -scores
    elif window == "hard":
        key = scores
    elif window == "middle":
        key = (scores - scores.median()).abs()
    else:
        raise ValueError(f"window must be one of {WINDOWS}")
    return key.rank(pct=True)


def select_window(df: pd.DataFrame, budget_hours: float, window: str = "middle",
                  score_col: str = "teacher_logprob", n_strata: int = 10) -> pd.DataFrame:
    """Keeps the best `window` fits per duration stratum until the hours budget."""
    strata = pd.qcut(df["duration_s"], n_strata, labels=False, duplicates="drop")
    rank = df.groupby(strata)[score_col].transform(lambda s: _window_rank(s, window))
    order = np.argsort(rank.to_numpy(), kind="stable")
    cum = df["duration_s"].to_numpy()[order].cumsum()
    k = int(np.searchsorted(cum, budget_hours * 3600.0) + 1)
    return df.iloc[order[:k]]


def select_wer_filter(df: pd.DataFrame, budget_hours: float | None = None,
                      max_wer: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """M3: label-quality filter — drop clips whose zero-shot WER exceeds max_wer.

    Distil-Whisper-style: a high teacher WER on clean-ish audio usually means a
    bad transcript, not hard audio. Orthogonal to difficulty windows (removes
    mislabeled data, not redundant or easy data). Near no-op on curated corpora
    like LibriSpeech; matters on crowd/web corpora. Fills to budget randomly.
    """
    out = df[df["zero_shot_wer"] <= max_wer]
    if budget_hours is not None and out["duration_s"].sum() > budget_hours * 3600.0:
        from .baselines import select_random_stratified
        out = select_random_stratified(out, budget_hours, seed=seed)
    return out
