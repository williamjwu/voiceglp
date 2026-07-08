"""T1: FFT/spectral selection methods (F1 density, F2 dedup, F3 coverage).

All operate on the stage-1 features parquet — no audio access, no GPU.
Common df columns: duration_s, speech_ratio, spectral_entropy, spectral_flatness,
tokens_per_sec, mel_signature (2*N_MELS float32 array per row).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _take_by_hours(df: pd.DataFrame, order: np.ndarray, budget_hours: float) -> pd.DataFrame:
    """Take rows in `order` (best first) until the hours budget is filled."""
    cum = df["duration_s"].to_numpy()[order].cumsum()
    k = int(np.searchsorted(cum, budget_hours * 3600.0) + 1)
    return df.iloc[order[:k]]


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-9)


def select_density(df: pd.DataFrame, budget_hours: float, w_noise_guard: float = 1.0) -> pd.DataFrame:
    """F1: keep the most information-dense clips.

    Score = z(spectral_entropy) + z(speech_ratio) + z(tokens_per_sec)
            - w_noise_guard * z(spectral_flatness)
    The flatness term is the noise guard: white noise maximizes entropy but also
    flatness, so subtracting flatness keeps "dense speech" ranked above "loud noise".
    """
    score = (
        _zscore(df["spectral_entropy"]) + _zscore(df["speech_ratio"])
        + _zscore(df["tokens_per_sec"]) - w_noise_guard * _zscore(df["spectral_flatness"])
    )
    return _take_by_hours(df, np.argsort(-score.to_numpy()), budget_hours)


def _signatures(df: pd.DataFrame) -> np.ndarray:
    """Mel signatures, mean-centered then L2-normalized.

    Centering matters: raw log-mel speech spectra share a large common component
    (spectral tilt), so DISTINCT utterances already have ~0.97 raw cosine.
    After centering, distinct pairs land near 0 and true near-dups stay near 1
    (measured on real LibriSpeech clips — see PROGRESS.md 2026-07-07 it.3).
    """
    x = np.stack(df["mel_signature"].to_numpy())
    x = x - x.mean(axis=0)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _default_clusters(n_rows: int) -> int:
    """~50 clips per k-means block (SemDeDup-style); at least 1."""
    return max(1, n_rows // 50)


def select_dedup(df: pd.DataFrame, budget_hours: float | None = None,
                 tau: float = 0.95, n_clusters: int | None = None, seed: int = 0) -> pd.DataFrame:
    """F2: SemDeDup-style near-duplicate removal on mel signatures.

    NEGATIVE RESULT — kept only as a benchmark row. Measured at scale, mel
    signatures cluster by speaker/channel, not content: on LS shard0 this
    removes 32% of unique clips (same speaker, different sentences), on
    MInDS-14 it removes same-caller clips (PROGRESS.md it.7-8). Use B2 (text),
    F2b (fingerprint), or M2 (semantic embeddings) for real dedup.

    k-means the L2-normalized signatures, then within each cluster greedily drop
    any clip with cosine sim > tau to an already-kept clip (O(n^2/k), like SemDeDup).
    If budget_hours is set, fill remaining budget randomly from the deduped pool.
    """
    from sklearn.cluster import KMeans  # deferred import; sklearn only needed here

    if n_clusters is None:
        n_clusters = _default_clusters(len(df))
    x = _signatures(df)
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


def select_dedup_hk(df: pd.DataFrame, budget_hours: float | None = None,
                    ber: float = 0.35, n_clusters: int | None = None, seed: int = 0) -> pd.DataFrame:
    """F2b: exact-recording dedup via Haitsma-Kalker fingerprints.

    Signature k-means gives candidate blocks (BER matching all pairs is too
    slow); within each block, drop any clip whose min BER to an already-kept
    clip is below `ber`. Validated on real audio (dup_recall_test.py):
    98.6-100% recall on gain/noise/trim/lowpass variants, 0% false positives
    (unrelated speech sits at BER ~0.5). Needs the `hk_fp` parquet column.
    """
    from sklearn.cluster import KMeans

    from ..scoring.fft_features import hk_ber, unpack_fp

    if n_clusters is None:
        n_clusters = _default_clusters(len(df))
    x = _signatures(df)
    labels = KMeans(n_clusters=min(n_clusters, len(df)), n_init=3, random_state=seed).fit_predict(x)
    fps = [unpack_fp(buf) for buf in df["hk_fp"]]
    keep_mask = np.zeros(len(df), dtype=bool)
    for c in np.unique(labels):
        kept: list[int] = []
        for i in np.where(labels == c)[0]:
            if all(hk_ber(fps[i], fps[j]) >= ber for j in kept):
                kept.append(i)
        keep_mask[kept] = True
    out = df[keep_mask]
    if budget_hours is not None and out["duration_s"].sum() > budget_hours * 3600.0:
        from .baselines import select_random_stratified
        out = select_random_stratified(out, budget_hours, seed=seed)
    return out


def select_coverage(df: pd.DataFrame, budget_hours: float,
                    n_clusters: int | None = None, seed: int = 0) -> pd.DataFrame:
    """F3: coverage sampling — uniform hours per k-means cluster of mel signatures.

    The spectral analog of COWERAGE's WER-bucket stratification: guarantees the
    subset spans acoustic conditions instead of concentrating on the head.
    """
    from sklearn.cluster import KMeans

    if n_clusters is None:
        n_clusters = _default_clusters(len(df))
    rng = np.random.default_rng(seed)
    x = _signatures(df)
    labels = KMeans(n_clusters=min(n_clusters, len(df)), n_init=3, random_state=seed).fit_predict(x)
    per_cluster_s = budget_hours * 3600.0 / len(np.unique(labels))
    picked = []
    leftover = 0.0
    for c in rng.permutation(np.unique(labels)):
        grp = df[labels == c]
        idx = rng.permutation(len(grp))
        cum = grp["duration_s"].to_numpy()[idx].cumsum()
        quota = per_cluster_s + leftover
        k = int(np.searchsorted(cum, quota) + 1)
        k = min(k, len(grp))
        picked.append(grp.iloc[idx[:k]])
        leftover = quota - grp["duration_s"].to_numpy()[idx[:k]].sum()  # small clusters donate
    return pd.concat(picked)
