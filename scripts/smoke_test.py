#!/usr/bin/env python
"""End-to-end smoke test on synthetic audio — no downloads, no GPU, <1 min.

Builds a fake corpus of 200 clips (speech-like AM tones, noise, near-duplicates,
half-silent clips), runs stage-1 FFT scoring + every T0/T1 selection method, and
asserts basic sanity (budgets respected, dedup drops the planted duplicates,
density prunes the silent clips).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Apple Accelerate BLAS raises spurious FP-flag RuntimeWarnings inside sklearn's
# kmeans matmuls (values are finite — verified); silence just those.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audioprune.scoring.fft_features import SR, clip_features, trim_silence
from audioprune.selection import baselines, fft_prune

rng = np.random.default_rng(0)


def fake_clip(kind: str, dur_s: float) -> np.ndarray:
    t = np.arange(int(dur_s * SR)) / SR
    if kind == "speech":  # AM tone stack ~ formant-ish, time-varying
        f0 = rng.uniform(100, 250)
        y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
        y *= 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 6) * t)  # syllabic AM
        return (y + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    if kind == "noise":
        return (0.3 * rng.standard_normal(len(t))).astype(np.float32)
    if kind == "half_silent":
        y = fake_clip("speech", dur_s)
        y[len(y) // 2:] = 0.001 * rng.standard_normal(len(y) - len(y) // 2)
        return y
    raise ValueError(kind)


def main() -> None:
    rows = []
    dup_base = fake_clip("speech", 4.0)
    for i in range(200):
        if i < 20:  # planted near-duplicates of one clip
            y, kind = dup_base + 0.005 * rng.standard_normal(len(dup_base)).astype(np.float32), "dup"
        elif i < 40:
            y, kind = fake_clip("noise", rng.uniform(2, 8)), "noise"
        elif i < 60:
            y, kind = fake_clip("half_silent", rng.uniform(2, 8)), "half_silent"
        else:
            y, kind = fake_clip("speech", rng.uniform(2, 8)), "speech"
        n_words = max(1, int(len(y) / SR * (0.5 if kind in ("half_silent", "noise") else 3)))
        feats = clip_features(y, transcript=" ".join(["word"] * n_words))
        feats.update(clip_id=f"clip{i:03d}", kind=kind)
        rows.append(feats)
    df = pd.DataFrame(rows)
    total_h = df["duration_s"].sum() / 3600
    budget = total_h * 0.5
    print(f"corpus: {len(df)} clips, {total_h * 60:.1f} min; budget = 50%")

    # F4 trim sanity
    y = fake_clip("half_silent", 6.0)
    assert len(trim_silence(y)) < 0.75 * len(y), "trim_silence should cut the silent half"

    for name, fn in [
        ("b1_random", lambda d: baselines.select_random_stratified(d, budget, seed=0)),
        ("f1_density", lambda d: fft_prune.select_density(d, budget)),
        ("f2_dedup", lambda d: fft_prune.select_dedup(d, budget, n_clusters=16, seed=0)),
        ("f3_coverage", lambda d: fft_prune.select_coverage(d, budget, n_clusters=16, seed=0)),
    ]:
        sub = fn(df)
        h = sub["duration_s"].sum() / 3600
        frac_by_kind = sub["kind"].value_counts(normalize=True).to_dict()
        assert h <= budget * 1.15, f"{name} blew the budget: {h:.2f}h > {budget:.2f}h"
        print(f"  {name:12s} {h / total_h * 100:5.1f}% of hours kept | kinds: "
              + ", ".join(f"{k}={v:.0%}" for k, v in sorted(frac_by_kind.items())))

    dedup = fft_prune.select_dedup(df, None, n_clusters=16, seed=0)
    n_dups_kept = (dedup["kind"] == "dup").sum()
    print(f"  f2 unbudgeted: kept {n_dups_kept}/20 planted duplicates (want ~1)")
    assert n_dups_kept <= 3, "dedup failed to remove planted near-duplicates"

    dense = fft_prune.select_density(df, budget)
    silent_frac = (dense["kind"] == "half_silent").mean()
    assert silent_frac < 0.20, f"density kept too many half-silent clips ({silent_frac:.0%})"
    print("smoke test PASSED")


if __name__ == "__main__":
    main()
