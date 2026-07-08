#!/usr/bin/env python
"""Measures F2 near-duplicate recall on real audio.

Takes real clips, synthesizes "in the wild" duplicate variants (gain, noise,
trim, lowpass — proxies for re-uploads and re-encodes), and reports, per
variant type, the cosine similarity to the original and recall at tau.
Also reports the false-positive rate over distinct-clip pairs.

Usage: python scripts/dup_recall_test.py [--tau 0.95]
Needs: `datasets[audio]<4` installed (uses the 73-clip dummy LibriSpeech set).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audioprune.scoring.fft_features import SR, clip_features, hk_ber, hk_fingerprint  # noqa: E402

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def make_variant(y: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """One synthetic duplicate: same recording as it might reappear in a corpus."""
    if kind == "gain":       # volume-normalized re-upload
        return 0.5 * y
    if kind == "noise":      # ~30 dB SNR background hiss
        return y + 10 ** (-30 / 20) * y.std() * rng.standard_normal(len(y)).astype(np.float32)
    if kind == "trim":       # 0.2 s shaved off the front
        return y[int(0.2 * SR):]
    if kind == "lowpass":    # re-encode proxy: zero FFT bins above 6 kHz
        spec = np.fft.rfft(y)
        spec[int(6000 * len(spec) / (SR / 2)):] = 0
        return np.fft.irfft(spec, n=len(y)).astype(np.float32)
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--ber", type=float, default=0.35, help="H-K duplicate threshold (0.35 = original paper)")
    args = ap.parse_args()

    from datasets import Audio, load_dataset

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                      split="validation").cast_column("audio", Audio(sampling_rate=SR))
    rng = np.random.default_rng(0)
    variant_kinds = ("gain", "noise", "trim", "lowpass")

    rows = []  # (label, signature); label = clip index or "idx:kind" for variants
    for i, ex in enumerate(ds):
        y = ex["audio"]["array"].astype(np.float32)
        rows.append((f"{i}", clip_features(y)["mel_signature"]))
        for kind in variant_kinds:
            rows.append((f"{i}:{kind}", clip_features(make_variant(y, kind, rng))["mel_signature"]))

    labels = [r[0] for r in rows]
    x = np.stack([r[1] for r in rows])
    x = x - x.mean(axis=0)                       # same centering as fft_prune._signatures
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    sim = x @ x.T

    print(f"tau = {args.tau}")
    orig_idx = {lab: i for i, lab in enumerate(labels) if ":" not in lab}
    for kind in variant_kinds:
        sims = np.array([sim[i, orig_idx[lab.split(":")[0]]]
                         for i, lab in enumerate(labels) if lab.endswith(f":{kind}")])
        print(f"  {kind:8s} sim to original: mean {sims.mean():.4f}  min {sims.min():.4f}"
              f"  | recall@tau {(sims > args.tau).mean():5.1%}")

    o = np.array(sorted(orig_idx.values()))
    distinct = sim[np.ix_(o, o)][np.triu_indices(len(o), k=1)]
    print(f"  false-positive rate (distinct pairs > tau): {(distinct > args.tau).mean():.2%}"
          f"  (max distinct sim {distinct.max():.3f})")

    print(f"\nF2b Haitsma-Kalker fingerprints (BER < {args.ber} = duplicate)")
    rng = np.random.default_rng(0)
    clips = [ex["audio"]["array"].astype(np.float32) for ex in ds]
    fps = [hk_fingerprint(y) for y in clips]
    for kind in variant_kinds:
        bers = np.array([hk_ber(hk_fingerprint(make_variant(y, kind, rng)), fp)
                         for y, fp in zip(clips, fps)])
        print(f"  {kind:8s} BER: mean {bers.mean():.3f}  max {bers.max():.3f}"
              f"  | recall {(bers < args.ber).mean():5.1%}")
    pairs = [(i, j) for i in range(len(fps)) for j in range(i + 1, len(fps))]
    sample = rng.choice(len(pairs), size=min(300, len(pairs)), replace=False)
    dist_bers = np.array([hk_ber(fps[pairs[k][0]], fps[pairs[k][1]]) for k in sample])
    print(f"  false-positive rate ({len(sample)} distinct pairs < {args.ber}): "
          f"{(dist_bers < args.ber).mean():.2%}  (min distinct BER {dist_bers.min():.3f})")


if __name__ == "__main__":
    main()
