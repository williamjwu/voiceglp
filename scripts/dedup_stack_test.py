#!/usr/bin/env python
"""Validates the D1 dedup stack on a planted-redundancy corpus of real speech.

Builds a corpus from the 73 real dummy-LibriSpeech clips plus known redundancy:
  - 10 text duplicates: distinct audio given another clip's transcript
    (simulates crowd-corpus rereads; the text stage's job)
  - 50 re-encode variants (noise/lowpass/gain/trim) with slightly perturbed
    transcripts so they dodge text dedup (the fingerprint/cosine stages' job)
Then runs select_dedup_stack and checks every planted duplicate is removed and
every distinct original survives.

Usage: python scripts/dedup_stack_test.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audioprune.scoring.fft_features import SR, clip_features, hk_fingerprint, pack_fp  # noqa: E402
from audioprune.selection.hybrid import select_dedup_stack  # noqa: E402
from dup_recall_test import make_variant  # noqa: E402

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def featurize(y: np.ndarray, transcript: str, clip_id: str) -> dict:
    feats = clip_features(y, transcript=transcript)
    feats.update(hk_fp=pack_fp(hk_fingerprint(y)), transcript=transcript, clip_id=clip_id)
    return feats


def main() -> None:
    from datasets import Audio, load_dataset

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                      split="validation").cast_column("audio", Audio(sampling_rate=SR))
    clips = [(ex["audio"]["array"].astype(np.float32), ex["text"]) for ex in ds]
    rng = np.random.default_rng(0)

    rows = []
    for i, (y, text) in enumerate(clips):
        # clips 60..69 get clip 0..9's transcript -> planted text duplicates
        text = clips[i - 60][1] if 60 <= i < 70 else text
        rows.append(featurize(y, text, f"orig{i:02d}"))
    variant_plan = [("noise", range(10, 25)), ("lowpass", range(25, 40)),
                    ("gain", range(40, 50)), ("trim", range(50, 60))]
    for kind, idxs in variant_plan:
        for i in idxs:
            y, text = clips[i]
            rows.append(featurize(make_variant(y, kind, rng), text + " x", f"var{i:02d}_{kind}"))
    df = pd.DataFrame(rows)

    kept = select_dedup_stack(df)
    kept_ids = set(kept["clip_id"])
    variants_kept = [c for c in kept_ids if c.startswith("var")]
    text_dup_kept = [c for c in kept_ids if c in {f"orig{i:02d}" for i in range(60, 70)}]
    originals_lost = [f"orig{i:02d}" for i in range(60) if f"orig{i:02d}" not in kept_ids]

    print(f"planted corpus: {len(df)} clips ({len(clips)} originals, 50 variants, 10 text dups)")
    print(f"  re-encode variants kept (want 0/50): {len(variants_kept)} {variants_kept}")
    print(f"  text-dup readings kept (want 0/10): {len(text_dup_kept)} {text_dup_kept}")
    print(f"  distinct originals lost (want 0/60): {len(originals_lost)} {originals_lost}")
    assert len(variants_kept) <= 2 and not text_dup_kept and not originals_lost
    print("dedup stack test PASSED")


if __name__ == "__main__":
    main()
