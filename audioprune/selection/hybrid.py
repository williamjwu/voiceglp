"""Composed selection pipelines (PLAN.md tier T3)."""
from __future__ import annotations

import pandas as pd

from . import baselines, fft_prune


def select_dedup_stack(df: pd.DataFrame, budget_hours: float | None = None,
                       seed: int = 0, ber: float = 0.35) -> pd.DataFrame:
    """D1: dedup stack — text, then fingerprint; fill to budget.

    Stage order is by duplicate type, cheapest detector first:
      1. B2 text: same normalized transcript (rereads, crowd-corpus redundancy)
      2. F2b fingerprint: same recording re-encoded (noise/codec/gain/trim)
    Signature-cosine dedup (F2) is deliberately NOT a stage: measured at scale
    it removes same-speaker clips with different content — telephone speech
    (MInDS-14, it.7) AND broadband read speech (LS shard0: would drop 32% of
    unique clips, it.8). Semantic near-dup removal needs real embeddings (M2).
    Prints per-stage removal counts — that attribution is a research output
    (which redundancy type dominates a given corpus).
    """
    n0 = len(df)
    out = baselines.select_transcript_dedup(df)
    n1 = len(out)
    out = fft_prune.select_dedup_hk(out, None, ber=ber, seed=seed)
    print(f"dedup stack: {n0} clips -> text -{n0 - n1} -> fingerprint -{n1 - len(out)}"
          f" => kept {len(out)}")
    if budget_hours is not None and out["duration_s"].sum() > budget_hours * 3600.0:
        out = baselines.select_random_stratified(out, budget_hours, seed=seed)
    return out
