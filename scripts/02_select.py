#!/usr/bin/env python
"""Stage 2: features parquet + method + budget → subset manifest (clip ids, json).

Usage:
  python scripts/02_select.py --features results/features_ls100.parquet \
      --method f1_density --budget-hours 25 --seed 0 --out results/manifests/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audioprune.selection import baselines, difficulty, fft_prune, hybrid, semantic  # noqa: E402

METHOD_NAMES = ["b1_random", "b2_text_dedup", "f1_density", "f2_dedup", "f2b_hk_dedup",
                "f3_coverage", "d1_dedup_stack", "m1_easy", "m1_hard", "m1_middle", "m2_dedup",
                "m3_wer_filter"]


def run_method(name: str, df: pd.DataFrame, budget_hours: float, seed: int) -> pd.DataFrame:
    """Runs the selection method `name` (see METHOD_NAMES) and returns the subset."""
    if name == "b1_random":
        return baselines.select_random_stratified(df, budget_hours, seed=seed)
    if name == "b2_text_dedup":
        return baselines.select_transcript_dedup(df, budget_hours, seed=seed)
    if name == "f1_density":
        return fft_prune.select_density(df, budget_hours)
    if name == "f2_dedup":
        return fft_prune.select_dedup(df, budget_hours, seed=seed)
    if name == "f2b_hk_dedup":
        return fft_prune.select_dedup_hk(df, budget_hours, seed=seed)
    if name == "f3_coverage":
        return fft_prune.select_coverage(df, budget_hours, seed=seed)
    if name == "d1_dedup_stack":
        return hybrid.select_dedup_stack(df, budget_hours, seed=seed)
    if name.startswith("m1_"):
        return difficulty.select_window(df, budget_hours, window=name.removeprefix("m1_"))
    if name == "m2_dedup":
        return semantic.select_dedup_m2(df, budget_hours, seed=seed)
    if name == "m3_wer_filter":
        return difficulty.select_wer_filter(df, budget_hours, seed=seed)
    raise ValueError(f"unknown method: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--method", choices=METHOD_NAMES, required=True)
    ap.add_argument("--budget-hours", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/manifests")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    sub = run_method(args.method, df, args.budget_hours, args.seed)
    manifest = {
        "method": args.method,
        "budget_hours": args.budget_hours,
        "seed": args.seed,
        "actual_hours": round(sub["duration_s"].sum() / 3600, 3),
        "n_clips": len(sub),
        "clip_ids": sub["clip_id"].tolist(),
    }
    out = Path(args.out) / f"{args.method}_h{args.budget_hours:g}_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest))
    print(f"{args.method} @ {args.budget_hours}h: {manifest['n_clips']} clips, "
          f"{manifest['actual_hours']}h actual -> {out}")


if __name__ == "__main__":
    main()
