#!/usr/bin/env python
"""Stage 1: corpus → per-clip features parquet (FFT features; model scores optional).

Usage (HF streaming):
  python scripts/01_score.py --dataset librispeech --split train.100 \
      --out results/features_ls100.parquet [--limit 500]
Usage (local parquet with audio bytes — the reliable path; fetch shards with curl
from the Hub's refs/convert/parquet branch, keep them under data/):
  python scripts/01_score.py --from-parquet data/minds14_en-US_train.parquet \
      --audio-col audio --text-col english_transcription --out results/features_minds14.parquet

CPU-only. Writes shards incrementally so an interrupted run resumes cheaply.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audioprune.scoring.fft_features import SR, clip_features, hk_fingerprint, pack_fp  # noqa: E402

DATASETS = {
    # name -> (hf_path, config, audio_col, text_col)
    "librispeech": ("openslr/librispeech_asr", "clean", "audio", "text"),
    "common_voice": ("mozilla-foundation/common_voice_17_0", "en", "audio", "sentence"),
    # 73 real clips, a few MB — for validating features without big downloads
    "librispeech_dummy": ("hf-internal-testing/librispeech_asr_dummy", "clean", "audio", "text"),
}


def iter_hf_stream(dataset: str, split: str):
    """Yields (audio_array_16k, transcript, clip_id) from a streaming HF dataset."""
    from datasets import Audio, load_dataset  # deferred: heavy import

    hf_path, config, audio_col, text_col = DATASETS[dataset]
    ds = load_dataset(hf_path, config, split=split, streaming=True)
    ds = ds.cast_column(audio_col, Audio(sampling_rate=SR))
    for n, ex in enumerate(ds):
        clip_id = ex.get("id") or ex[audio_col].get("path") or str(n)
        yield ex[audio_col]["array"].astype(np.float32), ex.get(text_col), clip_id


def iter_local_parquet(paths: list[str], audio_col: str, text_col: str):
    """Yields (audio_array_16k, transcript, clip_id) from parquets with audio bytes."""
    import io

    import soundfile as sf

    from audioprune.scoring.fft_features import resample_to_sr

    for path in paths:
        df = pd.read_parquet(path)
        for n, row in df.iterrows():
            audio = row[audio_col]  # HF parquet convention: {"bytes": ..., "path": ...}
            y, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            yield resample_to_sr(y, sr), row.get(text_col), audio.get("path") or f"{path}:{n}"


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=DATASETS)
    src.add_argument("--from-parquet", nargs="+", help="local parquet file(s) with audio bytes")
    ap.add_argument("--split", default="train.100")
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="clip count cap for smoke tests")
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--model-scores", action="store_true",
                    help="also run frozen-Whisper scoring (~40x realtime CPU vs ~1600x FFT-only)")
    args = ap.parse_args()

    model = processor = None
    if args.model_scores:
        from audioprune.scoring.model_scores import load_whisper, score_clip

        model, processor = load_whisper()

    if args.from_parquet:
        source = iter_local_parquet(args.from_parquet, args.audio_col, args.text_col)
    else:
        source = iter_hf_stream(args.dataset, args.split)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, shards, n = [], [], 0
    for y, transcript, clip_id in source:
        feats = clip_features(y, transcript=transcript)
        feats["hk_fp"] = pack_fp(hk_fingerprint(y))
        if model is not None:
            feats.update(score_clip(model, processor, y, transcript or ""))
        feats["clip_id"] = clip_id
        feats["transcript"] = transcript
        rows.append(feats)
        n += 1
        if len(rows) >= args.shard_size:
            shards.append(pd.DataFrame(rows)); rows = []
            print(f"  scored {n} clips...", flush=True)
        if args.limit and n >= args.limit:
            break
    if rows:
        shards.append(pd.DataFrame(rows))
    df = pd.concat(shards, ignore_index=True)
    df.to_parquet(out)
    print(f"wrote {len(df)} clips ({df['duration_s'].sum() / 3600:.1f} h) -> {out}")


if __name__ == "__main__":
    main()
