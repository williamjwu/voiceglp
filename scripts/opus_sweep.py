#!/usr/bin/env python
"""C1 bytes axis: Opus bitrate vs size and spectral distortion (no GPU proxy).

For a sample of clips, round-trips audio through Opus at several bitrates and
reports actual kbps plus log-mel distortion vs the original — an FFT-domain
proxy for ASR impact (the real WER-vs-kbps curve needs the GPU session).
Writes results/opus_sweep.csv.

Usage: python scripts/opus_sweep.py [--clips 40] [--bitrates 32 24 16 8]
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audioprune.scoring.fft_features import SR, _MEL_FB, resample_to_sr, stft_mag  # noqa: E402

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def opus_roundtrip(y: np.ndarray, kbps: int, workdir: Path) -> tuple[int, np.ndarray]:
    """Encodes y at `kbps`, decodes back. Returns (encoded_bytes, decoded_audio)."""
    wav_in, opus, wav_out = (workdir / f"c.{ext}" for ext in ("in.wav", "opus", "out.wav"))
    sf.write(wav_in, y, SR)
    quiet = ["-hide_banner", "-loglevel", "error", "-y"]
    subprocess.run([ffmpeg_exe(), *quiet, "-i", str(wav_in),
                    "-c:a", "libopus", "-b:a", f"{kbps}k", str(opus)], check=True)
    subprocess.run([ffmpeg_exe(), *quiet, "-i", str(opus), "-ar", str(SR), str(wav_out)],
                   check=True)
    dec, _ = sf.read(wav_out, dtype="float32")
    return opus.stat().st_size, dec


def logmel_db(y: np.ndarray) -> np.ndarray:
    mel = stft_mag(y) ** 2 @ _MEL_FB.T
    return 10.0 * np.log10(mel + mel.max() * 1e-6)


def mel_distortion_db(orig: np.ndarray, dec: np.ndarray) -> float:
    """Mean |log-mel difference| in dB, skipping the codec's priming frames."""
    a, b = logmel_db(orig), logmel_db(dec)
    n = min(len(a), len(b))
    return float(np.abs(a[10:n] - b[10:n]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=40)
    ap.add_argument("--bitrates", type=int, nargs="+", default=[32, 24, 16, 8])
    ap.add_argument("--features", default="data/ls_train100_shard0.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.features).sample(args.clips, random_state=0)
    clips = []
    for a in df["audio"]:
        y, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        clips.append(resample_to_sr(y if y.ndim == 1 else y.mean(axis=1), sr))
    secs = sum(len(y) for y in clips) / SR

    rows = []
    with tempfile.TemporaryDirectory(dir="results") as tmp:
        for kbps in args.bitrates:
            total_bytes, dists = 0, []
            for y in clips:
                nbytes, dec = opus_roundtrip(y, kbps, Path(tmp))
                total_bytes += nbytes
                dists.append(mel_distortion_db(y, dec))
            rows.append({"target_kbps": kbps, "actual_kbps": total_bytes * 8 / secs / 1000,
                         "mel_distortion_db": np.mean(dists), "worst_clip_db": np.max(dists)})
            print(f"  opus {kbps:>2}k: actual {rows[-1]['actual_kbps']:5.1f} kbps, "
                  f"mel distortion {rows[-1]['mel_distortion_db']:.2f} dB "
                  f"(worst clip {rows[-1]['worst_clip_db']:.2f})")
    out = Path("results/opus_sweep.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({args.clips} clips, {secs/60:.1f} min audio)")


if __name__ == "__main__":
    main()
