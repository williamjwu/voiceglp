"""FFT/spectral per-clip features for data selection (model-free, CPU-only).

Everything here is derived from a single STFT pass per clip:
  - energy-VAD speech ratio (F4 trimming + F1 density)
  - spectral entropy rate, flatness, flux (F1 information density)
  - log-mel signature vector (F2 dedup fingerprint, F3 coverage clustering)

Only numpy is required. 16 kHz mono float32 input assumed (resample upstream).
"""
from __future__ import annotations

import numpy as np

SR = 16_000
N_FFT = 512          # 32 ms window
HOP = 160            # 10 ms hop
N_MELS = 40
EPS = 1e-10
SIG_FLOOR = 1e-2     # 20 dB dynamic range for signatures (robustness, see below)


def resample_to_sr(y: np.ndarray, sr: int) -> np.ndarray:
    """Linear resample to SR (16 kHz). Adequate for features; not for playback."""
    if sr == SR:
        return y
    n_out = int(len(y) * SR / sr)
    return np.interp(np.linspace(0, len(y) - 1, n_out), np.arange(len(y)), y).astype(np.float32)


def stft_mag(y: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Magnitude STFT, shape (frames, n_fft//2 + 1). Hann window, no padding."""
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    win = np.hanning(n_fft).astype(y.dtype)
    n_frames = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = y[idx] * win
    # float64: power spectra overflow float32 in the mel matmul downstream
    return np.abs(np.fft.rfft(frames, axis=1)).astype(np.float64)


def _hz_to_mel(f: float) -> float:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SR) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft//2 + 1)."""
    mel_pts = np.linspace(_hz_to_mel(0), _hz_to_mel(sr / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * _mel_to_hz(mel_pts) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l:
            fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


_MEL_FB = _mel_filterbank()

# Mel band center frequencies (Hz), for selecting the fingerprint band range.
_MEL_CENTERS_HZ = _mel_to_hz(
    np.linspace(_hz_to_mel(0), _hz_to_mel(SR / 2), N_MELS + 2)[1:-1]
)

# Fingerprint bits per frame at the default 300-3000 Hz band range.
HK_WIDTH = int(((_MEL_CENTERS_HZ >= 300.0) & (_MEL_CENTERS_HZ <= 3000.0)).sum()) - 1


def hk_fingerprint(y: np.ndarray, lo_hz: float = 300.0, hi_hz: float = 3000.0) -> np.ndarray:
    """Haitsma-Kalker binary fingerprint: bool array (frames-1, bands-1).

    Bit[n, m] = sign of the temporal difference of the band-energy difference:
        (E[n,m] - E[n,m+1]) - (E[n-1,m] - E[n-1,m+1]) > 0
    Signs of energy differences survive gain changes, noise, and codec smearing
    (Haitsma & Kalker 2002) — this is the F2b exact-recording dedup detector.
    Only bands in [lo_hz, hi_hz] are used: that range survives re-encoding.
    """
    power = stft_mag(y) ** 2
    bands = (_MEL_CENTERS_HZ >= lo_hz) & (_MEL_CENTERS_HZ <= hi_hz)
    with np.errstate(all="ignore"):
        e = np.log(power @ _MEL_FB[bands].T + EPS)
    band_diff = e[:, :-1] - e[:, 1:]
    return np.diff(band_diff, axis=0) > 0


def pack_fp(bits: np.ndarray) -> bytes:
    """Packs a fingerprint to bytes for parquet storage."""
    return np.packbits(bits).tobytes()


def unpack_fp(buf: bytes, width: int = None) -> np.ndarray:
    """Inverse of pack_fp. `width` defaults to the standard fingerprint width."""
    width = width or HK_WIDTH
    flat = np.unpackbits(np.frombuffer(buf, dtype=np.uint8)).astype(bool)
    return flat[: len(flat) // width * width].reshape(-1, width)


def hk_ber(a: np.ndarray, b: np.ndarray, max_offset: int = 25,
           min_overlap: int = 100) -> float:
    """Min bit-error rate between two fingerprints over ±max_offset frame shifts.

    The offset search absorbs leading trims/padding (25 frames = 0.25 s).
    Returns 1.0 if the overlap is shorter than min_overlap frames (1 s).
    Duplicates score well under 0.25; unrelated audio sits near 0.5 (coin flips).
    """
    best = 1.0
    for off in range(-max_offset, max_offset + 1):
        aa, bb = (a[off:], b) if off >= 0 else (a, b[-off:])
        n = min(len(aa), len(bb))
        if n >= min_overlap:
            best = min(best, float(np.mean(aa[:n] != bb[:n])))
    return best


def energy_vad(mag: np.ndarray, threshold_db: float = -35.0) -> np.ndarray:
    """Boolean speech mask per frame: energy above `threshold_db` rel. to clip peak."""
    energy_db = 20.0 * np.log10(mag.sum(axis=1) + EPS)
    return energy_db > (energy_db.max() + threshold_db)


def clip_features(y: np.ndarray, transcript: str | None = None) -> dict:
    """All FFT-derived scalar features + mel signature for one clip.

    Returns dict with:
      duration_s, speech_ratio, spectral_entropy (bits/frame, speech frames only),
      spectral_flatness, spectral_flux, mel_signature (2*N_MELS float32: mean||std
      of log-mel over speech frames), and tokens_per_sec if transcript given
      (whitespace tokens per SPEECH second — the density metric).
    """
    mag = stft_mag(y)
    power = mag**2
    speech = energy_vad(mag)
    sp = power[speech] if speech.any() else power

    # entropy of the normalized power spectrum, per frame, averaged
    p = sp / (sp.sum(axis=1, keepdims=True) + EPS)
    entropy = float(-(p * np.log2(p + EPS)).sum(axis=1).mean())

    # flatness: geometric / arithmetic mean of power spectrum (1.0 = white noise)
    flatness = float(
        np.exp(np.log(sp + EPS).mean(axis=1)).mean() / (sp.mean(axis=1).mean() + EPS)
    )

    # flux: mean L2 distance between consecutive normalized magnitude frames
    m = mag / (np.linalg.norm(mag, axis=1, keepdims=True) + EPS)
    flux = float(np.linalg.norm(np.diff(m, axis=0), axis=1).mean()) if len(m) > 1 else 0.0

    # Signature design (validated in scripts/dup_recall_test.py, PROGRESS it.4):
    #   - power floored at clip_peak * SIG_FLOOR: bounds the impact of quiet
    #     bands, where hiss and codec artifacts live
    #   - scalar CMN (subtract per-clip mean log-energy): exact gain invariance
    # Catches gain/trim duplicates at 100% recall, FP 0%; noise/codec variants
    # need a real fingerprint (Haitsma-Kalker) — see PLAN.md.
    # np.errstate: Apple Accelerate BLAS raises spurious FP-flag warnings on
    # matmul (values verified finite); harmless on other BLAS backends too
    with np.errstate(all="ignore"):
        mel = power @ _MEL_FB.T
        logmel = np.log(mel + mel.max() * SIG_FLOOR + EPS)
    lm = logmel[speech] if speech.any() else logmel
    lm = lm - lm.mean()
    signature = np.concatenate([lm.mean(axis=0), lm.std(axis=0)]).astype(np.float32)

    duration_s = len(y) / SR
    speech_ratio = float(speech.mean())
    out = {
        "duration_s": duration_s,
        "speech_s": duration_s * speech_ratio,
        "speech_ratio": speech_ratio,
        "spectral_entropy": entropy,
        "spectral_flatness": flatness,
        "spectral_flux": flux,
        "mel_signature": signature,
    }
    if transcript is not None:
        out["n_tokens"] = len(transcript.split())
        out["tokens_per_sec"] = out["n_tokens"] / max(out["speech_s"], 0.5)
    return out


def trim_silence(y: np.ndarray, threshold_db: float = -35.0, pad_frames: int = 5) -> np.ndarray:
    """F4: cut leading/trailing/internal silence runs longer than `pad_frames` (energy VAD)."""
    mag = stft_mag(y)
    speech = energy_vad(mag, threshold_db)
    # dilate mask by pad_frames so we keep short pauses (prosody) and boundaries
    kernel = np.ones(2 * pad_frames + 1, dtype=bool)
    keep = np.convolve(speech, kernel, mode="same") > 0
    sample_keep = np.repeat(keep, HOP)[: len(y)]
    if len(sample_keep) < len(y):
        sample_keep = np.pad(sample_keep, (0, len(y) - len(sample_keep)), constant_values=True)
    return y[sample_keep]
