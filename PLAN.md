# Experiment Plan — Slimming Audio Datasets, Same WER

Companion to `RESEARCH_BRIEF.md` (the "why") and `PAPER_NOTES.md` (prior work).
This file is the "how": the concrete run matrix and the code skeleton that implements it.

Design bias for this plan: **prefer cheap, model-free, FFT/spectral methods first**,
then escalate to model-based scoring only where spectral methods leave WER on the table.
Rationale: FFT-domain scores cost O(corpus) CPU (no GPU inference pass), are
reproducible, and are the least-studied axis in prior work (COWERAGE/DDP-ASR are all
loss/WER-based; nobody benchmarks signal-domain selection for ASR fine-tuning).

## 1. Method matrix

Tiers: **T0 baselines** → **T1 FFT/spectral (model-free)** → **T2 model-based** → **T3 hybrid**.
All budgets are in **hours** (not sample count). Budgets: {10, 25, 50, 100}%.

| ID | Method | Signal used | Cost | Module |
|----|--------|-------------|------|--------|
| B0 | Full data | — | — | — |
| B1 | Random, duration-stratified | none | ~0 | `selection/baselines.py` |
| B2 | Transcript-exact dedup (normalized text), random fill to budget | text | ~0 | `selection/baselines.py` |
| F1 | Spectral information density: keep clips with high spectral entropy rate + high speech ratio (FFT energy VAD) | STFT | CPU | `selection/fft_prune.py` |
| F2 | ~~Spectral cosine dedup~~ **Retired (measured negative result):** mel signatures cluster by speaker/channel, not content — removes 32% of unique LS clips. Kept as benchmark row only; signatures still used for F2b blocking + F3 coverage | STFT/mel | CPU | `selection/fft_prune.py` |
| F2b | Haitsma-Kalker binary fingerprint dedup (BER < 0.35, ±0.25 s alignment search). Validated: 98.6–100% recall on gain/noise/trim/lowpass variants, 0% FP | STFT | CPU | `selection/fft_prune.py` |
| F3 | Spectral coverage: k-means on log-mel signatures, uniform-per-cluster fill to budget. Proxy check: no advantage over B1 on LS shard0 (speaker/vocab coverage) — WER verdict pending | STFT/mel | CPU | `selection/fft_prune.py` |
| F4 | Intra-sample: FFT energy-VAD silence trimming (composable with all others). Measured: −31% hours spontaneous, −9% read speech | STFT | CPU | `scoring/fft_features.py` |
| M1 | Difficulty windows: frozen whisper-small per-sample avg log-prob; keep-easy / keep-hard / keep-middle | model | 1 GPU inference pass | `selection/difficulty.py` |
| M2 | SemDeDup-audio: Whisper-encoder embeddings → k-means → within-cluster near-dup removal | model | same pass as M1 | `selection/dedup.py` |
| D1 | Dedup stack: B2 text → F2b fingerprint → F2 cosine, fill to budget. Validated on planted-redundancy corpus: 59/60 dups removed, 0 originals lost | text+STFT | CPU | `selection/hybrid.py` |
| H1 | Hybrid: D1 dedup → M1 keep-middle window → F3 coverage fill | both | CPU + 1 pass | `selection/hybrid.py` (later) |
| C1 | Bytes axis (orthogonal): re-encode winning subset @ Opus → GB-vs-WER curve. Probed: default Opus = 5.2× smaller than FLAC (27 vs 141 kbps); bitrate sweep needs ffmpeg | codec (MDCT) | CPU | `scripts/reencode.py` (later) |

Key comparisons the matrix answers:
- **F1–F3 vs B1**: does signal-domain selection beat random at all? (novel either way)
- **F2 vs M2 vs B2**: does a $0 FFT fingerprint dedup recover most of embedding
  dedup's win — and do either beat plain text dedup on read corpora?
- **M1 windows**: does keep-middle > keep-easy/hard (LLM finding, untested in ASR)?
- **budget flip**: does keep-easy win at 10% and keep-hard at 50% (Sorscher theory)?

## 2. Datasets / models / eval (fixed recipe)

- Iterate: LibriSpeech `train-clean-100` (100h) → headline: Common Voice 17 EN.
- Cheap tier: `facebook/wav2vec2-base` + CTC head, 2 seeds. Headline: `openai/whisper-small`, 1 seed, winners only.
- Fixed training steps (not epochs) across budgets; report both interpretations once.
- Eval: in-domain test (LS test-clean/other, CV test) + OOD: AMI, Earnings-22, TED-LIUM (ESB). WER via jiwer + Whisper normalizer.
- Headline plots: WER vs training-hours; WER vs stored-GB; "hours to match full-data WER".

## 3. Pipeline (what the skeleton implements)

```
stage 1  score    scripts/01_score.py     corpus → per-clip features parquet
                                          (FFT: entropy, flatness, flux, speech_ratio,
                                           tokens/sec, log-mel signature;
                                           model: logprob, zero-shot WER, embedding)
stage 2  select   scripts/02_select.py    parquet + method + budget → subset manifest (json)
stage 3  train    scripts/03_train.py     manifest → fine-tuned checkpoint
stage 4  eval     scripts/04_eval.py      checkpoint → WER table row in results/
```

Everything downstream of stage 1 works from the cached parquet — scoring runs once per corpus.

## 4. Milestones

1. **W1**: stage-1 FFT scoring runs end-to-end on LS-100h (CPU); B0/B1 training repro (wav2vec2).
2. **W1–2**: F1/F2/F3 @ {10,25,50}% vs B1 — the first real plot. Decision gate: if F* ≈ B1 everywhere, FFT methods become the "cheap negative result" section and M1/M2 become the headline.
3. **W2–3**: model scoring pass (M1/M2) + windows + budget-flip grid.
4. **W3+**: hybrid H1, whisper-small confirmation runs, bytes axis C1, OOD eval.

## 5. Risks specific to the FFT emphasis

- Time-averaged log-mel signatures capture channel/speaker more than content → F2 may
  dedup "same mic" not "same sentence". Mitigation: concat text-transcript hash/embedding
  when transcripts exist; report both. (For Common Voice, transcript-based exact dedup is
  the trivially strong baseline — must be included or reviewers will ask.)
- Spectral entropy correlates with noise (noise is high-entropy) → F1 must combine with
  speech-ratio and not reward noisy clips. Check score-vs-WER scatter before trusting.
- COWERAGE precedent: random is strong. All plots show B1 with seed error bars.
