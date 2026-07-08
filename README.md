# voiceglp

Can we train/fine-tune speech foundation models on a fraction of the audio data
(hours **and** bytes) with no WER loss — and can cheap **FFT/spectral methods**
(no GPU scoring pass) get most of the win?

| Doc | Contents |
|---|---|
| `RESEARCH_BRIEF.md` | Problem, gap analysis, why this is open |
| `PAPER_NOTES.md` | Prior-work deep dive (COWERAGE, DDP-ASR, SemDeDup, ...) |
| `PLAN.md` | Method matrix, pipeline design, milestones, risks |
| `PROGRESS.md` | Running log — read this first to see current state |

## Layout

```
audioprune/
  scoring/fft_features.py    STFT features: VAD, entropy, flatness, flux, mel signature (working)
  scoring/model_scores.py    frozen-whisper logprob/WER/embeddings (stub — GPU TODO)
  selection/baselines.py     B1 stratified random, B2 transcript dedup (working)
  selection/fft_prune.py     F1 density, F2b fingerprint dedup, F3 coverage (working;
                             F2 cosine retired — measured negative result, see PROGRESS)
  selection/hybrid.py        D1 dedup stack: text -> fingerprint (working)
  selection/difficulty.py    M1 keep-easy/hard/middle windows (working, needs model scores)
scripts/
  01_score.py … 04_eval.py   pipeline stages: score → select → train → eval
  smoke_test.py              synthetic end-to-end test, CPU, <1 min
configs/  results/           run configs, features parquet + manifests + WER csv
```

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas scikit-learn pyarrow
.venv/bin/python scripts/smoke_test.py          # should print "smoke test PASSED"
```

Real data — fetch parquet shards with curl (reliable without HF_TOKEN; the
`datasets` streaming client stalls unauthenticated), then score locally:

```bash
.venv/bin/pip install 'datasets[audio]<4' soundfile
curl -sL -o data/shard0.parquet \
  "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/refs%2Fconvert%2Fparquet/clean/train.100/0000.parquet"
.venv/bin/python scripts/01_score.py --from-parquet data/shard0.parquet \
  --text-col text --out results/features.parquet
```

Training/eval stages need a GPU box — see TODOs in `scripts/03_train.py` / `04_eval.py`.
