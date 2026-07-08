# Reducing Audio Training Datasets While Preserving Benchmark Accuracy

Research brief — July 2026

## 1. Problem statement

Given a large audio training dataset D (e.g., Common Voice, GigaSpeech) and a fixed
training recipe for an open-source foundation model, find a much smaller subset (or
compressed version) S such that a model trained/fine-tuned on S matches the benchmark
score (e.g., WER on speech-to-text benchmarks) of a model trained on all of D.

Two distinct "size" axes — audio is unusual in that both matter:
- **Hours** (drives training compute): select fewer / shorter clips.
- **Bytes** (drives storage/bandwidth): audio is ~low information density per GB;
  re-encoding (Opus 16kbps, neural codecs like EnCodec/Mimi) can shrink bytes without
  touching hours.

A third axis is **intra-sample**: silence, filler, and redundant chunks inside clips.

## 2. What already exists (gap analysis)

### Closest prior work in speech
| Work | What it did | Gap it leaves |
|---|---|---|
| COWERAGE (2022) — [arXiv:2203.09829](https://arxiv.org/abs/2203.09829) | Subset selection for fine-tuning wav2vec2/HuBERT (TIMIT, LibriSpeech). Found vision pruning metrics ≈ random; proposed coverage over early-epoch WER. | Pre-Whisper-era models, small datasets, no multi-domain eval. |
| DDP-ASR (2024) — [arXiv:2406.18373](https://arxiv.org/abs/2406.18373) | Dynamic data pruning during ASR training; prune 30% of instances + 30% chunk dropping with no loss. | Dynamic (during training), not a static reusable coreset; doesn't compare selection metrics. |
| Distil-Whisper (2023) — [arXiv:2311.00430](https://arxiv.org/abs/2311.00430) | WER-threshold (~10%) filtering of pseudo-labels for distillation. | Filtering for label quality, not for data efficiency/redundancy. |
| uDistil-Whisper (NAACL 2025) — [arXiv:2407.01257](https://arxiv.org/abs/2407.01257) | Label-free proxy filtering for distillation in low-data regimes. | Same: distillation-focused. |
| DDFAD (2024) — [arXiv:2407.10446](https://arxiv.org/html/2407.10446v1) | Dataset distillation (synthetic samples) for audio **classification**. | Not ASR/seq2seq; distillation for variable-length seq targets is open. |
| DITTO — [arXiv:2110.04908](https://arxiv.org/pdf/2110.04908) | Submodular targeted subset selection for accent adaptation. | Targeted adaptation, not general coreset. |

### Transferable ideas from the text/vision world (mostly untested on ASR)
- **Beyond Neural Scaling Laws** ([arXiv:2206.14486](https://arxiv.org/pdf/2206.14486)) — pruning metrics
  (EL2N, forgetting); key finding: keep-hard vs keep-easy depends on data budget; good pruning
  can beat power-law scaling.
- **When Less is More** ([arXiv:2309.04564](https://arxiv.org/pdf/2309.04564)) — perplexity-based pruning
  beat EL2N/memorization for LLM pretraining; ~30% of data sufficed. The ASR analog of perplexity
  is per-sample CTC loss or Whisper log-prob / zero-shot WER — cheap to compute and unstudied at scale.
- **SemDeDup** ([arXiv:2303.09540](https://ar5iv.labs.arxiv.org/html/2303.09540)) — embedding-space
  near-duplicate removal; 50% of LAION removed with ~no loss. Never systematically applied to
  speech corpora (crowd-read corpora like Common Voice have huge semantic redundancy: same
  sentences read by many speakers).

### The gap you can own
Nobody has published a systematic benchmark of modern pruning/selection metrics
(difficulty, diversity, dedup, information-density) for fine-tuning **current foundation
ASR models** (Whisper, wav2vec2/HuBERT/OWSM) on **popular Hugging Face datasets**, with
evaluation on the multi-domain **ESB / Open ASR leaderboard suite** — and nobody frames the
problem in audio-native units (WER vs. training-hours AND vs. stored-GB), nor combines
inter-sample pruning with intra-sample (silence/chunk) and codec compression.

## 3. Proposed experimental design

### Datasets (Hugging Face)
Start small → scale:
1. **LibriSpeech** (`openslr/librispeech_asr`, 960h / use 100h `train-clean-100` for iteration) — most-studied baseline.
2. **Common Voice 17 English** (`mozilla-foundation/common_voice_17_0`) — crowd-read, high redundancy → best case for dedup/pruning. ~GBs–100s of GB.
3. **GigaSpeech** (`speechcolab/gigaspeech`, subsets XS 10h → XL 10kh) — built-in size ladder, diverse domains.
Stretch: People's Speech (30kh), VoxPopuli.

### Models
- **Cheap iteration:** `facebook/wav2vec2-base` (or large) + CTC head fine-tuning — hours per run on one GPU.
- **Headline result:** `openai/whisper-small` (then medium) seq2seq fine-tuning — well-documented HF recipe.
- Stretch: a speech-LLM setup (SLAM-ASR style: frozen speech encoder + small LLM + linear projector) to connect to the "audio for LLMs" framing; Falcon3-Audio ([arXiv:2509.07526](https://arxiv.org/html/2509.07526v1)) shows competitive audio-LLMs are possible with <30k public hours, i.e., data efficiency is the live question in that space too.

### Selection methods to compare (the core matrix)
All budgets matched in **hours**, not sample count (audio-specific: sample cost ∝ duration).
1. **Random** (duration-stratified) — the baseline every method must beat.
2. **Difficulty-based:** zero-shot Whisper WER or avg log-prob per sample; CTC loss from a small model; EL2N analog. Test keep-easy / keep-hard / keep-middle windows.
3. **Diversity/coverage:** embed every clip (Whisper encoder mean-pool; optionally + speaker embedding + text embedding of transcript), then k-means/k-center coverage sampling and SemDeDup-style near-dup removal.
4. **Information density (novel, audio-native):** transcript tokens per audio second, speech/silence ratio (VAD), transcript n-gram novelty vs. already-selected set. Prune low-density clips.
5. **Hybrid pipeline:** dedup → difficulty window → diversity-stratified fill (mirrors D4/DoReMi-style pipelines from text).
6. **Intra-sample:** VAD silence trimming; DDP-ASR-style chunk dropping — composable with any of the above.
7. **Bytes axis (orthogonal):** re-encode selected subset at Opus 24/16/8 kbps and/or EnCodec tokens; measure WER impact. Deliverable: "GB → WER" curve, not just "hours → WER".

### Protocol
- Budgets: 10 / 25 / 50 / 100% of hours. Fixed recipe (same steps or same epochs — report both, this choice matters and papers get it wrong), ≥2 seeds at small scale.
- **Eval:** in-domain test set + out-of-domain via the ESB suite / [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) datasets (AMI, Earnings-22, TED-LIUM, SPGISpeech, VoxPopuli...) to check pruned sets don't overfit the domain. Report WER (jiwer, Whisper normalizer).
- Headline plots: WER vs. training hours (per method), WER vs. stored GB, and "hours needed to reach full-data WER".

### Compute reality check
- wav2vec2-base on ≤100h subsets: single 24GB GPU, a few hours/run → the full method matrix is feasible here.
- whisper-small on Common Voice EN subsets: ~1 GPU-day-scale per run → run only the winners from the cheap tier.
- Scoring passes (zero-shot WER, embeddings) are one-time inference over the corpus — cache to parquet.

## 4. Risks / known negative results
- COWERAGE found vision pruning metrics ≈ random for ASR fine-tuning — random is a strong
  baseline; your contribution can be positive even if the result is "dedup + density wins,
  difficulty doesn't," as long as the benchmark is rigorous.
- Fine-tuning strong foundation models may be insensitive to data choice at moderate budgets
  (everything works) — mitigate by including a harder setting (smaller model, OOD eval,
  low-resource language from Common Voice).
- Common Voice sentence overlap between train/test must be handled (known leakage issue).

## 5. Suggested first two weeks
1. Repro baseline: fine-tune wav2vec2-base on LibriSpeech train-clean-100 (full), eval test-clean/test-other.
2. Build the scoring pipeline: run whisper-small zero-shot over the corpus → per-sample {WER, logprob, duration, tokens/sec, VAD speech ratio, encoder embedding} → parquet.
3. Run random 25%/50% vs. one difficulty and one dedup method at the same budgets. That first plot tells you if the effect is real before scaling anything.

## Sources
- https://arxiv.org/abs/2203.09829 (COWERAGE)
- https://arxiv.org/abs/2406.18373 (DDP-ASR)
- https://arxiv.org/pdf/2110.04908 (DITTO)
- https://arxiv.org/html/2407.10446v1 (DDFAD)
- https://ar5iv.labs.arxiv.org/html/2311.00430 (Distil-Whisper)
- https://arxiv.org/abs/2407.01257 (uDistil-Whisper)
- https://arxiv.org/pdf/2206.14486 (Beyond neural scaling laws)
- https://arxiv.org/pdf/2309.04564 (When Less is More: data pruning for LLM pretraining)
- https://ar5iv.labs.arxiv.org/html/2303.09540 (SemDeDup)
- https://github.com/huggingface/open_asr_leaderboard (Open ASR Leaderboard)
- https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
- https://arxiv.org/html/2510.06961v1 (Open ASR Leaderboard paper)
- https://arxiv.org/html/2509.07526v1 (Falcon3-Audio, data-efficient audio-LLM)
- https://github.com/guang000/awesome-dataset-distillation
