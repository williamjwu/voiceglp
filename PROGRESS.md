# Progress log

Newest first. One entry per work session / loop iteration. Keep findings HERE, not in chat.

## 2026-07-07 — STOPPED at user request (mid-pipeline): 10/14 shards model-scored

Loop (iterations 17–39), pipeline, monitor, and cron all stopped; in-flight audio
shard purged (449 MB). Remaining on disk: results/ 240 MB (features + 10 model-score
shards + manifests + pilot), data/hf_cache 148 MB (whisper-tiny — kept so a resume
doesn't re-download; delete if space matters).

**To resume model scoring** (it picks up where it left off — the script skips
completed shards): `nohup ./results/model_scoring_pipeline.sh > results/model_scoring.log 2>&1 &`
Shards 10–13 remain (~40 min CPU). Then: concat results/model_scores_ls100_shard*.parquet
→ results/features_ls100_scored.parquet → cut m1_easy/hard/middle, m2_dedup,
m3_wer_filter manifests via 02_select.py (all methods implemented and scale-tested,
it.17–21). Then the GPU-day runbook (it.12/16 entries) applies unchanged.

## 2026-07-07 — iteration 21: m3 WER filter added; m1 windows verified at scale

- New `select_wer_filter` (m3): Distil-Whisper-style label-quality filter
  (drop zero-shot WER > 0.5), wired into 02_select as `m3_wer_filter`. On clean
  LS shard0 it drops 23/2039 clips (0.7% hours) — expected near-no-op; its real
  test is crowd/web corpora.
- **m1 windows at scale (shard0, 25% budget): duration control holds at
  0.99–1.01× corpus mean** (naive was 1.80×), with clean difficulty separation:
  easy/middle/hard logprob −1.22/−1.53/−2.02, zero-shot WER 3.0%/4.9%/11.3%.
- Pipeline: shard 1 scoring. Method inventory now complete for the GPU day:
  B1/B2, F1/F2b/F3/F4, D1, M1×3, M2, M3, C1 — every row implemented + sanity-checked.

## 2026-07-07 — iteration 20: shard-0 model scores validated at scale (n=2,039)

Pipeline on schedule (~10 min/shard → ETA ~2.5 h; shard 1 scoring). Shard-0 checks:
- zero-shot WER mean 6.1% / median 3.4% (whisper-tiny.en on clean read speech — sane);
  **34.8% of clips at exactly 0 WER** → WER is too quantized for window selection;
  use teacher_logprob (continuous) for m1 windows, keep WER for Distil-Whisper-style
  threshold filtering only.
- **Duration confound confirmed at scale: corr(logprob, duration) = +0.70** (pilot
  +0.64). The it.17 duration-stratified windows fix is mandatory, not optional.
- FFT↔model correlations SHRINK at scale (entropy −0.06, speech_ratio +0.11,
  tok/s +0.20): the cheap tier and the model tier measure nearly orthogonal things —
  strengthens the two-tier framing of the paper.
- logprob vs zero-shot WER: −0.49 — related, not redundant; both worth benchmarking.

## 2026-07-07 — iteration 19: manifest-overlap audit — drop B2 from the LS training grid

**Jaccard between 25h manifests:** b1_s0 vs b2 = **0.92** (LS has no text dups →
B2 = B1 with the same fill seed → SAME subset). Everything else ≈ 0.14-0.15 =
the theoretical overlap of independent 25% selections (p/(2−p) ≈ 0.143), incl.
b1_s0 vs b1_s1 (0.15 — seeds behave independently) and f1 vs f3 (0.15 — methods
pick genuinely different data).
**Grid decision: do NOT spend GPU runs on b2 manifests for LibriSpeech** (identical
to b1_s0 modulo 8% of clips); B2's WER row only makes sense on redundant corpora
(FLEURS: −43% hours, it.16). LS training arms: b1×2 seeds, f1, f3 (+ m1×3, m2
once model scores land).
LS model-scoring pipeline: shard 0 still scoring (242% CPU, healthy; LS clips
decode ~2× slower than pilot clips → revise ETA to ~5-7 h total).

## 2026-07-07 — iteration 18: M2 semantic dedup implemented + pre-validated

- New `selection/semantic.py`: `select_dedup_m2` (SemDeDup on whisper encoder
  embeddings, centered+normalized, k-means blocking). `m2_dedup` wired into 02_select.
- **Pre-validation on pilot embeddings (73 distinct clips, whisper-tiny):**
  distinct-pair cosine p50 0.003 / p95 0.661 / max 0.922 — embeddings separate
  content where mel signatures could not (p50 was 0.983). tau set 0.95 provisionally
  (max distinct 0.922); proper recall/FP validation needs FLEURS + model scores
  (same-sentence/different-speaker = true semantic-dup positives) — queued after
  the LS pipeline frees the CPU.
- LS model-scoring pipeline: still on shard 0 (~40× realtime as expected, ETA ~3 h
  total). Monitor next iterations; on completion: merge → m1/m2 manifests.

## 2026-07-07 — iteration 17 (loop resumed, no stop bound set): M1 fixed; full model scoring launched

- **difficulty.py duration-control implemented + validated:** windows now rank within
  duration strata. Naive keep-easy selected 1.80× corpus mean duration; stratified
  version 0.96–1.05× across all three windows while preserving difficulty separation
  (easy −2.16 / hard −2.77 / corpus −2.48 mean logprob on the pilot).
- `01_score.py --model-scores` flag added (FFT + frozen-Whisper in one pass).
- **Staged LS-100h model-scoring pipeline launched** (results/model_scoring_pipeline.sh,
  nohup pid in results/model_scoring.log): download shard → score (whisper-tiny.en,
  ~40× realtime CPU) → delete shard (purge rule: one shard on disk at a time).
  ETA ~3 h. Resumable (skips existing outputs). Output: model_scores_ls100_shard*.parquet
  → merge with features_ls100_full.parquet → m1/m2 manifests.
- HF_HOME pinned to data/hf_cache for the pipeline (folder rule). Note: the it.15
  pilot cached whisper-tiny to ~/.cache/huggingface before this was set (disclosed).

## 2026-07-07 — iteration 16 (LOOP ENDED, ALL JOBS STOPPED): D1 headline result on FLEURS

**D1 on FLEURS en_us (2,602 clips / 7.5 h; 1,476 unique sentences, 1,080 read by
2+ speakers): 43.4% of hours removed at zero model cost.** Text stage removed
exactly 1,126 clips = corpus size minus unique sentences (perfect attribution);
fingerprint −0 (correct: no re-encodes). With it.7/it.8 this completes the D1
characterization across corpus types:
| corpus | redundancy type | D1 removal |
|---|---|---|
| FLEURS (multi-reading read speech) | same sentence, other speakers | **43.4%** |
| MInDS-14 (spontaneous telephone) | few repeated phrases | 4% clips |
| LibriSpeech (book read speech) | none by construction | 0% (correctly) |

**Raw audio purged per user rule (~7.7 GB freed):** features live in results/
(126 MB total). Re-fetch shards via curl when training needs audio:
`https://huggingface.co/datasets/<repo>/resolve/refs%2Fconvert%2Fparquet/<config>/<split>/0000.parquet`
(repos used: openslr/librispeech_asr clean/train.100 ×14 shards; google/fleurs
en_us/train; PolyAI/minds14 en-US/train).

**Loop closed after iteration 16 (user instruction); cron deleted, tasks stopped.**
The ➜ RESUME runbook from iteration 12 still applies, with these updates:
- model_scores.py is DONE (it.15) — m1/m2 scoring can run on CPU (~3 h for LS-100h
  with whisper-tiny) or GPU (whisper-small).
- difficulty.py needs the duration-control fix before m1 manifests (it.15 finding 2).
- C1 GPU prediction on record (it.14): WER holds at ≥16 kbps Opus, breaks at 8.
- Only 03_train.py and 04_eval.py remain as stubs.

## 2026-07-07 — iteration 15: model_scores.py real + CPU-validated; two design findings

**`model_scores.py` is no longer a stub** — `load_whisper` + `score_clip` implemented
(teacher_logprob = −loss, zero-shot WER via tokenizer.normalize + jiwer, mean-pooled
encoder embedding). Pilot on 73 real clips (whisper-tiny.en, CPU,
results/model_scores_pilot.parquet): mean zero-shot WER 0.107, logprob p10/p50/p90
= −3.71/−2.26/−1.59, 384-d embeddings.

**Finding 1 — CPU model scoring is feasible:** 36.3× realtime with whisper-tiny.en →
LS-100h ≈ 3 CPU-hours. Proxy-quality m1/m2 scores don't strictly need the GPU
(uDistil-Whisper precedent for proxy-model filtering); GPU still wanted for
whisper-small scores + all training.

**Finding 2 — duration confound in M1:** corr(teacher_logprob, duration_s) = +0.64
(longer clip → more context → higher mean token logprob). Keep-easy windows would
silently select long clips. **M1 must be duration-controlled** (residualize logprob
on duration, or window within duration strata) — add to difficulty.py before real runs.
Also: FFT features correlate only weakly with model difficulty (|r| ≤ 0.43) →
tiers are complementary, not redundant (good for the paper's framing).

**it.13 still in flight:** FLEURS at 1.3/1.7 GB (slow LFS host). D1 attribution on
it = iteration 16, then full shutdown (user-confirmed).

## 2026-07-07 — iterations 13-14 (loop resumed, will stop after 16): C1 Opus sweep done

**Session resumed**; loop re-armed (job 883c6a34), hard stop after iteration 16.

**it.14 — C1 Opus bitrate sweep (scripts/opus_sweep.py, results/opus_sweep.csv):**
ffmpeg obtained IN-REPO via `pip install imageio-ffmpeg` (binary inside .venv —
no system install, folder rule respected). 40 LS clips, log-mel distortion as the
FFT-domain proxy for ASR impact:
| target | actual kbps | mel distortion | worst clip |
|---|---|---|---|
| 32k | 31.3 | 0.82 dB | 1.70 |
| 24k | 23.6 | 1.04 dB | 1.97 |
| 16k | 15.9 | 1.31 dB | 2.25 |
| 8k  |  8.1 | **3.54 dB** | 5.86 |
**Knee between 16k and 8k.** GPU-day prediction to validate: WER preserved at
≥16 kbps (= 8.8× bytes vs FLAC's 141 kbps), degrades at 8 kbps.

**it.13 (in flight):** FLEURS en_us train downloading (1.7 GB, slow LFS host) —
the by-construction-redundant corpus (sentences read by ~3 speakers) for the
first real-world D1 text-dedup measurement. transformers 4.57.6 + jiwer + torch
2.8 now in .venv for the it.15 whisper-tiny pilot (model_scores.py).

## 2026-07-07 — iteration 12 (SESSION PAUSED HERE): LS-100h complete, manifest grid ready

**Done:** full LS-100h scored — 28,539 clips / 100.6 h → `results/features_ls100_full.parquet`
(105 MB; the 6.6 GB audio in `data/` is only needed again at training time).
Manifest grid generated (budgets hit within 0.1%): b1_random(s0,s1) / b2_text_dedup /
f1_density / f3_coverage × {10, 25, 50}h → 15 manifests in `results/manifests/`.
Session /loop stopped and all background tasks cleaned up at user request.

## ➜ RESUME HERE (next session)

The CPU-side program is complete. Everything remaining needs a GPU (or ffmpeg):
1. **GPU session** (the critical path — all WER claims):
   a. Fill `audioprune/scoring/model_scores.py` TODO → run whisper-small pass over
      `features_ls100_full.parquet` clips → adds teacher_logprob/zero_shot_wer/embedding
      → then generate m1_easy/hard/middle + m2 manifests (methods already implemented).
   b. Fill `scripts/03_train.py` TODO → wav2vec2-base CTC on each manifest
      (fixed steps, configs to pin). ~15 runs × 3-6 h on a 24 GB GPU.
   c. Fill `scripts/04_eval.py` TODO → WER on test-clean/test-other + OOD (AMI,
      Earnings-22, TED-LIUM) → results/wer.csv → the WER-vs-hours plot.
2. **ffmpeg** (optional, CPU): Opus 24/16/8 kbps sweep for the GB-vs-WER curve.
3. Hypotheses to test against the it.1-12 measurements: does F1's density advantage
   on spontaneous speech translate to WER? Is F3's proxy null result confirmed by
   training? Budget-flip (easy/hard) on m1 windows?

State of every method + all measured findings: see iterations 1-11 below.
Key caches: features (results/*.parquet), raw audio shards (data/), tests
(scripts/smoke_test.py, dup_recall_test.py, dedup_stack_test.py — all passing).

## 2026-07-07 — iteration 11: full LS-100h acquired; scoring pass launched

- All 14 train.100 parquet shards curl-downloaded to `data/` (~6.6 GB, zero failures,
  ~40 s/shard unauthenticated — the curl path is fully reliable).
- `01_score.py --from-parquet` now accepts multiple shards (nargs="+").
- Full-corpus scoring (28.5k clips, 100 h) running → `results/features_ls100_full.parquet`.
- README quickstart rewritten around the curl+from-parquet path.
- Manifest-grid decision: generate b1(×2 seeds)/b2/f1/f3 × {10,25,50}% for LS-100h.
  f2b/d1 manifests deliberately SKIPPED on LibriSpeech — measured no-ops there
  (it.8: no re-encodes, unique sentences; ~40 min of BER for an output equal to
  b2). m1_* manifests wait for the GPU scoring pass.

## 2026-07-07 — iteration 10: F3 null result (proxy); H1-coverage skipped; cost asymmetry quantified

**F3 coverage vs B1 random (speaker/vocab coverage proxies, LS shard0, 3 seeds):**
No advantage — F3 slightly WORSE (10% budget: 2221 vs 2288 vocab; 18/19 vs 19/19
speakers). Explanation ties to it.8: signatures cluster by speaker/channel, so F3
≈ speaker-uniform sampling, and duration-stratified random already covers 19
well-mixed speakers. COWERAGE-consistent (random is strong). Caveats: small
speaker count, weak proxies — F3 could still matter on long-tail acoustic corpora,
but claims must wait for WER runs.
**Decision (minimalism rule): NOT building the H1 coverage-fill variant** — no
evidence it beats D1's existing stratified-random fill. D1+budget IS H1-lite.

**Cost asymmetry (the paper's economic argument), measured:** FFT scoring =
**1,642× realtime single-core** (8 ms/clip incl. decode + fingerprint).
Full-corpus scoring: LS-100h ≈ 0.1 core-h, CV17-en ≈ 1.5, GigaSpeech-XL ≈ 6.
The model-based tier (Whisper inference ~10-30× realtime on GPU) is ~3 orders of
magnitude more expensive. "What does the free tier buy you?" is the framing.

**Next:**
1. curl remaining train.100 shards (10 × ~470 MB) → full LS-100h features parquet
   → generate the complete manifest grid (b1/b2/f1/f2b/d1 × 10/25/50%) so a GPU
   session can start training immediately.
2. GPU session TODOs unchanged — still the critical path for all WER claims.

## 2026-07-07 — iteration 9: F4 + bytes axis measured — the no-training slimming stack quantified

**F4 (energy-VAD silence trimming), 300-clip samples, post-trim speech intact:**
- MInDS-14 (spontaneous telephone): **31.4% of hours removed**
- LS shard0 (curated read): **9.1% of hours removed**
Confirms the F1/F4 division of labor from it.8: spontaneous speech has huge
intra-clip slack; curated read speech still yields ~9%.

**C1 bytes axis (60 LS clips via soundfile — libsndfile 1.2.2 has OPUS!):**
- WAV/PCM16 250 kbps | FLAC 141 kbps | **Opus (default VBR) 27 kbps**
- Opus = 9.3× smaller than WAV, 5.2× smaller than FLAC (= how HF ships audio)
- F4 trimming stacks another 4–9% bytes on read speech (∝ its hours saving)
- Limitation: soundfile can't set Opus bitrate — the 24/16/8 kbps sweep from
  PLAN needs ffmpeg. Default-Opus point is already the big step though.

**No-training slimming stack, quantified so far (before any WER validation):**
| corpus type | hours lever | bytes lever |
|---|---|---|
| spontaneous (MInDS-like) | F4 −31%, F1@50% viable, D1 text −4% | Opus ~5× vs FLAC |
| curated read (LS-like) | F4 −9% (F1/D1 ≈ 0, correctly) | Opus ~5× vs FLAC |

WER-preservation validation of all of it = the GPU-session work (03/04 stubs).

**Next:**
1. GPU session: model_scores.py, 03_train.py, 04_eval.py TODOs — this is now THE
   blocker for every remaining claim (WER vs hours / WER vs GB curves).
2. Optional CPU: ffmpeg for Opus bitrate sweep; more LS shards for scale.

## 2026-07-07 — iteration 8 (final): F2 cosine dedup RETIRED (measured at scale); F1 characterized

**Headline finding — F2 signature-cosine dedup does not survive scale.** On LS
train.100 shard0 (2,039 clips, 7.3 h, 2.08M distinct pairs): 7,826 pairs > 0.95
cosine (p99.9 = 0.970), 81% same-speaker; as a dedup stage it would remove **647
unique clips (32%)**. Same failure as MInDS-14 (it.7) but on broadband read speech.
The it.3 "0% FP on LibriSpeech" was a 73-clip sample-size artifact (max sim 0.903
there vs 0.993 at 2k clips — same-speaker/chapter tail only appears at scale).
**Verdict: mel-stat signatures detect speaker+channel, not content.** Actions:
- D1 stack is now permanently text → fingerprint (cosine stage deleted).
- `select_dedup` (F2) kept only as a benchmark row, docstring carries the verdict.
- Signatures remain for what they're good at: F2b blocking, F3 coverage clustering.
- Paper framing: "cheap spectral stats cannot replace semantic embeddings for dedup;
  binary fingerprints CAN replace them for exact-recording dedup at zero model cost."
- Methodological note for the paper: validate selection metrics at ≥10³ clips —
  tiny-sample FP rates are misleading.
- D1 on LS shard0: text −0, fingerprint −0 (correct: unique book sentences, no
  re-encodes). Re-validated: dedup_stack_test + smoke test pass.

**F1 on read speech (predicted it.8-early, now measured):** speech_ratio p10–p90 =
0.73–0.92, tokens/sec 2.7–4.1 — narrow. F1's exploitable headroom on LibriSpeech is
~10–15% of hours (vs ~50% on silence-heavy MInDS-14). F1 is a *spontaneous-speech*
hours-compressor; on curated read corpora VAD trimming (F4) is the relevant lever.

## 2026-07-07 — iteration 8 (in flight): F1 face validity confirmed; LS-100h shard 0 scored

**F1 density audit on MInDS-14 (@50% hours budget):** kept clips average
speech_ratio 0.61 / 3.7 tok/s vs pruned 0.49 / 2.8; first-pruned clip is 58.5 s
containing nine words ("hello yes I would like to make a payment" + dead air).
Exactly the intended behavior — F1 is an hours-compressor for silence-heavy
spontaneous speech. Note: on read speech (LibriSpeech) silence is rare, so F1
should show much weaker effect there — measure on shard 0.

**Data at scale:** `data/ls_train100_shard0.parquet` downloaded via curl (470 MB,
seconds-fast, no auth needed). Scoring pass running.

## 2026-07-07 — iteration 7: first real-corpus D1 run; cosine-dedup FP found on telephone speech

**Data unblock (big):** the `datasets` streaming client stalls unauthenticated, but
plain `curl` against the Hub's `refs/convert/parquet` branch downloads fine (33 MB in
seconds). New pipeline path: fetch parquet shards with curl into `data/`, score via
`01_score.py --from-parquet` (new mode; decodes audio bytes with soundfile, linear
resample to 16 kHz via `resample_to_sr`). This also un-blocks LS-100h without HF_TOKEN.

**First real-corpus D1 run — MInDS-14 en-US (563 clips, 1.34 h, 8 kHz telephone):**
- text stage −22 (genuine crowd redundancy: "show me my account balance" ×4, etc.)
- fingerprint −0 (correct: no re-encoded copies exist in this corpus)
- cosine −24 → **audited: FALSE POSITIVES.** Distinct utterances at cosine 0.96–0.995,
  all same-caller/same-channel pairs. On narrowband audio the time-averaged signature
  fingerprints channel+voice, not content (the PLAN.md risk, now measured).
  p50 distinct sim −0.017 but fat tail: 44 pairs > 0.95.
- **Action: D1's cosine stage now opt-in (`use_cosine=False` default),** docstring
  explains when it's safe (broadband read speech: measured 0% FP on LibriSpeech).
  Honest negative result for the paper: signature-cosine dedup is corpus-sensitive;
  fingerprint (F2b) + text (B2) are the robust pair.
- Final: 563 → 541 clips (1.34 → 1.32 h) with the safe stack.

**Folder-containment rule (user):** everything stays inside the repo now — downloads
in `data/`, no /tmp. Set `HF_HOME=data/hf_cache` for any future HF-library runs.

**Next:**
1. curl a couple of LS train.100 parquet shards into `data/` → broadband corpus at
   scale; measure D1 + F2(cosine-on) there; validate F1 density ranking on it.
2. GPU session TODOs unchanged (model_scores.py, 03_train.py, 04_eval.py).

## 2026-07-07 — iteration 6: D1 dedup stack composed + validated; blocking bug fixed

**Done:**
- **Bug fix (would have silently no-opped on real corpora):** `select_dedup`,
  `select_dedup_hk`, `select_coverage` had fixed `n_clusters` defaults (1024/256) —
  any corpus smaller than that got ~1 clip per k-means block, so dedup compared
  nothing. Now `n_clusters=None` → auto ~50 clips/block (`_default_clusters`).
- New `selection/hybrid.py`: `select_dedup_stack` (D1) = B2 text → F2b fingerprint →
  F2 cosine → stratified fill to budget. Prints per-stage removal (stage attribution
  is a research output). CLI: `d1_dedup_stack`.
- New `scripts/dedup_stack_test.py`: planted-redundancy corpus from real speech —
  73 originals + 10 text-dups (distinct audio, copied transcript) + 50 re-encode
  variants (perturbed transcripts so they dodge the text stage).
- **Result: text stage removed exactly the 10 rereads, fingerprint 49/50 variants,
  0 distinct originals lost** (123 → 64 kept, ground truth 63).

**Finding — blocking miss:** the 1 escape (lowpass variant) had a signature damaged
enough to land in a different k-means block than its original, so BER never ran on
the pair. Classic SemDeDup blocking limitation. Mitigations if it matters at scale:
fewer/larger blocks for F2b, or block on fingerprint LSH instead of signatures.

**Next:**
1. Find a real redundant corpus slice that downloads unauthenticated (try
   `google/fleurs` en_us — sentences read by up to 3 speakers — or PolyAI/minds14)
   and measure D1's real-world removal rate + stage attribution.
2. LS-100h scoring still blocked on HF_TOKEN.
3. GPU session TODOs unchanged (model_scores.py, 03_train.py, 04_eval.py).

## 2026-07-07 — iteration 5: F2b Haitsma-Kalker fingerprint dedup — implemented, validated, wired in

**Done:**
- `fft_features.py`: `hk_fingerprint()` (sign of temporal-diff of band-energy-diff,
  mel bands 300–3000 Hz), `hk_ber()` (min bit-error rate over ±25-frame alignment
  search), `pack_fp`/`unpack_fp` (parquet storage as packed bytes).
- **Validation on real audio (dup_recall_test.py, now covers both detectors):**
  BER of true variants — gain 0.000, trim 0.000, lowpass 0.000, noise 0.244 mean.
  Distinct speech pairs sit at BER ≥ 0.469 (theory: unrelated ≈ 0.5 coin flips).
  At threshold 0.35 (Haitsma & Kalker's own): **gain/trim/lowpass 100%, noise 98.6%
  recall, 0% FP** — met the >90% target from it.4. Threshold has wide safety margin.
- Wired in: `01_score.py` stores `hk_fp` per clip; `select_dedup_hk()` in fft_prune
  (signature-kmeans blocking + within-block BER dedup); `f2b_hk_dedup` in `02_select.py`.
- Verified end-to-end: unbudgeted F2b keeps 73/73 distinct real clips; CLI manifest
  written; smoke test still passes.

**Method status now:** F2 (cheap cosine) for volume/trim re-uploads; F2b (fingerprint)
for noise/codec re-encodes; B2 (text) for same-sentence redundancy; M2 (embeddings,
GPU pending) for semantic near-dups. Together they cover the dup taxonomy.

**Next:**
1. Compose the dedup stack (B2 → F2b → F2) into a single "dedup pipeline" selection
   and measure combined removal rate on a real corpus slice.
2. LS-100h scoring still blocked on HF_TOKEN.
3. GPU session TODOs unchanged (model_scores.py, 03_train.py, 04_eval.py).

## 2026-07-07 — iteration 4: F2 robustness measured; signature hardened (gain/trim solved)

**Done:**
- New `scripts/dup_recall_test.py`: synthesizes realistic duplicate variants of the 73
  real clips (gain ×0.5, 30 dB-SNR noise, 0.2 s trim, 6 kHz lowpass) and measures F2
  recall + false-positive rate. Reusable for tuning tau per corpus.
- **Finding: original signature was fragile.** Recall@0.95 — gain 0%, noise 0%,
  lowpass 0% (sim −0.73!), trim 97%. Causes: gain = constant log-offset in every bin;
  lowpass = zeroed bands crash to log(EPS).
- **Fix (validated, now in `fft_features.py`):** floor mel power at clip_peak×1e-2
  (20 dB range) + scalar per-clip CMN. Result: **gain 100%, trim 100%, FP 0%**
  (max distinct sim 0.903 at tau 0.95). Floor sweep (40/30/20 dB) showed monotonic but
  plateauing gains for noise (32%) / lowpass (16%) — stat-signatures can't fix those.
- Smoke test re-passed; `features_dummy.parquet` regenerated with new signatures.

**Conclusion + next approach:** signature dedup is now solid for volume/trim re-uploads
but noise/codec-variant dups need a proper fingerprint: **Haitsma-Kalker binary
fingerprints** (sign of energy differences across band/time, Hamming match) — designed
for exactly this, ~20 lines. Added to PLAN.md as F2b.

**Next:**
1. Implement F2b Haitsma-Kalker fingerprint; validate on dup_recall_test (target:
   noise+lowpass recall >90% at FP ~0).
2. LS-100h scoring still blocked on HF_TOKEN (user not yet provided).
3. GPU session TODOs unchanged.

## 2026-07-07 — iteration 3: first real-speech features; F2 centering fix (key finding)

**Done:**
- Killed the stalled LS-100h streaming run (20 min, ~0 bytes — unauthenticated HF rate
  limit). Added `librispeech_dummy` (hf-internal-testing, 73 real clips) to `01_score.py`
  for download-free validation. Scored it → `results/features_dummy.parquet`.
- **KEY FINDING — raw mel signatures are unusable for dedup on real speech:** distinct
  utterances already have mean pairwise cosine **0.975** (p50 0.983) because log-mel
  speech spectra share a large common component (spectral tilt). Old tau=0.98 would have
  flagged 58% of distinct pairs as duplicates. **Fix: mean-center signatures before
  normalizing** → distinct pairs drop to mean −0.01 (max 0.907), true dups stay ~1.
  Implemented in `_signatures()`; default tau 0.98 → 0.95.
- Verified after fix: F2 keeps 73/73 distinct real clips (no false dups); synthetic smoke
  test still passes (19/20 planted dups removed); stage-2 CLI runs end-to-end on real
  parquet (`f1_density @ 0.05h` manifest written).
- Real-speech feature ranges (vs synthetic): entropy 3.3–4.4 bits (noise was 7.4),
  flatness ~0.01 (noise 0.56) → the F1 noise guard has huge separation. speech_ratio
  0.40–0.95, tokens/sec ~3.0. All sane.
- Pinned `datasets[audio]<4` (4.x needs torchcodec+FFmpeg; 3.x decodes via soundfile).

**Next:**
1. Full LS-100h (or train.100 subset) scoring needs HF_TOKEN — ask user or defer to GPU box.
2. Near-dup recall test on real audio: synthesize true dups from dummy clips
   (re-encode/noise/gain variants), measure F2 recall at tau=0.95.
3. GPU session TODOs unchanged (model_scores.py, 03_train.py, 04_eval.py).

## 2026-07-07 — iteration 2: style pass + B2 baseline; real-audio scoring in flight

**Done:**
- Style pass per user rules (readable > efficient, Google Python style, minimal files):
  deleted empty `audioprune/train|eval` packages; replaced all named lambdas with defs
  (`_hz_to_mel`/`_mel_to_hz`, `_zscore`); `02_select.py` lambda dict → `run_method()`
  dispatch. Smoke test re-passed with identical numbers (behavior-preserving).
- Added **B2 transcript-exact dedup** baseline (`baselines.select_transcript_dedup`,
  method `b2_text_dedup`): the free-text control F2/M2 acoustic dedup must beat. Unit-checked.
- Installed `datasets 4.5.0` + soundfile in .venv; started `01_score.py` on LibriSpeech.

**Findings / gotchas:**
- HF parquet conversion renames LibriSpeech splits: use `train.100`, NOT `train.clean.100`
  (available: test, train.100, train.360, validation).
- Unauthenticated HF Hub = rate-limited slow downloads; set HF_TOKEN before big scoring
  runs. 50-clip streaming smoke run took >10 min mostly downloading (still running,
  background task b1tej5311, output -> results/features_smoke.parquet).

**Next:**
1. When scoring run lands: inspect features on real speech (entropy/flatness/speech_ratio
   ranges vs synthetic; do F1 rankings look sane on real clips?).
2. Tune F2 tau on real signatures (synthetic tau=0.98 over-collapsed).
3. GPU session TODOs unchanged (model_scores.py, 03_train.py, 04_eval.py).

## 2026-07-07 — iteration 1: plan + skeleton + passing smoke test

**Done:**
- `PLAN.md`: method matrix (T0 baselines / T1 FFT / T2 model / T3 hybrid), 4-stage
  pipeline, milestones, FFT-specific risks. FFT tier is the novelty bet: nobody has
  benchmarked signal-domain selection for ASR fine-tuning (prior work is all loss/WER-based).
- `audioprune/` package skeleton + `scripts/01..04`. Working today (CPU, no downloads):
  - `scoring/fft_features.py` — STFT, energy VAD, spectral entropy/flatness/flux,
    log-mel signature, silence trimming. numpy-only.
  - `selection/baselines.py` (B1 duration-stratified random), `selection/fft_prune.py`
    (F1 density, F2 signature dedup, F3 coverage), `selection/difficulty.py` (M1 windows).
  - Stubs w/ TODOs (need GPU session): `scoring/model_scores.py`, `scripts/03_train.py`, `04_eval.py`.
- `.venv` (python3.9 system; numpy/pandas/sklearn/pyarrow). `scripts/smoke_test.py` PASSES:
  budgets respected; F2 removed 19/20 planted near-dups; F1 pruned all silent/noise clips.

**Findings / gotchas (verified, don't re-derive):**
- macOS Apple Accelerate BLAS raises **spurious** RuntimeWarnings in matmul (values
  finite — verified). Guarded in fft_features.py; filtered in smoke_test.py.
- Power spectra must be float64 before the mel matmul (float32 overflows).
- On synthetic tone corpora, mel signatures collapse (F2 at tau=0.98 dropped ~90%):
  synthetic clips are too self-similar. Expect saner behavior on real speech, but this
  confirms the PLAN.md risk that signatures capture channel/timbre — tau needs tuning
  on real data, and transcript-exact dedup must be the comparison baseline.
- Only python3.9 + pip on this machine (no uv, no 3.10+). Package targets 3.10+ but
  runs on 3.9 via `from __future__ import annotations`. Fine for CPU tier.

**Next (in order):**
1. Run `01_score.py --dataset librispeech --limit 500` for a real-audio smoke test
   (needs `pip install datasets[audio] soundfile` in .venv — big; maybe defer to GPU box).
2. Sanity-check F1 score vs. clip quality on those 500 real clips (score-vs-WER scatter
   needs model scores, but score-vs-speech_ratio/duration is checkable now).
3. Tune F2 tau on real data; add transcript-exact-dedup baseline to 02_select.py.
4. GPU session: fill model_scores.py, 03_train.py, 04_eval.py TODOs; B0/B1 repro on LS-100h.
