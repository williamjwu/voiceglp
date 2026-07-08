# Deep-Dive Paper Notes + Gap Analysis

July 2026. Companion to RESEARCH_BRIEF.md.

## A. Speech-specific prior work

### 1. COWERAGE — Representative Subset Selection for SSL Speech Recognition (arXiv:2203.09829, 2022)
**The paper closest to your project. Read first.**
- Setup: fine-tune wav2vec2-base/large + HuBERT-base (CTC) on TIMIT, LibriSpeech-10h, LJSpeech. Prune 10–90% of the labeled fine-tuning set.
- Metric: per-example training WER at an early epoch. Tested keep-hard (top-k), keep-easy (bottom-k), random, and their method.
- Method: stratified sampling — bucket examples by early-epoch WER into M buckets, sample uniformly from every bucket (coverage, not difficulty).
- Results: TIMIT @50% pruning — random 0.357 WER, top-k 0.392, bottom-k 0.411, COWERAGE 0.339. Up to 17% relative WER gain over other pruning at 90% pruning. Both pure keep-hard and keep-easy LOSE to random.
- Bonus findings: subsets transfer across models (wav2vec2→HuBERT) — "model-agnostic, dataset-specific". WER coverage implicitly gives phonemic diversity.
- Limitations they admit: tiny clean datasets, needs an initial training run to get WER scores, untested on noisy/multilingual data, pre-Whisper models.

### 2. DDP-ASR — Dynamic Data Pruning for ASR (arXiv:2406.18373, 2024)
- Setup: 243M conformer trained FROM SCRATCH on LibriSpeech-960h (+AV data). Dynamic: re-select which samples to use each epoch, nothing is permanently discarded.
- Methods: instance-wise (keep easy / hard / easy2hard curriculum by loss) + time-wise dropping inside clips (point vs chunk; chunk wins).
- Results: easy2hard @70% kept matches full data (2.53 vs 2.58 WER test-clean); + 70% chunk dropping → 1.6× speedup, no WER loss. Degrades past ~50% instance pruning. Keep-hard fails when train/test length distributions mismatch.
- Key difference from your project: dynamic ≠ a reusable static subset; doesn't reduce storage or benefit anyone else's training run. From-scratch conformer, not foundation-model fine-tuning.

### 3. Distil-Whisper (arXiv:2311.00430) & uDistil-Whisper (arXiv:2407.01257, NAACL 2025)
- Filtering for LABEL QUALITY during distillation, not for data efficiency.
- Distil-Whisper: discard sample if WER(pseudo-label vs ground truth) > ~10%.
- uDistil-Whisper: label-free proxies — proxy-model WER, entropy, mean token confidence, LM NLL, SONAR speech-text embedding similarity, PESQ on TTS-resynthesized audio. Proxy-WER AUC 0.82 at detecting bad samples; filtered ~27%; unsupervised ≈ supervised filtering; students beat teacher by 5–7 WER on Arabic/Swahili.
- Useful to you: a menu of cheap per-sample quality scores that need no ground-truth labels. But they only ever REMOVE BAD data; nobody uses these scores to remove REDUNDANT data.

### 4. DDFAD — Dataset Distillation for Audio (arXiv:2407.10446, 2024)
- Synthetic dataset distillation (MTT trajectory matching on FD-MFCC features + Griffin-Lim reconstruction). CLASSIFICATION ONLY (spoken digits, UrbanSound8K, RAVDESS).
- Results are weak outside toy data: UrbanSound 62.75% distilled vs 93.89% full. 60+ GB GPU memory.
- Lesson: synthetic distillation for variable-length seq2seq ASR is unsolved and heavy — STAY AWAY for an attainable paper; do coreset SELECTION instead.

## B. General pruning literature (ideas to import)

### 5. Beyond Neural Scaling Laws (arXiv:2206.14486, NeurIPS 2022 best paper)
- Theory + ImageNet: optimal strategy flips with budget — small budgets → keep EASY examples; large budgets → keep HARD. Good pruning can beat power-law scaling.
- Self-supervised prototype metric: k-means in embedding space, difficulty = distance to prototype; works ~as well as supervised metrics.
- Import for speech: (a) the budget-dependent flip has never been verified in ASR; (b) prototype difficulty from Whisper-encoder embeddings = label-free metric.

### 6. When Less is More — Data Pruning for LLM Pretraining (arXiv:2309.04564)
- 124M–1.5B LMs on CommonCrawl; metrics: reference-model perplexity, EL2N, memorization; keep top/middle/bottom at 10/30/50/70%.
- KEEP-MIDDLE by perplexity wins; training on 50% beat the full-data baseline (~1% better perplexity); perplexity > EL2N > memorization; bigger/cleaner reference models rank better; 55%-trained checkpoint ≈ fully-trained (scores are cheap).
- Import for speech: ASR analog of perplexity = frozen Whisper's per-sample mean log-prob or zero-shot WER. "Keep-middle by teacher score" is directly testable and NOT what COWERAGE tested (stratified-across-all ≠ middle-window).

### 7. SemDeDup (arXiv:2303.09540)
- Embed (CLIP/OPT) → k-means (50k clusters) → within-cluster cosine similarity > 1-ε ⇒ duplicates → keep one per group. O(n²/k).
- LAION: remove 50%, <0.5% ImageNet drop, 2× faster training, better OOD robustness. C4: 15% efficiency gain.
- Which duplicate you keep barely matters (their Table 3).
- Import for speech: NEVER applied to speech corpora. Common Voice is the ideal target — same sentences read by thousands of speakers = extreme semantic redundancy by construction. Embeddings: Whisper encoder mean-pool (audio), sentence-transformer on transcript (text), or concat.

## C. Scoop check (July 2026)
Searched for 2025–2026 static coreset/data-selection work for Whisper/foundation-ASR fine-tuning: found LoRA/hyperparameter work, model (layer) pruning for Whisper/SLAM-ASR, generic vision coreset papers (UNSEEN), LLM coreset (GRACE) — but no systematic selection-metric benchmark for speech foundation-model fine-tuning. Gap holds. Re-check arXiv before writing (search "data selection speech foundation model fine-tuning").

## D. The gaps, ranked by attainability

| # | Gap | Why it's open | Effort | Risk |
|---|-----|---------------|--------|------|
| G1 | Modern selection metrics (teacher log-prob/WER windows, embedding dedup, prototype difficulty) benchmarked for FINE-TUNING Whisper-class models on popular HF data, with OOD eval | COWERAGE = pre-Whisper SSL on tiny sets; DDP-ASR = dynamic + from-scratch; LLM findings untested in speech | Medium | Low — negative results still publishable as benchmark |
| G2 | SemDeDup for speech (Common Voice) | Never done; CV redundancy makes a positive result likely | Low (subset of G1) | Low |
| G3 | Does keep-easy/keep-hard flip with budget (Sorscher theory) hold in ASR? | Never tested | Low (falls out of G1 grid) | Low |
| G4 | Hours vs samples accounting: selection under a DURATION budget | Prior speech work counts examples; cost ∝ hours | Low (a design choice, sell as contribution) | Low |
| G5 | Bytes axis: WER vs stored-GB (codec re-encoding × pruning) | Nobody combines them | Low-Medium | Medium (may be "Opus is fine", still useful) |
| G6 | Low-resource language replication (one CV language) | COWERAGE said untested | Medium | Medium |
| G7 | Audio-LLM (Qwen2-Audio-style) data selection | Open but heavy | High | High — CUT |
| G8 | Synthetic dataset distillation for ASR | DDFAD can't do seq2seq | High | High — CUT |

## E. Narrowed paper proposal (attainable)

**Title shape:** "Less Audio, Same WER: A Benchmark of Data Selection Strategies for Fine-Tuning Speech Foundation Models"

**In scope:** static selection, one main dataset (Common Voice EN; LibriSpeech-100h for iteration), 2 models (wav2vec2-base CTC cheap tier; whisper-small headline), budgets {10,25,50}% of HOURS, methods: random / keep-easy / keep-hard / keep-middle (frozen-Whisper log-prob) / SemDeDup-audio / COWERAGE reimpl / (stretch) tokens-per-sec density. Eval: CV test + 2–3 ESB sets OOD (AMI, Earnings-22, TED-LIUM). 2 seeds on cheap tier, 1 on Whisper tier.

**Out of scope (say so in the paper):** dynamic pruning, synthetic distillation, audio-LLMs, multilingual (unless G6 stretch), intra-clip chunk dropping, from-scratch pretraining.

**Hypotheses:** H1 keep-middle > keep-hard/easy (transfer of LLM finding). H2 SemDeDup removes 30–50% of CV at ≤0.2 WER cost. H3 easy/hard flips with budget. H4 selected subsets transfer wav2vec2→Whisper (echo COWERAGE transferability).

**Run count:** ~6 methods × 3 budgets × 2 seeds ≈ 36 cheap runs (wav2vec2, LS-100h) + ~8–10 whisper-small confirmation runs + 1 scoring pass (whisper-small inference over corpus, cached to parquet). Single 24GB GPU: cheap run ~3–6h; whisper-small run ~1 GPU-day at 50% CV budget (or cap CV at a few hundred hours).

**Skills to learn along the way:** HF datasets audio streaming, HF Trainer/CTC fine-tuning, jiwer + Whisper text normalizer, faiss/k-means on embeddings, basic experiment tracking (wandb).
