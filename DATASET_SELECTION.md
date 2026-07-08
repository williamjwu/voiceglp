# Dataset selection guidelines

Task: pick **5 candidate voice datasets from Hugging Face** for the slimming
benchmark. Read `README.md` first, then `PROGRESS.md` (top entry + it.16 table)
for what's already measured.

## How this fits the project

The experiment per dataset: train a fixed model on the full data (**B0
baseline**) and on random subsets (**B1**, the bar every method must beat),
then train the *same* model+recipe on slimmed subsets (10/25/50% of hours,
manifests from `scripts/02_select.py`) and show WER stays within noise of
baseline. The model doesn't need to be SOTA — it needs to be *fixed* so all
comparisons are deltas (cheap tier: wav2vec2-base CTC; headline: whisper-small).

**Key measured fact (PROGRESS it.7/8/16): which slimming methods work is
determined by the corpus's redundancy type.** Text dedup removes 43% of FLEURS
hours but 0% of LibriSpeech; silence trimming saves 31% on spontaneous telephone
speech but 9% on read audiobooks. The 5 candidates must therefore SPAN
archetypes, not just be 5 good corpora.

## Required archetype coverage (aim for one each)

| # | Archetype | Why needed | Already-scored example |
|---|---|---|---|
| 1 | Curated read speech (control) | Methods should correctly remove ~nothing → tests false-positive behavior | LibriSpeech-100h (done, keep as-is) |
| 2 | Crowd/multi-reading read speech | Text + semantic dedup should shine (same sentences, many speakers) | FLEURS en_us (features done; −43% via text dedup) |
| 3 | Spontaneous / silence-heavy | VAD trimming + density pruning should shine | MInDS-14 (features done; −31% hours via trimming) |
| 4 | Web-scraped / multi-source | Fingerprint dedup (re-encodes) + quality filter should shine | none yet — **highest-value pick** |
| 5 | Your call | e.g. low-resource language, noisy field recordings, or a bigger version of 2/3 | — |

Candidates 1–3 may reuse the already-scored corpora (don't re-do work), but
propose an upgrade if you find a strictly better fit.

## Hard requirements (reject if any fail)

1. **License** permits research use and redistribution of derived subsets
   (CC-BY/CC0/Apache-like; check the HF dataset card, note it explicitly).
2. **Transcripts included** — several methods need text (B2, density tokens/sec,
   teacher scoring). ASR-quality transcripts, not just intent labels.
3. **Accessible without gating** if possible; verify the parquet path works:
   `curl -sI "https://huggingface.co/datasets/<repo>/resolve/refs%2Fconvert%2Fparquet/<config>/<split>/0000.parquet"`
   → expect HTTP 302 + `x-linked-size`. (The `datasets` streaming client stalls
   unauthenticated — PROGRESS it.7. Gated sets like Common Voice/GigaSpeech are
   acceptable only if you confirm we can get a token.)
4. **Sample rate ≥ 16 kHz** preferred. Narrowband (8 kHz telephone) is fine for
   the spontaneous slot only — note it, since bandwidth changed method behavior
   (PROGRESS it.7).
5. **A held-out test split** (or an obvious ESB / Open ASR Leaderboard eval
   pairing) so "scoring stays comparable" is measurable in-domain AND
   out-of-domain. Flag known train/test leakage (e.g. Common Voice sentence
   overlap).
6. **Size 10–500 h** for the training tier (bigger is fine if it has official
   subsets we can ladder, like GigaSpeech XS→XL).

## Scoring rubric for the writeup (per candidate)

- repo id + config + splits, hours, clip count, sample rate, license
- archetype (table above) and expected redundancy profile — which of our
  methods (B2/F1/F2b/F4/M1-3) you predict will bite, and why
- eval story: in-domain test set + which OOD sets pair with it
- access check result (the curl line above) + total download GB
- risks (leakage, label quality, licensing ambiguity)

Deliver as a table + 1 short paragraph per candidate, PR'd against this repo.
Once agreed, stage-1 scoring is one command per dataset
(`scripts/01_score.py --from-parquet ...`, ~1600× realtime CPU) and produces
the manifests the training tier consumes.
