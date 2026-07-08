#!/usr/bin/env python
"""Stage 4: checkpoint → WER rows appended to results/wer.csv. SKELETON.

Usage:
  python scripts/04_eval.py --ckpt results/ckpts/b1_random_h25_s0 \
      --eval-sets ls_test_clean ls_test_other ami

One row per (method, budget_hours, seed, eval_set): WER via jiwer with the
Whisper EnglishTextNormalizer applied to BOTH reference and hypothesis
(Open ASR Leaderboard convention — comparability with published numbers).

TODO(first GPU session): batched greedy decode per eval set; append rows to
results/wer.csv (method, budget_hours, seed, eval_set, wer, n_utts, ckpt_path).
Eval sets: in-domain (ls test-clean/other or cv test) + OOD via ESB
(edinburghcstr/ami, revdotcom/earnings22, LIUM/tedlium).
"""
raise NotImplementedError("eval skeleton — fill in on first GPU session (see TODO)")
