#!/usr/bin/env python
"""Stage 3: manifest → fine-tuned checkpoint. SKELETON (needs GPU session).

Usage:
  python scripts/03_train.py --manifest results/manifests/b1_random_h25_s0.json \
      --model facebook/wav2vec2-base --steps 20000 --out results/ckpts/

Fixed recipe across ALL runs (PLAN.md §2): same total STEPS regardless of subset
size (report the same-epochs interpretation once, in an appendix run).

TODO(first GPU session):
  1. load manifest, filter HF dataset to clip_ids (build id->index once, cache).
  2. wav2vec2 path: Wav2Vec2ForCTC + processor, HF Trainer, fp16, group_by_length.
     whisper path: WhisperForConditionalGeneration + Seq2SeqTrainer.
  3. hyperparams pinned in configs/<model>.yaml — identical for every method/budget.
  4. log to wandb: method, budget_hours, seed, actual_hours as run tags.
"""
raise NotImplementedError("training skeleton — fill in on first GPU session (see TODO)")
