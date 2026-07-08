"""T2 stage-1 scoring: frozen-Whisper pass over the corpus (one-time).

Fills per-clip columns: teacher_logprob (avg token log-prob of the reference
transcript under the frozen model — the ASR analog of LM perplexity),
zero_shot_wer, encoder_embedding (mean-pooled, for M2 SemDeDup).
Validated CPU pilot: whisper-tiny.en, PROGRESS.md it.15. GPU day swaps the
model name for whisper-small and batches.
"""
from __future__ import annotations

import numpy as np


def load_whisper(name: str = "openai/whisper-tiny.en", device: str = "cpu"):
    """Returns (model, processor) in eval mode."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name).to(device).eval()
    return model, processor


def score_clip(model, processor, y: np.ndarray, transcript: str,
               device: str = "cpu") -> dict:
    """Scores one 16 kHz clip against its reference transcript.

    Returns {teacher_logprob, zero_shot_wer, encoder_embedding}.
    """
    import jiwer
    import torch

    features = processor(y, sampling_rate=16_000, return_tensors="pt").input_features.to(device)
    labels = processor.tokenizer(transcript, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(input_features=features, labels=labels)
        emb = model.model.encoder(features).last_hidden_state.mean(dim=1)
        pred_ids = model.generate(features, max_new_tokens=200)
    hyp = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    norm = processor.tokenizer.normalize  # Whisper English text normalizer
    ref_n, hyp_n = norm(transcript), norm(hyp)
    return {
        "teacher_logprob": -float(out.loss),   # mean token log-prob (loss is mean NLL)
        "zero_shot_wer": jiwer.wer(ref_n, hyp_n) if ref_n else np.nan,
        "encoder_embedding": emb[0].cpu().numpy().astype(np.float32),
    }
