"""Sampling strategies: greedy, temperature, top-k, top-p and repetition penalty.

All samplers operate on a single 1-D logits vector and are deliberately pure
functions so they can be unit-tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .backend import softmax

NEG_INF = -np.inf  # filtered-out logits


@dataclass
class SamplingParams:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0          # 0 disables top-k filtering
    top_p: float = 1.0      # 1.0 disables nucleus filtering
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    stop_tokens: tuple[int, ...] = field(default_factory=tuple)

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def apply_repetition_penalty(
    logits: np.ndarray, generated: Sequence[int], penalty: float
) -> np.ndarray:
    if penalty == 1.0 or not generated:
        return logits
    logits = logits.astype(np.float64, copy=True)
    for token in set(int(t) for t in generated):
        if token < 0 or token >= logits.shape[0]:
            continue
        if logits[token] > 0:
            logits[token] /= penalty
        else:
            logits[token] *= penalty
    return logits


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= logits.shape[0]:
        return logits
    kth = np.partition(logits, -k)[-k]
    return np.where(logits < kth, NEG_INF, logits)


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    if p >= 1.0:
        return logits
    if p <= 0.0:
        out = np.full_like(logits, NEG_INF)
        out[int(np.argmax(logits))] = logits[int(np.argmax(logits))]
        return out
    order = np.argsort(-logits)
    sorted_logits = logits[order]
    probs = softmax(sorted_logits)
    cumulative = np.cumsum(probs)
    # Drop tokens once the cumulative mass *excluding* them already exceeds p.
    remove = (cumulative - probs) > p
    sorted_logits = np.where(remove, NEG_INF, sorted_logits)
    out = np.empty_like(logits)
    out[order] = sorted_logits
    return out


def sample(
    logits: np.ndarray,
    params: Optional[SamplingParams] = None,
    generated: Sequence[int] = (),
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Turn a logits vector into a single token id."""
    params = params or SamplingParams()
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)

    logits = apply_repetition_penalty(logits, generated, params.repetition_penalty)

    if params.greedy:
        return int(np.argmax(logits))

    logits = logits / max(params.temperature, 1e-6)
    logits = top_k_filter(logits, params.top_k)
    logits = top_p_filter(logits, params.top_p)

    probs = softmax(logits.astype(np.float64))
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(logits))
    probs = probs / total
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.choice(probs.shape[0], p=probs))