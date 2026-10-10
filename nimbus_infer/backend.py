"""Numerical backend.

nimbus-infer keeps exactly one third-party dependency -- NumPy -- and uses it
only as the numerical substrate. Every model-specific behaviour (attention, KV
cache, sampling, tokenization, quantization and weight parsing) lives in this
project. This module centralises the small set of array primitives the rest of
the engine needs, which keeps the model code free of ad-hoc money math and makes
the numerics trivial to unit test.
"""
from __future__ import annotations

import numpy as np

# The engine works in float32 end to end. bfloat16 is only materialised when
# loading weights, then immediately widened.
DTYPE = np.float32


def as_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=DTYPE)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """GPT-2 uses the tanh approximation of GELU."""
    c = 0.7978845608028654  # sqrt(2 / pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def layer_norm(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    ms = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * weight


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """y = x @ W (+ b). Weights are stored as (in_features, out_features)."""
    y = x @ weight
    if bias is not None:
        y = y + bias
    return y


def causal_mask(n_query: int, n_key: int, offset: int = 0) -> np.ndarray:
    """Boolean keep-mask of shape (n_query, n_key).

    ``offset`` is the absolute position of the first query row; a key at index j
    is visible to a query at absolute position ``offset + i`` iff ``j <= offset + i``.
    """
    q_pos = np.arange(offset, offset + n_query)[:, None]
    k_pos = np.arange(n_key)[None, :]
    return k_pos <= q_pos