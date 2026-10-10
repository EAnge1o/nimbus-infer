"""Key/Value cache for autoregressive decoding.

Without a cache, generating N tokens recomputes attention over the whole prefix
every step (O(N^2) work). Caching the projected keys and values turns each decode
step into a single-token forward pass (O(N) work and memory per step).
"""
from __future__ import annotations

import numpy as np

from .backend import DTYPE


class KVCache:
    """Pre-allocated per-layer key/value stores.

    Arrays are shaped ``(n_kv_heads, max_seq_len, head_dim)`` and filled in place
    at the current write cursor :attr:`pos`, so no per-step reallocation happens.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
    ) -> None:
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.pos = 0
        shape = (n_kv_heads, max_seq_len, head_dim)
        self.k = [np.zeros(shape, dtype=DTYPE) for _ in range(n_layers)]
        self.v = [np.zeros(shape, dtype=DTYPE) for _ in range(n_layers)]

    # -- writing ---------------------------------------------------------
    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Store ``k``/``v`` (shape ``(n_kv_heads, seq, head_dim)``) at the cursor."""
        seq = k.shape[1]
        end = self.pos + seq
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: trying to write up to {end} > {self.max_seq_len}"
            )
        self.k[layer][:, self.pos:end] = k
        self.v[layer][:, self.pos:end] = v

    def advance(self, seq: int) -> None:
        self.pos += seq

    def reset(self) -> None:
        self.pos = 0

    # -- reading ---------------------------------------------------------
    def keys(self, layer: int, upto: int | None = None) -> np.ndarray:
        return self.k[layer][:, : (self.pos + 1 if upto is None else upto)]

    def values(self, layer: int, upto: int | None = None) -> np.ndarray:
        return self.v[layer][:, : (self.pos + 1 if upto is None else upto)]

    # -- introspection ---------------------------------------------------
    @property
    def seq_len(self) -> int:
        return self.pos

    @property
    def bytes(self) -> int:
        """Resident size of the cache in bytes."""
        per_layer = self.n_kv_heads * self.max_seq_len * self.head_dim
        return 2 * self.n_layers * per_layer * np.dtype(DTYPE).itemsize

    @property
    def used_bytes(self) -> int:
        """Bytes actually occupied by cached positions."""
        per_layer = self.n_kv_heads * self.pos * self.head_dim
        return 2 * self.n_layers * per_layer * np.dtype(DTYPE).itemsize

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"KVCache(layers={self.n_layers}, kv_heads={self.n_kv_heads}, "
            f"head_dim={self.head_dim}, pos={self.pos}/{self.max_seq_len})"
        )