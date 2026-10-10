"""Decoder-only transformer, implemented from scratch on top of the backend.

One implementation covers both architecture families via :class:`ModelConfig`:

* GPT-2 -- learned positional embeddings, LayerNorm, tanh-GELU MLP, MHA.
* LLaMA -- RoPE, RMSNorm, SwiGLU MLP, grouped-query attention (GQA).

Weights are plain ``dict[str, np.ndarray]`` entries stored as
``(in_features, out_features)`` so a linear layer is just ``x @ W`` (+ bias).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import backend as B
from .backend import DTYPE
from .cache import KVCache
from .config import ModelConfig


# ---------------------------------------------------------------------------
# Parameter initialisation
# ---------------------------------------------------------------------------
def init_params(config: ModelConfig, seed: int = 0) -> dict[str, np.ndarray]:
    """Randomly initialise a parameter dict (used by the demo and tests)."""
    rng = np.random.default_rng(seed)
    d = config.d_model
    hd = config.resolved_head_dim
    n_kv = config.resolved_n_kv_heads
    mlp = config.d_mlp

    def normal(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.02).astype(DTYPE)

    params: dict[str, np.ndarray] = {"wte": normal(config.vocab_size, d)}
    if config.pos_encoding == "learned":
        params["wpe"] = normal(config.max_seq_len, d)

    for i in range(config.n_layers):
        p = f"h.{i}."
        params[p + "norm1.weight"] = np.ones(d, dtype=DTYPE)
        params[p + "norm2.weight"] = np.ones(d, dtype=DTYPE)
        if config.norm == "layernorm":
            params[p + "norm1.bias"] = np.zeros(d, dtype=DTYPE)
            params[p + "norm2.bias"] = np.zeros(d, dtype=DTYPE)

        params[p + "attn.q.weight"] = normal(d, config.n_heads * hd)
        params[p + "attn.k.weight"] = normal(d, n_kv * hd)
        params[p + "attn.v.weight"] = normal(d, n_kv * hd)
        params[p + "attn.o.weight"] = normal(config.n_heads * hd, d)

        if config.activation == "swiglu":
            params[p + "mlp.gate.weight"] = normal(d, mlp)
            params[p + "mlp.up.weight"] = normal(d, mlp)
            params[p + "mlp.down.weight"] = normal(mlp, d)
        else:
            params[p + "mlp.fc.weight"] = normal(d, mlp)
            params[p + "mlp.proj.weight"] = normal(mlp, d)

    params["norm_f.weight"] = np.ones(d, dtype=DTYPE)
    if config.norm == "layernorm":
        params["norm_f.bias"] = np.zeros(d, dtype=DTYPE)
    if not config.tie_embeddings:
        params["lm_head.weight"] = normal(d, config.vocab_size)
    return params


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------
def build_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Precompute cos/sin tables of shape (max_seq_len, head_dim // 2)."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    t = np.arange(max_seq_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq)
    return np.cos(freqs).astype(DTYPE), np.sin(freqs).astype(DTYPE)


def _apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotate ``x`` of shape (heads, seq, head_dim) by precomputed tables."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
class Transformer:
    def __init__(self, config: ModelConfig, params: dict[str, np.ndarray]) -> None:
        self.config = config
        self.params = params
        self._rope = None
        if config.pos_encoding == "rope":
            self._rope = build_rope(
                config.resolved_head_dim, config.max_seq_len, config.rope_theta
            )
        # Populated only when ``forward(..., capture_attention=True)`` is used.
        self.captured_attention: list[np.ndarray] = []
        self._capture = False

    # -- convenience -----------------------------------------------------
    def n_params(self) -> int:
        return sum(int(v.size) for v in self.params.values())

    def new_cache(self) -> KVCache:
        return KVCache(
            n_layers=self.config.n_layers,
            n_kv_heads=self.config.resolved_n_kv_heads,
            head_dim=self.config.resolved_head_dim,
            max_seq_len=self.config.max_seq_len,
        )

    def _lm_head_weight(self) -> np.ndarray:
        if "lm_head.weight" in self.params:
            return self.params["lm_head.weight"]
        return self.params["wte"].T  # weight tying

    # -- forward ---------------------------------------------------------
    def forward(
        self,
        tokens: np.ndarray | list[int],
        cache: Optional[KVCache] = None,
        return_hidden: bool = False,
        capture_attention: bool = False,
    ):
        """Run the model.

        ``tokens`` is a 1-D sequence of ids. With a fresh cache this performs the
        *prefill* pass; subsequent single-token calls reuse the cache (*decode*).
        Returns logits of shape (seq, vocab).

        When ``capture_attention`` is set, ``self.captured_attention`` holds one
        ``(seq, total)`` matrix per layer (averaged over heads) for visualisation.
        """
        cfg = self.config
        p = self.params
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        seq = tokens.shape[0]

        self._capture = capture_attention
        if capture_attention:
            self.captured_attention = []

        pos_offset = cache.pos if cache is not None else 0
        if pos_offset + seq > cfg.max_seq_len:
            raise ValueError(
                f"sequence length {pos_offset + seq} exceeds max_seq_len {cfg.max_seq_len}"
            )
        total = pos_offset + seq

        x = p["wte"][tokens]
        if cfg.pos_encoding == "learned":
            positions = np.arange(pos_offset, total)
            x = x + p["wpe"][positions]

        for i in range(cfg.n_layers):
            x = self._block(i, x, cache, total, pos_offset)

        if cfg.norm == "rmsnorm":
            x = B.rms_norm(x, p["norm_f.weight"], cfg.norm_eps)
        else:
            x = B.layer_norm(x, p["norm_f.weight"], p["norm_f.bias"], cfg.norm_eps)

        self._capture = False
        if return_hidden:
            return x
        return x @ self._lm_head_weight()

    def _block(
        self,
        layer: int,
        x: np.ndarray,
        cache: Optional[KVCache],
        total: int,
        pos_offset: int,
    ) -> np.ndarray:
        cfg = self.config
        p = self.params
        pre = f"h.{layer}."

        # ---- attention ----
        h = self._norm(pre + "norm1", x)
        h = self._attention(pre, h, cache, total, pos_offset)
        x = x + h

        # ---- mlp ----
        h = self._norm(pre + "norm2", x)
        h = self._mlp(pre, h)
        return x + h

    def _norm(self, prefix: str, x: np.ndarray) -> np.ndarray:
        cfg = self.config
        p = self.params
        if cfg.norm == "rmsnorm":
            return B.rms_norm(x, p[prefix + ".weight"], cfg.norm_eps)
        return B.layer_norm(x, p[prefix + ".weight"], p[prefix + ".bias"], cfg.norm_eps)

    def _attention(
        self,
        prefix: str,
        x: np.ndarray,
        cache: Optional[KVCache],
        total: int,
        pos_offset: int,
    ) -> np.ndarray:
        cfg = self.config
        p = self.params
        seq = x.shape[0]
        n_heads = cfg.n_heads
        n_kv = cfg.resolved_n_kv_heads
        hd = cfg.resolved_head_dim

        q = B.linear(x, p[prefix + "attn.q.weight"], p.get(prefix + "attn.q.bias")).reshape(seq, n_heads, hd)
        k = B.linear(x, p[prefix + "attn.k.weight"], p.get(prefix + "attn.k.bias")).reshape(seq, n_kv, hd)
        v = B.linear(x, p[prefix + "attn.v.weight"], p.get(prefix + "attn.v.bias")).reshape(seq, n_kv, hd)
        # (heads, seq, head_dim)
        q = q.transpose(1, 0, 2)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)

        if self._rope is not None:
            cos_full, sin_full = self._rope
            cos = cos_full[pos_offset:total][None, :, :]
            sin = sin_full[pos_offset:total][None, :, :]
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        if cache is not None:
            layer = int(prefix.split(".")[1])
            cache.append(layer, k, v)
            k = cache.keys(layer, total)
            v = cache.values(layer, total)

        if cfg.kv_group_size > 1:  # GQA: broadcast KV heads to query heads
            k = np.repeat(k, cfg.kv_group_size, axis=0)
            v = np.repeat(v, cfg.kv_group_size, axis=0)

        scores = (q @ k.transpose(0, 2, 1)) * cfg.attn_scale  # (heads, seq, total)
        mask = B.causal_mask(seq, total, offset=pos_offset)
        scores = np.where(mask[None, :, :], scores, np.float32(-1e9))
        probs = B.softmax(scores, axis=-1)
        if self._capture:
            self.captured_attention.append(probs.mean(axis=0).copy())
        out = probs @ v  # (heads, seq, head_dim)
        out = out.transpose(1, 0, 2).reshape(seq, n_heads * hd)
        return B.linear(out, p[prefix + "attn.o.weight"], p.get(prefix + "attn.o.bias"))

    def _mlp(self, prefix: str, x: np.ndarray) -> np.ndarray:
        cfg = self.config
        p = self.params
        if cfg.activation == "swiglu":
            gate = B.silu(B.linear(x, p[prefix + "mlp.gate.weight"], p.get(prefix + "mlp.gate.bias")))
            up = B.linear(x, p[prefix + "mlp.up.weight"], p.get(prefix + "mlp.up.bias"))
            return B.linear(gate * up, p[prefix + "mlp.down.weight"], p.get(prefix + "mlp.down.bias"))
        hidden = B.gelu(B.linear(x, p[prefix + "mlp.fc.weight"], p.get(prefix + "mlp.fc.bias")))
        return B.linear(hidden, p[prefix + "mlp.proj.weight"], p.get(prefix + "mlp.proj.bias"))


def param_count_from_dict(params: dict[str, np.ndarray]) -> int:
    return sum(int(v.size) for v in params.values())