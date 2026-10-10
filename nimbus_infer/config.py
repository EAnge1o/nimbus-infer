"""Model configuration for nimbus-infer.

A single :class:`ModelConfig` describes both a GPT-2 style decoder (learned
positional embeddings, LayerNorm, GELU, multi-head attention) and a modern
LLaMA style decoder (RoPE, RMSNorm, SwiGLU, grouped-query attention). The engine
reads the flags instead of hard-coding an architecture, so one code path serves
both families.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional

# Architecture flag vocabularies. Kept as plain strings so a config file stays
# human readable and forward compatible with future variants.
NORMS = ("layernorm", "rmsnorm")
ACTIVATIONS = ("gelu", "swiglu")
POS_ENCODINGS = ("learned", "rope")


@dataclass
class ModelConfig:
    """Static shape/layout description of a decoder-only transformer."""

    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_mlp: int = 3072
    max_seq_len: int = 1024

    # If unset, head_dim = d_model // n_heads and n_kv_heads = n_heads.
    head_dim: Optional[int] = None
    n_kv_heads: Optional[int] = None

    norm: str = "layernorm"
    activation: str = "gelu"
    pos_encoding: str = "learned"

    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    scale_attn_by_head_dim: bool = True
    name: str = "model"

    def __post_init__(self) -> None:
        if self.norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {self.norm!r}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {ACTIVATIONS}, got {self.activation!r}"
            )
        if self.pos_encoding not in POS_ENCODINGS:
            raise ValueError(
                f"pos_encoding must be one of {POS_ENCODINGS}, got {self.pos_encoding!r}"
            )
        if self.d_model % self.n_heads != 0 and self.head_dim is None:
            raise ValueError(
                "d_model must be divisible by n_heads when head_dim is unset"
            )
        if self.n_kv_heads is not None and self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

    # -- resolved layout -------------------------------------------------
    @property
    def resolved_head_dim(self) -> int:
        return self.head_dim or self.d_model // self.n_heads

    @property
    def resolved_n_kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    @property
    def attn_scale(self) -> float:
        if self.scale_attn_by_head_dim:
            return self.resolved_head_dim ** -0.5
        return 1.0

    @property
    def kv_group_size(self) -> int:
        """How many query heads share one KV head (1 for plain MHA)."""
        return self.n_heads // self.resolved_n_kv_heads

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    @classmethod
    def from_json(cls, text: str) -> "ModelConfig":
        return cls.from_dict(json.loads(text))

    # -- parameter accounting --------------------------------------------
    def param_count(self, embedding_params: int = 0) -> int:
        """Analytically count parameters for the described architecture."""
        d = self.d_model
        hd = self.resolved_head_dim
        n_kv = self.resolved_n_kv_heads
        mlp = self.d_mlp

        q = d * (self.n_heads * hd)
        k = d * (n_kv * hd)
        v = d * (n_kv * hd)
        o = (self.n_heads * hd) * d
        attn = q + k + v + o

        if self.activation == "swiglu":
            mlp_params = 3 * d * mlp
        else:
            mlp_params = 2 * d * mlp

        if self.norm == "rmsnorm":
            norm_params = d  # weight only
        else:
            norm_params = 2 * d  # weight + bias

        per_layer = attn + mlp_params + 2 * norm_params
        body = self.n_layers * per_layer

        if not embedding_params:
            # token embedding (+ learned positions if used), plus final norm
            embedding_params = self.vocab_size * d
            if self.pos_encoding == "learned":
                embedding_params += self.max_seq_len * d
            if not self.tie_embeddings:
                embedding_params += self.vocab_size * d
        return body + embedding_params + norm_params


# ---------------------------------------------------------------------------
# Ready-made presets
# ---------------------------------------------------------------------------
def gpt2_124m() -> ModelConfig:
    return ModelConfig(
        name="gpt2-124m",
        vocab_size=50257,
        n_layers=12,
        n_heads=12,
        d_model=768,
        d_mlp=3072,
        max_seq_len=1024,
        norm="layernorm",
        activation="gelu",
        pos_encoding="learned",
        tie_embeddings=True,
        norm_eps=1e-5,
    )


def llama_style(vocab_size: int = 32000) -> ModelConfig:
    """The architecture family used by LLaMA / Mistral style models."""
    return ModelConfig(
        name="llama-style",
        vocab_size=vocab_size,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        d_model=256,
        d_mlp=688,
        max_seq_len=2048,
        norm="rmsnorm",
        activation="swiglu",
        pos_encoding="rope",
        tie_embeddings=False,
        rope_theta=10000.0,
        norm_eps=1e-6,
    )


def tiny(cfg: Optional[ModelConfig] = None) -> ModelConfig:
    """A CPU-friendly model used by the demo and the test suite."""
    if cfg is not None:
        return cfg
    return ModelConfig(
        name="tiny-gpt2",
        vocab_size=512,
        n_layers=4,
        n_heads=4,
        d_model=128,
        d_mlp=512,
        max_seq_len=256,
        norm="layernorm",
        activation="gelu",
        pos_encoding="learned",
        tie_embeddings=True,
    )


PRESETS = {
    "gpt2-124m": gpt2_124m,
    "llama-style": llama_style,
    "tiny": tiny,
}


def get_preset(name: str) -> ModelConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]()