"""Weight-only quantization (int8 / int4), implemented from scratch.

Quantizing a checkpoint shrinks it on disk. We use symmetric round-to-nearest
quantization with a scale stored per output channel, or per group of input
channels when ``group_size`` is given.

4-bit weights are genuinely packed two-per-byte (nibble packing), so an int4
checkpoint is half the size of int8 rather than being stored as int8 with a flag.
The engine dequantizes into float32 at load time; :func:`quantized_linear`
exposes a fused path for extensions that keep weights packed during inference.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# nibble packing helpers
# ---------------------------------------------------------------------------
def _pack_int4(q: np.ndarray) -> np.ndarray:
    """Pack int8 values in [-8, 7] two per byte (little nibble first)."""
    flat = (q.astype(np.int16).reshape(-1) + 8).astype(np.uint8)  # bias to 0..15
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
    low = flat[0::2] & 0x0F
    high = (flat[1::2] & 0x0F) << 4
    return (low | high).astype(np.uint8)


def _unpack_int4(packed: np.ndarray, numel: int) -> np.ndarray:
    """Inverse of :func:`_pack_int4`."""
    packed = np.asarray(packed, dtype=np.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    flat = np.empty(packed.size * 2, dtype=np.int16)
    flat[0::2] = low
    flat[1::2] = high
    return (flat[:numel] - 8).astype(np.int8)


# ---------------------------------------------------------------------------
# quantize / dequantize
# ---------------------------------------------------------------------------
def quantize(weight: np.ndarray, bits: int = 8, group_size: int | None = None) -> dict[str, Any]:
    """Quantize a 2-D weight of shape (in_features, out_features)."""
    if bits not in (4, 8):
        raise ValueError("only 4-bit and 8-bit symmetric quantization are supported")
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("weight must be 2-D")

    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    in_f, out_f = w.shape

    if group_size:
        if in_f % group_size != 0:
            raise ValueError("in_features must be divisible by group_size")
        groups = in_f // group_size
        groups_view = w.reshape(groups, group_size, out_f)
        scale = np.abs(groups_view).max(axis=1, keepdims=True) / qmax
        scale = np.where(scale == 0, np.float32(1e-8), scale).astype(np.float32)
        q = np.round(groups_view / scale).clip(qmin, qmax).astype(np.int8)
    else:
        scale = np.abs(w).max(axis=0, keepdims=True) / qmax
        scale = np.where(scale == 0, np.float32(1e-8), scale).astype(np.float32)
        q = np.round(w / scale).clip(qmin, qmax).astype(np.int8)

    packed: dict[str, Any] = {
        "scale": scale,
        "bits": bits,
        "group_size": group_size,
        "shape": (in_f, out_f),
        "q_shape": tuple(q.shape),
    }
    if bits == 4:
        packed["q"] = _pack_int4(q)
        packed["qnumel"] = int(q.size)
    else:
        packed["q"] = q
    return packed


def dequantize(packed: dict[str, Any]) -> np.ndarray:
    """Rebuild a float32 weight from a quantized dict."""
    if packed.get("bits") == 4:
        q = _unpack_int4(packed["q"], packed["qnumel"]).reshape(packed["q_shape"])
    else:
        q = packed["q"]
    q = q.astype(np.float32)
    scale = packed["scale"]
    if packed.get("group_size"):
        groups, group_size, out_f = packed["q_shape"]
        restored = q.reshape(groups, group_size, out_f) * scale  # scale: (g,1,out)
        return restored.reshape(packed["shape"])
    return q * scale


def is_quantized(value: Any) -> bool:
    return isinstance(value, dict) and "q" in value and "scale" in value


def quantized_linear(x: np.ndarray, packed: dict[str, Any], bias: np.ndarray | None = None) -> np.ndarray:
    """Linear layer that dequantizes per-call (for keeping weights packed in memory)."""
    return x @ dequantize(packed) + (bias if bias is not None else 0.0)


def quantize_params(params: dict[str, Any], bits: int = 8, group_size: int | None = None) -> dict[str, Any]:
    """Quantize every 2-D weight in a parameter dict, leaving vectors untouched."""
    out: dict[str, Any] = {}
    for name, value in params.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            out[name] = quantize(value, bits=bits, group_size=group_size)
        else:
            out[name] = value
    return out


def dequantize_params(params: dict[str, Any]) -> dict[str, np.ndarray]:
    return {name: dequantize(v) if is_quantized(v) else v for name, v in params.items()}


def size_bytes(params: dict[str, Any]) -> int:
    """Serialized size of a (possibly quantized) parameter dict, in bytes."""
    total = 0
    for value in params.values():
        if is_quantized(value):
            total += value["q"].nbytes + value["scale"].nbytes
        elif isinstance(value, np.ndarray):
            total += value.nbytes
    return total


def compression_ratio(original: dict[str, Any], quantized: dict[str, Any]) -> float:
    orig = size_bytes(original)
    quant = size_bytes(quantized)
    return orig / quant if quant else 0.0


def effective_bits(params: dict[str, Any]) -> float:
    """Average stored bits per weight across all quantized matrices."""
    total_bits = 0
    total_weights = 0
    for value in params.values():
        if is_quantized(value):
            total_bits += value["q"].nbytes * 8
            total_weights += int(np.prod(value["shape"]))
    return (total_bits / total_weights) if total_weights else 0.0


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((64, 32)) * 0.1).astype(np.float32)
    for bits in (8, 4):
        packed = quantize(w, bits=bits)
        restored = dequantize(packed)
        err = float(np.abs(w - restored).max())
        print(f"int{bits}: max_err={err:.5f} stored_bits={effective_bits({'w': packed}):.2f} "
              f"size={size_bytes({'w': packed})} (float32 {size_bytes({'w': w})})")