"""Weight loading, implemented from scratch.

Supported containers:

* ``.safetensors`` -- parsed by hand (uint64 header length + JSON header + raw
  buffer), including bfloat16 widening, because the reference ``safetensors``
  package is a dependency we do not want.
* ``.npz`` -- NumPy's own archive format.
* HuggingFace transformer checkpoints -- the ``*_proj`` / ``c_attn`` naming is
  remapped onto nimbus-infer's parameter names, including the (out, in) -> (in, out)
  transpose that HuggingFace uses for linear weights.
"""
from __future__ import annotations

import json
import os
import struct
from typing import Any

import numpy as np

from .backend import DTYPE
from .config import ModelConfig

_SAFETENSORS_TO_NP = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U64": np.uint64,
    "U32": np.uint32,
    "U16": np.uint16,
    "U8": np.uint8,
    "BOOL": np.bool_,
}
_NP_TO_SAFETENSORS = {
    np.dtype(np.float64): "F64",
    np.dtype(np.float32): "F32",
    np.dtype(np.float16): "F16",
    np.dtype(np.int64): "I64",
    np.dtype(np.int32): "I32",
    np.dtype(np.int16): "I16",
    np.dtype(np.int8): "I8",
    np.dtype(np.uint64): "U64",
    np.dtype(np.uint32): "U32",
    np.dtype(np.uint16): "U16",
    np.dtype(np.uint8): "U8",
    np.dtype(np.bool_): "BOOL",
}


def bfloat16_to_float32(raw: np.ndarray) -> np.ndarray:
    """bfloat16 is the upper 16 bits of a float32 -- widen by shifting."""
    u16 = np.asarray(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def read_safetensors(path: str) -> dict[str, np.ndarray]:
    """Parse a safetensors file without any third-party dependency."""
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len).decode("utf-8"))
        buffer = fh.read()

    tensors: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = info["dtype"]
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]
        chunk = buffer[start:end]
        if dtype == "BF16":
            array = bfloat16_to_float32(np.frombuffer(chunk, dtype=np.uint16)).reshape(shape)
        else:
            np_dtype = _SAFETENSORS_TO_NP[dtype]
            array = np.frombuffer(chunk, dtype=np_dtype).reshape(shape)
        # Copy out so the big file buffer can be released.
        tensors[name] = np.array(array, dtype=DTYPE if array.dtype.kind == "f" else array.dtype)
    return tensors


def save_safetensors(path: str, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
    """Write a safetensors file (used by the tests to prove the reader is correct)."""
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    blobs = []
    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        data = array.tobytes()
        dtype = np.dtype(array.dtype)
        if dtype == np.dtype(np.float32):
            kind = "F32"
        elif dtype in _NP_TO_SAFETENSORS:
            kind = _NP_TO_SAFETENSORS[dtype]
        else:
            raise ValueError(f"unsupported dtype for safetensors: {dtype}")
        header[name] = {
            "dtype": kind,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        blobs.append(data)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * padding
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for blob in blobs:
            fh.write(blob)


def save_npz(path: str, params: dict[str, np.ndarray]) -> None:
    np.savez(path, **params)


def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: np.asarray(data[k], dtype=DTYPE if data[k].dtype.kind == "f" else data[k].dtype) for k in data.files}


def load_weights(path: str) -> dict[str, np.ndarray]:
    """Load a raw tensor dict, dispatching on the file extension."""
    if path.endswith(".safetensors"):
        return read_safetensors(path)
    if path.endswith(".npz"):
        return load_npz(path)
    raise ValueError(f"unsupported weight file: {path}")


# ---------------------------------------------------------------------------
# HuggingFace checkpoint -> nimbus-infer parameter mapping
# ---------------------------------------------------------------------------
def _transpose_linear(w: np.ndarray) -> np.ndarray:
    """HuggingFace stores linear weights as (out, in); we want (in, out)."""
    return np.ascontiguousarray(np.asarray(w, dtype=DTYPE).T)


def convert_hf_llama(tensors: dict[str, np.ndarray], config: ModelConfig) -> dict[str, np.ndarray]:
    """Map a HuggingFace LLaMA/Mistral state dict onto nimbus-infer names."""
    out: dict[str, np.ndarray] = {}
    out["wte"] = np.asarray(tensors["model.embed_tokens.weight"], dtype=DTYPE)
    if "model.norm.weight" in tensors:
        out["norm_f.weight"] = np.asarray(tensors["model.norm.weight"], dtype=DTYPE)
    for i in range(config.n_layers):
        src = f"model.layers.{i}."
        dst = f"h.{i}."
        out[dst + "norm1.weight"] = np.asarray(tensors[src + "input_layernorm.weight"], dtype=DTYPE)
        out[dst + "norm2.weight"] = np.asarray(
            tensors[src + "post_attention_layernorm.weight"], dtype=DTYPE
        )
        out[dst + "attn.q.weight"] = _transpose_linear(tensors[src + "self_attn.q_proj.weight"])
        out[dst + "attn.k.weight"] = _transpose_linear(tensors[src + "self_attn.k_proj.weight"])
        out[dst + "attn.v.weight"] = _transpose_linear(tensors[src + "self_attn.v_proj.weight"])
        out[dst + "attn.o.weight"] = _transpose_linear(tensors[src + "self_attn.o_proj.weight"])
        out[dst + "mlp.gate.weight"] = _transpose_linear(tensors[src + "mlp.gate_proj.weight"])
        out[dst + "mlp.up.weight"] = _transpose_linear(tensors[src + "mlp.up_proj.weight"])
        out[dst + "mlp.down.weight"] = _transpose_linear(tensors[src + "mlp.down_proj.weight"])
    if "lm_head.weight" in tensors:
        out["lm_head.weight"] = _transpose_linear(tensors["lm_head.weight"])
    return out


def convert_hf_gpt2(tensors: dict[str, np.ndarray], config: ModelConfig) -> dict[str, np.ndarray]:
    """Map a HuggingFace GPT-2 state dict (which already stores (in, out))."""
    def get(name: str) -> np.ndarray:
        return np.asarray(tensors[name], dtype=DTYPE)

    out: dict[str, np.ndarray] = {}
    out["wte"] = get("transformer.wte.weight")
    out["wpe"] = get("transformer.wpe.weight")
    for i in range(config.n_layers):
        src = f"transformer.h.{i}."
        dst = f"h.{i}."
        out[dst + "norm1.weight"] = get(src + "ln_1.weight")
        out[dst + "norm1.bias"] = get(src + "ln_1.bias")
        out[dst + "norm2.weight"] = get(src + "ln_2.weight")
        out[dst + "norm2.bias"] = get(src + "ln_2.bias")

        qkv = get(src + "attn.c_attn.weight")  # (d_model, 3 * d_model)
        qkv_b = get(src + "attn.c_attn.bias")
        d = qkv.shape[1] // 3
        for j, key in enumerate(("q", "k", "v")):
            out[dst + f"attn.{key}.weight"] = qkv[:, j * d : (j + 1) * d]
        # Distribute the combined bias onto the individual projections.
        out[dst + "attn.q.bias"] = qkv_b[:d]
        out[dst + "attn.k.bias"] = qkv_b[d : 2 * d]
        out[dst + "attn.v.bias"] = qkv_b[2 * d :]
        out[dst + "attn.o.weight"] = get(src + "attn.c_proj.weight")
        out[dst + "attn.o.bias"] = get(src + "attn.c_proj.bias")

        out[dst + "mlp.fc.weight"] = get(src + "mlp.c_fc.weight")
        out[dst + "mlp.fc.bias"] = get(src + "mlp.c_fc.bias")
        out[dst + "mlp.proj.weight"] = get(src + "mlp.c_proj.weight")
        out[dst + "mlp.proj.bias"] = get(src + "mlp.c_proj.bias")

    out["norm_f.weight"] = get("transformer.ln_f.weight")
    out["norm_f.bias"] = get("transformer.ln_f.bias")
    if not config.tie_embeddings and "lm_head.weight" in tensors:
        out["lm_head.weight"] = get("lm_head.weight")
    return out