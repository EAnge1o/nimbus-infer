"""Download and convert HuggingFace checkpoints using only the standard library.

This never imports ``transformers`` or ``huggingface_hub``: it fetches the raw
files with ``urllib``, maps the HF config onto our :class:`ModelConfig`, reuses
the from-scratch safetensors reader and the HF weight converters, and writes a
native nimbus checkpoint.

    python -m nimbus_infer convert --repo distilgpt2 --out checkpoints/distilgpt2

With real weights the engine produces real English instead of random token soup.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from typing import Callable, Optional

from .config import ModelConfig
from .engine import Engine
from .tokenizer import BPETokenizer
from .weights import convert_hf_gpt2, convert_hf_llama, read_safetensors

HF_BASE = "https://huggingface.co"
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.safetensors")
_TOKENIZER_FILES = ("vocab.json", "merges.txt")
_USER_AGENT = "nimbus-infer/0.1 (+https://github.com/EAnge1o/nimbus-infer)"


# ---------------------------------------------------------------------------
# config mapping
# ---------------------------------------------------------------------------
def is_llama_config(hf_config: dict) -> bool:
    model_type = str(hf_config.get("model_type", "")).lower()
    if model_type in ("llama", "mistral", "qwen2"):
        return True
    return "num_hidden_layers" in hf_config and "n_layer" not in hf_config


def config_from_hf(hf_config: dict, name: str = "hf-model") -> ModelConfig:
    """Map a HuggingFace ``config.json`` onto a nimbus :class:`ModelConfig`."""
    if not is_llama_config(hf_config):
        # GPT-2 family
        d_model = int(hf_config["n_embd"])
        return ModelConfig(
            name=name,
            vocab_size=int(hf_config["vocab_size"]),
            n_layers=int(hf_config["n_layer"]),
            n_heads=int(hf_config["n_head"]),
            d_model=d_model,
            d_mlp=int(hf_config.get("n_inner") or 4 * d_model),
            max_seq_len=int(hf_config.get("n_positions", 1024)),
            norm="layernorm",
            activation="gelu",
            pos_encoding="learned",
            tie_embeddings=bool(hf_config.get("tie_word_embeddings", True)),
            norm_eps=float(hf_config.get("layer_norm_epsilon", 1e-5)),
        )

    d_model = int(hf_config["hidden_size"])
    n_heads = int(hf_config["num_attention_heads"])
    return ModelConfig(
        name=name,
        vocab_size=int(hf_config["vocab_size"]),
        n_layers=int(hf_config["num_hidden_layers"]),
        n_heads=n_heads,
        n_kv_heads=int(hf_config.get("num_key_value_heads", n_heads)),
        head_dim=hf_config.get("head_dim"),
        d_model=d_model,
        d_mlp=int(hf_config.get("intermediate_size", 4 * d_model)),
        max_seq_len=int(hf_config.get("max_position_embeddings", 2048)),
        norm="rmsnorm",
        activation="swiglu",
        pos_encoding="rope",
        tie_embeddings=bool(hf_config.get("tie_word_embeddings", False)),
        norm_eps=float(hf_config.get("rms_norm_eps", 1e-6)),
        rope_theta=float(hf_config.get("rope_theta", 10000.0)),
    )


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
def download_file(
    repo: str,
    filename: str,
    dest_dir: str,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> Optional[str]:
    """Download one file from a HF repo. Returns the local path, or None if 404."""
    url = f"{HF_BASE}/{repo}/resolve/main/{filename}"
    dest = os.path.join(dest_dir, filename)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response:
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            with open(dest, "wb") as fh:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    received += len(chunk)
                    if progress:
                        progress(filename, received, total)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    return dest


def convert_hf_checkpoint(
    source: str,
    out_dir: str,
    name: Optional[str] = None,
    local: bool = False,
    progress: Optional[Callable[[str, int, int], None]] = print,
) -> Engine:
    """Convert an HF repo id (or a local HF directory) into a nimbus checkpoint."""
    cleanup = None
    if local or os.path.isdir(source):
        src_dir = source
        default_name = os.path.basename(os.path.normpath(source)) or "hf-model"
    else:
        src_dir = tempfile.mkdtemp(prefix="nimbus_hf_")
        cleanup = src_dir
        default_name = source.replace("/", "__")

    try:
        cfg_path = os.path.join(src_dir, "config.json")
        if not os.path.exists(cfg_path):
            download_file(source, "config.json", src_dir, progress)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"config.json not found for {source}")

        with open(cfg_path, encoding="utf-8") as fh:
            hf_config = json.load(fh)

        weight_path = None
        for candidate in _WEIGHT_FILES:
            path = os.path.join(src_dir, candidate)
            if not os.path.exists(path) and not local and not os.path.isdir(source):
                download_file(source, candidate, src_dir, progress)
            if os.path.exists(path):
                weight_path = path
                break
        if weight_path is None:
            raise FileNotFoundError(f"no safetensors weights found for {source}")

        for candidate in _TOKENIZER_FILES + ("tokenizer.json",):
            path = os.path.join(src_dir, candidate)
            if not os.path.exists(path) and not local and not os.path.isdir(source):
                download_file(source, candidate, src_dir, progress)

        config = config_from_hf(hf_config, name or default_name)
        tensors = read_safetensors(weight_path)
        if is_llama_config(hf_config):
            params = convert_hf_llama(tensors, config)
        else:
            params = convert_hf_gpt2(tensors, config)

        tokenizer = None
        has_tok = all(os.path.exists(os.path.join(src_dir, f)) for f in _TOKENIZER_FILES)
        if has_tok:
            tokenizer = BPETokenizer.load(src_dir)
        else:
            print(
                "  ! vocab.json + merges.txt not found; the checkpoint will have no "
                "tokenizer (this engine currently only reads byte-level BPE vocab files)."
            )

        engine = Engine(config, params, tokenizer)
        engine.save_pretrained(out_dir)
        return engine
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)