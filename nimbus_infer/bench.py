"""Micro-benchmarks: decoding throughput and the KV-cache speedup."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .engine import Engine
from .sampling import SamplingParams


def decode_benchmark(
    engine: Engine,
    prompt: str | list[int],
    n_tokens: int = 32,
    use_cache: bool = True,
) -> dict:
    """Greedy-decode ``n_tokens`` and report wall-clock throughput."""
    params = SamplingParams(max_new_tokens=n_tokens, temperature=0.0)
    start = time.perf_counter()
    produced = engine.generate_ids(prompt, params, use_cache=use_cache)
    elapsed = time.perf_counter() - start
    return {
        "use_cache": use_cache,
        "tokens": len(produced),
        "seconds": elapsed,
        "tokens_per_sec": (len(produced) / elapsed) if elapsed > 0 else float("inf"),
    }


def cache_speedup(engine: Engine, prompt: str | list[int], n_tokens: int = 32) -> dict:
    """Compare decode-with-cache against naive full-recompute decoding."""
    cached = decode_benchmark(engine, prompt, n_tokens, use_cache=True)
    naive = decode_benchmark(engine, prompt, n_tokens, use_cache=False)
    speedup = (naive["seconds"] / cached["seconds"]) if cached["seconds"] > 0 else float("inf")
    return {"cached": cached, "naive": naive, "speedup": speedup}


def cache_memory_report(engine: Engine, sequence_length: int) -> dict:
    config = engine.config
    per_token = (
        2
        * config.n_layers
        * config.resolved_n_kv_heads
        * config.resolved_head_dim
        * np.dtype(np.float32).itemsize
    )
    return {
        "bytes_per_token": per_token,
        "sequence_length": sequence_length,
        "total_bytes": per_token * sequence_length,
        "total_mb": per_token * sequence_length / (1024 ** 2),
    }