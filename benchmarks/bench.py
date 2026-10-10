"""Decode-throughput benchmark.

Builds a randomly-initialised model for a preset and reports:

* prefill + decode throughput,
* the KV-cache speedup against naive full-recompute decoding,
* the KV-cache memory footprint,
* int8 / int4 quantization compression.

    python benchmarks/bench.py --preset tiny --tokens 32
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nimbus_infer.bench import cache_memory_report, cache_speedup  # noqa: E402
from nimbus_infer.config import get_preset  # noqa: E402
from nimbus_infer.engine import Engine  # noqa: E402
from nimbus_infer.quant import compression_ratio, quantize_params, size_bytes  # noqa: E402
from nimbus_infer.sampling import SamplingParams  # noqa: E402


def _row(label: str, value: str) -> str:
    return f"  {label:<26} {value}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="tiny", help="tiny | llama-style | gpt2-124m")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=8)
    args = parser.parse_args()

    config = get_preset(args.preset)
    if args.preset == "gpt2-124m":
        # 124M params is only practical to benchmark on the smallest window.
        config.n_layers = 2
        config.max_seq_len = 128
        config.name = "gpt2-124m-truncated"

    engine = Engine.random(config, seed=0)
    prompt = list(range(1, args.prompt_len + 1))

    print(f"nimbus-infer benchmark :: {config.name}")
    print(_row("parameters", f"{engine.n_params:,}"))
    print(_row("layers / heads", f"{config.n_layers} / {config.n_heads} "
                                f"(kv {config.resolved_n_kv_heads})"))
    print(_row("architecture", f"{config.norm} + {config.activation} + {config.pos_encoding}"))
    print(_row("generated tokens", str(args.tokens)))

    # Warm up the BLAS kernels so the first measurement is not penalised.
    engine.generate_ids(prompt, SamplingParams(max_new_tokens=2, temperature=0.0))

    result = cache_speedup(engine, prompt, n_tokens=args.tokens)
    cached, naive = result["cached"], result["naive"]
    print()
    print(_row("with KV cache", f"{cached['tokens_per_sec']:.1f} tok/s ({cached['seconds']:.3f}s)"))
    print(_row("full recompute", f"{naive['tokens_per_sec']:.1f} tok/s ({naive['seconds']:.3f}s)"))
    print(_row("cache speedup", f"{result['speedup']:.2f}x"))

    memory = cache_memory_report(engine, config.max_seq_len)
    print(_row("KV bytes / token", str(memory["bytes_per_token"])))
    print(_row("KV cache @ max_len", f"{memory['total_mb']:.2f} MB"))

    print()
    for bits in (8, 4):
        packed = quantize_params(engine.params, bits=bits)
        ratio = compression_ratio(engine.params, packed)
        print(_row(f"int{bits} compression", f"{ratio:.2f}x "
                                            f"({size_bytes(packed):,} bytes)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())