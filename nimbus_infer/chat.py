"""Interactive terminal REPL.

    python -m nimbus_infer chat --model checkpoints/distilgpt2

Keeps one KV cache alive across turns, so every turn after the first only
prefills the new input instead of the whole conversation. Type ``/reset`` to
clear the context, ``/exit`` to leave.
"""
from __future__ import annotations

import sys
import time
from typing import Optional

from .engine import Engine
from .sampling import SamplingParams
from .tokenizer import IncrementalDecoder


def run_chat(
    engine: Engine,
    params: Optional[SamplingParams] = None,
    template: Optional[str] = None,
    show_stats: bool = True,
) -> int:
    if engine.tokenizer is None:
        print("this checkpoint has no tokenizer; run `convert` on a byte-level BPE model")
        return 1

    params = params or SamplingParams(
        max_new_tokens=64, temperature=0.8, top_p=0.95, repetition_penalty=1.1
    )
    info = engine.info()
    arch = info["architecture"]

    print("=" * 68)
    print(f"nimbus-infer chat :: {info['name']}")
    print(f"  {info['parameters']:,} params | {arch['layers']} layers | "
          f"{arch['norm']} + {arch['activation']} + {arch['pos_encoding']}")
    print(f"  sampling: temp={params.temperature} top_k={params.top_k} "
          f"top_p={params.top_p} rep={params.repetition_penalty}")
    print("  commands: /reset (clear context)  /exit")
    print("=" * 68)

    cache = engine.model.new_cache()
    while True:
        try:
            line = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        command = line.strip()
        if command in ("/exit", "/quit"):
            break
        if command == "/reset":
            cache.reset()
            print("[context cleared]")
            continue
        if not command:
            continue

        prompt = template.format(user=command) if template else command
        new_ids = engine.encode(prompt)
        if cache.pos + len(new_ids) + params.max_new_tokens > engine.config.max_seq_len:
            cache.reset()
            print("[context window full -> reset]")

        start = time.perf_counter()
        produced = 0
        decoder = IncrementalDecoder(engine.tokenizer)

        print("ai> ", end="", flush=True)
        try:
            for token in engine.generate_into(cache, new_ids, params):
                produced += 1
                sys.stdout.write(decoder.push(token))
                sys.stdout.flush()
            sys.stdout.write(decoder.flush())
        except KeyboardInterrupt:
            print(" [interrupted]", end="")
        elapsed = time.perf_counter() - start
        print()

        if show_stats:
            tps = produced / elapsed if elapsed > 0 else 0.0
            print(
                f"   [{produced} tokens | {tps:.1f} tok/s | "
                f"kv cache {cache.used_bytes / (1024 ** 2):.2f} MB | "
                f"context {cache.seq_len}]"
            )
    print("bye.")
    return 0