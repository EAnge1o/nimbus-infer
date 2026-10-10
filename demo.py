"""One-command demonstration of nimbus-infer.

Runs the whole pipeline on the CPU with no downloads:

    python demo.py

1. train a byte-level BPE tokenizer from scratch on a tiny corpus
2. build a GPT-2 style decoder and a LLaMA style decoder (random weights)
3. save the checkpoint, reload it, and generate text
4. show streaming generation
5. measure the KV-cache speedup and the cache footprint
6. quantize the weights to int8 and report the compression ratio

The weights are randomly initialised, so the generated text is gibberish on
purpose -- the point is that every stage is a real implementation, not a stub.
"""
from __future__ import annotations

import os

from nimbus_infer.bench import cache_memory_report, cache_speedup
from nimbus_infer.cli import configure_console_output
from nimbus_infer.config import ModelConfig, llama_style
from nimbus_infer.engine import Engine
from nimbus_infer.quant import compression_ratio, quantize_params, size_bytes
from nimbus_infer.sampling import SamplingParams
from nimbus_infer.tokenizer import BPETokenizer

CORPUS = (
    "nimbus infer is a from scratch inference engine written in pure numpy. "
    "the engine tokenizes text with a byte level byte pair encoding vocabulary. "
    "attention is causal and keys and values are cached between decode steps. "
    "grouped query attention reduces the cache while rotary embeddings encode position. "
) * 40


def rule(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main() -> int:
    configure_console_output()
    rule("1. byte-level BPE tokenizer (trained here, no downloads)")
    tokenizer = BPETokenizer.train(CORPUS, vocab_size=384)
    sample = "the engine caches keys and values"
    ids = tokenizer.encode(sample)
    print(f"corpus bytes      : {len(CORPUS):,}")
    print(f"vocab size        : {tokenizer.vocab_size}")
    print(f"encode({sample!r}) -> {ids}")
    print(f"pieces            : {tokenizer.token_strings(ids)}")
    print(f"round-trip        : {tokenizer.decode(ids)!r}")
    print(f"compression       : {len(sample)} chars -> {len(ids)} tokens")

    rule("2. GPT-2 style decoder: LayerNorm + GELU + learned positions")
    gpt2 = ModelConfig(
        name="demo-gpt2",
        vocab_size=tokenizer.vocab_size,
        n_layers=4,
        n_heads=4,
        d_model=128,
        d_mlp=512,
        max_seq_len=128,
        norm="layernorm",
        activation="gelu",
        pos_encoding="learned",
        tie_embeddings=True,
    )
    engine = Engine.random(gpt2, tokenizer=tokenizer, seed=42)
    print(f"parameters        : {engine.n_params:,}")

    rule("3. checkpoint round-trip (safetensors written by hand)")
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "demo-gpt2")
    engine.save_pretrained(checkpoint_dir)
    weights = os.path.join(checkpoint_dir, "weights.safetensors")
    print(f"checkpoint dir    : {os.path.relpath(checkpoint_dir)}")
    print(f"checkpoint files  : {sorted(os.listdir(checkpoint_dir))}")
    print(f"weights file size : {os.path.getsize(weights):,} bytes")
    reloaded = Engine.from_pretrained(checkpoint_dir)
    print(f"reloaded params   : {reloaded.n_params:,} (match: {reloaded.n_params == engine.n_params})")

    rule("4. greedy generation (random weights -> gibberish, by design)")
    prompt = "the engine"
    text = reloaded.generate(prompt, SamplingParams(max_new_tokens=24, temperature=0.0))
    print(f"prompt     : {prompt!r}")
    print(f"continuation: {text!r}")

    rule("5. streaming generation (top-p sampling)")
    stream_params = SamplingParams(max_new_tokens=20, temperature=0.9, top_p=0.9, seed=7)
    print("prompt     : 'nimbus infer' -> ", end="", flush=True)
    for delta in reloaded.stream("nimbus infer", stream_params):
        print(delta, end="", flush=True)
    print()

    rule("6. KV cache: speedup and footprint")
    bench = cache_speedup(reloaded, prompt, n_tokens=24)
    cached, naive = bench["cached"], bench["naive"]
    print(f"with KV cache     : {cached['tokens_per_sec']:8.1f} tok/s ({cached['seconds']:.3f}s)")
    print(f"full recompute    : {naive['tokens_per_sec']:8.1f} tok/s ({naive['seconds']:.3f}s)")
    print(f"speedup           : {bench['speedup']:.2f}x")
    memory = cache_memory_report(reloaded, gpt2.max_seq_len)
    print(f"cache @ {gpt2.max_seq_len} tokens : {memory['total_mb']:.3f} MB "
          f"({memory['bytes_per_token']} bytes/token)")

    rule("7. LLaMA style decoder: RMSNorm + SwiGLU + RoPE + GQA")
    llama = llama_style(vocab_size=tokenizer.vocab_size)
    llama.n_layers = 4
    llama.max_seq_len = 256
    llama_engine = Engine.random(llama, tokenizer=tokenizer, seed=1)
    print(f"parameters        : {llama_engine.n_params:,}")
    print(f"query heads       : {llama.n_heads}, kv heads: {llama.resolved_n_kv_heads} "
          f"(group size {llama.kv_group_size})")
    out = llama_engine.generate("rotary embeddings", SamplingParams(max_new_tokens=16, temperature=0.0))
    print(f"continuation      : {out!r}")

    rule("8. int8 weight-only quantization")
    packed = quantize_params(reloaded.params, bits=8)
    ratio = compression_ratio(reloaded.params, packed)
    print(f"float32 size      : {size_bytes(reloaded.params):,} bytes")
    print(f"int8 size          : {size_bytes(packed):,} bytes")
    print(f"compression ratio : {ratio:.2f}x")

    rule("done")
    print("All stages ran locally with NumPy as the only dependency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())