"""Command line interface: ``python -m nimbus_infer <command>``."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from typing import Optional

from .bench import cache_memory_report, cache_speedup
from .config import get_preset
from .engine import Engine
from .sampling import SamplingParams
from .tokenizer import BPETokenizer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nimbus-infer",
        description="A from-scratch, NumPy-only LLM inference engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate text from a prompt")
    gen.add_argument("--model", required=True, help="checkpoint directory")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-new-tokens", type=int, default=64)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--repetition-penalty", type=float, default=1.0)
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--stream", action="store_true")

    tok = sub.add_parser("tokenize", help="inspect BPE tokenization")
    tok.add_argument("--tokenizer", required=True, help="directory with vocab.json + merges.txt")
    tok.add_argument("--text", required=True)

    ins = sub.add_parser("inspect", help="print model metadata")
    ins.add_argument("--model", required=True)

    ben = sub.add_parser("bench", help="measure decode throughput and cache speedup")
    ben.add_argument("--model", required=True)
    ben.add_argument("--prompt", default="The history of computing")
    ben.add_argument("--tokens", type=int, default=32)

    conv = sub.add_parser("convert", help="download + convert a HuggingFace checkpoint")
    conv.add_argument("--repo", required=True, help="HF repo id, e.g. distilgpt2, or a local dir")
    conv.add_argument("--out", required=True, help="output nimbus checkpoint directory")
    conv.add_argument("--name", default=None, help="override the model name")
    conv.add_argument("--local", action="store_true", help="treat --repo as a local directory")

    cht = sub.add_parser("chat", help="interactive terminal REPL with KV-cache reuse")
    cht.add_argument("--model", required=True)
    cht.add_argument("--temperature", type=float, default=0.8)
    cht.add_argument("--top-k", type=int, default=40)
    cht.add_argument("--top-p", type=float, default=0.95)
    cht.add_argument("--repetition-penalty", type=float, default=1.1)
    cht.add_argument("--max-new-tokens", type=int, default=64)
    cht.add_argument("--template", default=None, help="optional prompt template, e.g. 'Q: {user}\\nA:'")

    srv = sub.add_parser("serve", help="browser visualisation demo")
    srv.add_argument("--model", default=None, help="checkpoint directory")
    srv.add_argument("--preset", default=None, help="or a random preset: tiny | llama-style")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--open", action="store_true", help="open a browser window")
    return parser


def _params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )


def _cmd_generate(args: argparse.Namespace) -> int:
    engine = Engine.from_pretrained(args.model)
    params = _params(args)
    if args.stream:
        for delta in engine.stream(args.prompt, params):
            sys.stdout.write(delta)
            sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        print(engine.generate(args.prompt, params))
    return 0


def _cmd_tokenize(args: argparse.Namespace) -> int:
    tokenizer = BPETokenizer.load(args.tokenizer)
    ids = tokenizer.encode(args.text)
    pieces = tokenizer.token_strings(ids)
    print(f"vocab_size = {tokenizer.vocab_size}")
    print(f"n_tokens   = {len(ids)}")
    print(f"ids        = {ids}")
    print("pieces     = " + " | ".join(repr(p) for p in pieces))
    print("round-trip = " + repr(tokenizer.decode(ids)))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    engine = Engine.from_pretrained(args.model)
    print(json.dumps(engine.info(), indent=2, ensure_ascii=False))
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    engine = Engine.from_pretrained(args.model)
    result = cache_speedup(engine, args.prompt, args.tokens)
    cached, naive = result["cached"], result["naive"]
    print(f"model            : {engine.config.name} ({engine.n_params:,} params)")
    print(f"tokens           : {cached['tokens']}")
    print(f"with KV cache    : {cached['tokens_per_sec']:.1f} tok/s ({cached['seconds']:.3f}s)")
    print(f"no cache (recompute): {naive['tokens_per_sec']:.1f} tok/s ({naive['seconds']:.3f}s)")
    print(f"cache speedup    : {result['speedup']:.2f}x")
    memory = cache_memory_report(engine, engine.config.max_seq_len)
    print(f"KV cache @ max_len: {memory['total_mb']:.2f} MB "
          f"({memory['bytes_per_token']} bytes/token)")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    from .hf import convert_hf_checkpoint

    def progress(filename: str, received: int, total: int) -> None:
        if total:
            sys.stdout.write(
                f"\r  downloading {filename}: {received / 1e6:7.1f} / {total / 1e6:.1f} MB "
                f"({100 * received / total:3.0f}%)"
            )
        else:
            sys.stdout.write(f"\r  downloading {filename}: {received / 1e6:.1f} MB")
        sys.stdout.flush()

    print(f"converting {args.repo} -> {args.out}")
    try:
        engine = convert_hf_checkpoint(
            args.repo, args.out, name=args.name, local=args.local, progress=progress
        )
    except urllib.error.URLError as exc:
        print()
        print(f"network error while fetching from huggingface.co: {exc}", file=sys.stderr)
        print(
            "hint: pre-download config.json + model.safetensors + vocab.json + merges.txt, "
            "then run again with --local DIR",
            file=sys.stderr,
        )
        return 1
    print()
    print(f"done : {engine.n_params:,} params, vocab {engine.config.vocab_size}, "
          f"tokenizer {'yes' if engine.tokenizer else 'no'}")
    print(f"saved: {args.out}")
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from .chat import run_chat

    engine = Engine.from_pretrained(args.model)
    params = SamplingParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    return run_chat(engine, params, template=args.template)


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    if args.model:
        engine = Engine.from_pretrained(args.model)
    elif args.preset:
        engine = Engine.demo(args.preset)
    else:
        print("provide --model DIR or --preset NAME", file=sys.stderr)
        return 2
    serve(engine, host=args.host, port=args.port, open_browser=args.open)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    dispatch = {
        "generate": _cmd_generate,
        "tokenize": _cmd_tokenize,
        "inspect": _cmd_inspect,
        "bench": _cmd_bench,
        "convert": _cmd_convert,
        "chat": _cmd_chat,
        "serve": _cmd_serve,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())