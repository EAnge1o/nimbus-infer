"""High-level inference engine: checkpoint I/O and the generation loop.

The generation loop implements the two phases every LLM runtime has:

* **prefill** -- run the whole prompt once, filling the KV cache.
* **decode**  -- run one token at a time, reusing the cache.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Iterator, Optional, Sequence

import numpy as np

from .model import Transformer, init_params
from .config import ModelConfig
from .sampling import SamplingParams, sample
from .tokenizer import BPETokenizer, IncrementalDecoder
from .weights import load_weights, save_npz, save_safetensors

# A tiny built-in corpus so ``Engine.demo()`` can train a tokenizer on the fly
# and give the CLI / tests a fully working engine without any downloads.
_DEMO_CORPUS = (
    "nimbus infer is a from scratch inference engine written in pure numpy. "
    "the engine tokenizes text with a byte level byte pair encoding vocabulary. "
    "attention is causal and keys and values are cached between decode steps. "
    "grouped query attention reduces the cache while rotary embeddings encode position. "
    "the history of computing began with mechanical calculators and punch cards. "
) * 30


class Engine:
    def __init__(
        self,
        config: ModelConfig,
        params: dict[str, np.ndarray],
        tokenizer: Optional[BPETokenizer] = None,
    ) -> None:
        self.config = config
        self.params = params
        self.tokenizer = tokenizer
        self.model = Transformer(config, params)
        # Set by ``Engine.random`` / ``Engine.demo`` so the UI can say up front
        # that the output is meaningless on purpose.
        self.random_weights = False

    # -- construction ----------------------------------------------------
    @classmethod
    def from_config_params(
        cls, config: ModelConfig, params: dict[str, np.ndarray], tokenizer=None
    ) -> "Engine":
        return cls(config, params, tokenizer)

    @classmethod
    def random(cls, config: ModelConfig, tokenizer=None, seed: int = 0) -> "Engine":
        engine = cls(config, init_params(config, seed=seed), tokenizer)
        engine.random_weights = True
        return engine

    @classmethod
    def demo(cls, preset: str = "tiny", seed: int = 42, vocab_size: int = 512) -> "Engine":
        """A self-contained engine: trains a BPE tokenizer on a built-in corpus.

        Used by the CLI (``serve --preset``) and tests so they need no downloads.
        Weights are random, so generated text is not meaningful -- only the
        mechanics are.
        """
        from .config import get_preset

        tokenizer = BPETokenizer.train(_DEMO_CORPUS, vocab_size=vocab_size)
        config = get_preset(preset)
        config.vocab_size = tokenizer.vocab_size
        config.__post_init__()
        return cls.random(config, tokenizer=tokenizer, seed=seed)

    @classmethod
    def from_pretrained(cls, path: str) -> "Engine":
        """Load a nimbus checkpoint directory (config.json + weights + tokenizer)."""
        if os.path.isdir(path):
            return cls._load_dir(path)
        # A bare weight file next to a sibling config.json.
        config_path = os.path.join(os.path.dirname(path) or ".", "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"no config.json found next to {path}; pass a checkpoint directory"
            )
        with open(config_path, encoding="utf-8") as fh:
            config = ModelConfig.from_dict(json.load(fh))
        return cls(config, load_weights(path))

    @classmethod
    def _load_dir(cls, directory: str) -> "Engine":
        config_path = os.path.join(directory, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{config_path} not found")
        with open(config_path, encoding="utf-8") as fh:
            config = ModelConfig.from_dict(json.load(fh))

        params = None
        for candidate in ("weights.safetensors", "weights.npz", "model.safetensors"):
            full = os.path.join(directory, candidate)
            if os.path.exists(full):
                params = load_weights(full)
                break
        if params is None:
            raise FileNotFoundError(f"no weights file found in {directory}")

        tokenizer = None
        tok_dir = os.path.join(directory, "tokenizer")
        if os.path.isdir(tok_dir):
            tokenizer = BPETokenizer.load(tok_dir)
        return cls(config, params, tokenizer)

    def save_pretrained(self, directory: str, use_safetensors: bool = True) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as fh:
            fh.write(self.config.to_json())
        if use_safetensors:
            save_safetensors(os.path.join(directory, "weights.safetensors"), self.params)
        else:
            save_npz(os.path.join(directory, "weights.npz"), self.params)
        if self.tokenizer is not None:
            self.tokenizer.save(os.path.join(directory, "tokenizer"))

    # -- introspection ---------------------------------------------------
    @property
    def n_params(self) -> int:
        return self.model.n_params()

    def info(self) -> dict:
        return {
            "name": self.config.name,
            "architecture": {
                "norm": self.config.norm,
                "activation": self.config.activation,
                "pos_encoding": self.config.pos_encoding,
                "layers": self.config.n_layers,
                "heads": self.config.n_heads,
                "kv_heads": self.config.resolved_n_kv_heads,
                "d_model": self.config.d_model,
                "head_dim": self.config.resolved_head_dim,
                "max_seq_len": self.config.max_seq_len,
            },
            "vocab_size": self.config.vocab_size,
            "parameters": self.n_params,
            "kv_cache_bytes": self.model.new_cache().bytes,
            "tokenizer": None if self.tokenizer is None else self.tokenizer.vocab_size,
            "random_weights": self.random_weights,
        }

    # -- generation ------------------------------------------------------
    def encode(self, prompt: str | Sequence[int]) -> list[int]:
        """Public helper: text (or ids) -> token ids."""
        return self._resolve_prompt_ids(prompt)

    def _resolve_prompt_ids(self, prompt: str | Sequence[int]) -> list[int]:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("a tokenizer is required to encode a text prompt")
            return self.tokenizer.encode(prompt)
        return [int(t) for t in prompt]

    def iter_token_ids(
        self,
        prompt: str | Sequence[int],
        params: Optional[SamplingParams] = None,
        use_cache: bool = True,
    ) -> Iterator[int]:
        """Yield generated token ids one at a time (prefill once, then decode).

        This is the single source of truth for the generation loop; ``generate_ids``
        and :meth:`stream` are thin wrappers over it, and the CLI/web demos drive it
        directly so they can report live metrics between tokens.
        """
        ids = self._resolve_prompt_ids(prompt)
        if not ids:
            raise ValueError("prompt produced an empty token sequence")
        cache = self.model.new_cache() if use_cache else None
        yield from self._generate(cache, ids, params or SamplingParams())

    def generate_into(
        self,
        cache,
        prompt: str | Sequence[int],
        params: Optional[SamplingParams] = None,
    ) -> Iterator[int]:
        """Prefill new ids into an *existing* cache and decode.

        Used by the REPL to reuse the KV cache across conversation turns: only the
        new user input is prefilled, the previous turns are already cached.
        """
        ids = self._resolve_prompt_ids(prompt)
        if not ids:
            raise ValueError("prompt produced an empty token sequence")
        yield from self._generate(cache, ids, params or SamplingParams())

    def _generate(self, cache, ids: list[int], params: SamplingParams) -> Iterator[int]:
        base = cache.pos if cache is not None else 0
        budget = self.config.max_seq_len - base - len(ids)
        n_new = min(params.max_new_tokens, max(budget, 0))
        rng = np.random.default_rng(params.seed)

        turn = list(ids)  # context used for repetition penalty / no-cache recompute
        logits = self.model.forward(ids, cache)
        if cache is not None:
            cache.advance(len(ids))
        last = logits[-1]

        for step in range(n_new):
            token = sample(last, params, generated=turn, rng=rng)
            turn.append(token)
            yield token

            if token in params.stop_tokens or step == n_new - 1:
                return

            if cache is not None:
                logits = self.model.forward([token], cache)
                cache.advance(1)
            else:
                logits = self.model.forward(turn, None)
            last = logits[-1]

    def generate_ids(
        self,
        prompt: str | Sequence[int],
        params: Optional[SamplingParams] = None,
        on_token: Optional[Callable[[int, int], None]] = None,
        use_cache: bool = True,
    ) -> list[int]:
        produced: list[int] = []
        for index, token in enumerate(self.iter_token_ids(prompt, params, use_cache)):
            produced.append(token)
            if on_token is not None:
                on_token(token, index)
        return produced

    def generate(self, prompt: str | Sequence[int], params: Optional[SamplingParams] = None) -> str:
        ids = self.generate_ids(prompt, params)
        if self.tokenizer is None:
            return " ".join(str(i) for i in ids)
        return self.tokenizer.decode(ids)

    def stream(
        self, prompt: str | Sequence[int], params: Optional[SamplingParams] = None
    ) -> Iterator[str]:
        """Yield text deltas as tokens are produced.

        Decoding is incremental (see :class:`IncrementalDecoder`), so a character
        split across two byte-level tokens is emitted exactly once and only when
        complete -- ``"".join(stream(...))`` equals the single-shot decode.
        """
        if self.tokenizer is None:
            for token in self.iter_token_ids(prompt, params):
                yield f"{token} "
            return

        decoder = IncrementalDecoder(self.tokenizer)
        for token in self.iter_token_ids(prompt, params):
            text = decoder.push(token)
            if text:
                yield text
        tail = decoder.flush()
        if tail:
            yield tail