"""Byte-level BPE tokenizer, implemented from scratch.

Two capabilities that normally require the ``tokenizers`` / ``transformers``
packages are reimplemented here:

* **Training** a byte-pair-encoding vocabulary on a raw corpus.
* **Loading** a HuggingFace-compatible ``vocab.json`` + ``merges.txt`` pair.

The tokenizer is fully byte-level (every input round-trips losslessly through
UTF-8 bytes), so no ``<unk>`` token is ever needed.
"""
from __future__ import annotations

import codecs
import json
import os
from collections import Counter
from typing import Iterable, Optional

# ``regex`` (if installed) gives real Unicode property classes; otherwise we
# fall back to an ASCII approximation using the standard library only.
try:  # pragma: no cover - depends on optional package
    import regex as _re

    _PATTERN = (
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
        r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )
    _HAS_REGEX = True
except ImportError:  # pragma: no cover - default path
    import re as _re

    _PATTERN = (
        r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+"
        r"| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
    )
    _HAS_REGEX = False

_COMPILED = _re.compile(_PATTERN)


def bytes_to_unicode() -> dict[int, str]:
    """Reversible map from the 256 byte values to printable unicode chars.

    This is the mapping introduced by GPT-2 so that arbitrary bytes can be
    embedded in a text vocabulary without control characters.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_ENCODER = bytes_to_unicode()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}


def _merge_sequence(seq: tuple, pair: tuple) -> tuple:
    """Replace every non-overlapping occurrence of ``pair`` in ``seq``."""
    first, second = pair
    out = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == first and seq[i + 1] == second:
            out.append(first + second)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    """A byte-level BPE tokenizer.

    Tokens are stored as strings of *byte-level unicode characters*
    (see :func:`bytes_to_unicode`). Merges are ranked; encoding greedily applies
    the lowest-ranked merge first, exactly like GPT-2 / LLaMA tokenizers.
    """

    def __init__(
        self,
        vocab: Optional[dict[str, int]] = None,
        merges: Optional[dict[tuple[str, str], int]] = None,
        pattern: str = _PATTERN,
        special_tokens: Optional[dict[str, int]] = None,
    ) -> None:
        self.encoder: dict[str, int] = dict(vocab or {})
        self.decoder: dict[int, str] = {i: t for t, i in self.encoder.items()}
        self.merges: dict[tuple[str, str], int] = dict(merges or {})
        self.pattern = pattern
        self._compiled = _re.compile(pattern)
        self.special_tokens: dict[str, int] = dict(special_tokens or {})
        for tok, idx in self.special_tokens.items():
            self.encoder[tok] = idx
            self.decoder[idx] = tok

    # -- properties ------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def __len__(self) -> int:
        return self.vocab_size

    # -- core BPE --------------------------------------------------------
    @staticmethod
    def _get_pairs(word: list[str]) -> set[tuple[str, str]]:
        return set(zip(word[:-1], word[1:]))

    def bpe(self, token: str) -> list[str]:
        """Apply ranked merges to one byte-level string."""
        word = list(token)
        if len(word) < 2:
            return word
        while len(word) > 1:
            pairs = self._get_pairs(word)
            bigram = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if bigram not in self.merges:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def _encode_chunk(self, chunk: str) -> list[str]:
        if not chunk:
            return []
        encoded = "".join(_BYTE_ENCODER[b] for b in chunk.encode("utf-8"))
        return self.bpe(encoded)

    # -- public API ------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self._compiled.findall(text):
            for tok in self._encode_chunk(chunk):
                ids.append(self.encoder[tok])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        pieces = []
        for i in ids:
            tok = self.decoder.get(int(i))
            if tok is None:
                continue
            pieces.append(tok)
        text = "".join(pieces)
        data = bytes(_BYTE_DECODER.get(ch, ord("?")) for ch in text)
        return data.decode("utf-8", errors="replace")

    def token_strings(self, ids: Iterable[int]) -> list[str]:
        """Human-readable token pieces (byte-level chars decoded to text)."""
        out = []
        for i in ids:
            tok = self.decoder.get(int(i), "")
            data = bytes(_BYTE_DECODER.get(ch, ord("?")) for ch in tok)
            out.append(data.decode("utf-8", errors="replace"))
        return out

    def token_bytes(self, token_id: int) -> bytes:
        """Raw bytes represented by a single token id.

        A byte-level token is a (possibly partial) UTF-8 sequence, which is why
        decoding a token stream must be done incrementally -- see
        :class:`IncrementalDecoder`.
        """
        token = self.decoder.get(int(token_id))
        if token is None:
            return b""
        return bytes(_BYTE_DECODER.get(ch, ord("?")) for ch in token)

    # -- training --------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int = 512, verbose: bool = False) -> "BPETokenizer":
        """Train a byte-level BPE vocabulary directly on ``text``."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (one id per byte)")

        word_counts: dict[tuple[str, ...], int] = {}
        for chunk in _COMPILED.findall(text):
            seq = tuple(_BYTE_ENCODER[b] for b in chunk.encode("utf-8"))
            if seq:
                word_counts[seq] = word_counts.get(seq, 0) + 1

        merges: list[tuple[str, str]] = []
        for step in range(vocab_size - 256):
            pair_counts: Counter = Counter()
            for seq, count in word_counts.items():
                for pair in zip(seq[:-1], seq[1:]):
                    pair_counts[pair] += count
            if not pair_counts:
                break
            # Deterministic tie-break: most frequent, then lexicographically smallest.
            best = min(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            merges.append(best)
            word_counts = _recount(word_counts, best)
            if verbose and (step + 1) % 100 == 0:
                print(f"  merge {step + 1}/{vocab_size - 256}: {best}")

        tok = cls()
        tok._install_base_vocab()
        tok._install_merges(merges)
        return tok

    def _install_base_vocab(self) -> None:
        self.encoder = {}
        self.decoder = {}
        for b in range(256):
            ch = _BYTE_ENCODER[b]
            self.encoder[ch] = b
            self.decoder[b] = ch

    def _install_merges(self, merges: list[tuple[str, str]]) -> None:
        self.merges = {}
        for rank, (a, b) in enumerate(merges):
            token = a + b
            idx = 256 + rank
            self.merges[(a, b)] = rank
            self.encoder[token] = idx
            self.decoder[idx] = token

    # -- persistence -----------------------------------------------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "vocab.json"), "w", encoding="utf-8") as fh:
            json.dump(self.encoder, fh, ensure_ascii=False)
        with open(os.path.join(directory, "merges.txt"), "w", encoding="utf-8") as fh:
            fh.write("#version: 0.2\n")
            # merges are ordered by rank which equals insertion order
            ordered = sorted(self.merges.items(), key=lambda kv: kv[1])
            for (a, b), _ in ordered:
                fh.write(f"{a} {b}\n")

    @classmethod
    def load(cls, directory: str) -> "BPETokenizer":
        vocab_path = os.path.join(directory, "vocab.json")
        merges_path = os.path.join(directory, "merges.txt")
        encoder: dict[str, int] = {}
        if os.path.exists(vocab_path):
            with open(vocab_path, encoding="utf-8") as fh:
                encoder = json.load(fh)
        merges: dict[tuple[str, str], int] = {}
        if os.path.exists(merges_path):
            with open(merges_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line or line.startswith("#version"):
                        continue
                    parts = line.split(" ")
                    if len(parts) != 2:
                        continue
                    merges[(parts[0], parts[1])] = len(merges)
        tok = cls(vocab=encoder, merges=merges)
        if not encoder:
            # Reconstruct a byte-level vocab purely from the merges.
            tok._install_base_vocab()
            for (a, b), rank in sorted(merges.items(), key=lambda kv: kv[1]):
                tok.encoder[a + b] = 256 + rank
                tok.decoder[256 + rank] = a + b
        return tok


class IncrementalDecoder:
    """Decode a token stream into text without ever splitting a character.

    A byte-level token may be only *part* of a UTF-8 sequence, so naively
    decoding each prefix and slicing off the previous length silently drops
    characters (e.g. ``你`` arrives as three tokens and disappears). This wraps
    the standard library's incremental UTF-8 decoder, which buffers a truncated
    trailing sequence until the next token completes it.

    The result is that ``"".join(stream(...))`` is byte-identical to a
    single-shot ``decode(...)``.
    """

    def __init__(self, tokenizer: "BPETokenizer", errors: str = "replace") -> None:
        self.tokenizer = tokenizer
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors=errors)

    def push(self, token_id: int) -> str:
        """Feed one token id; returns whatever text is now complete (may be empty)."""
        return self._decoder.decode(self.tokenizer.token_bytes(token_id), final=False)

    def flush(self) -> str:
        """Finalise the stream, surfacing any genuinely invalid trailing bytes."""
        return self._decoder.decode(b"", final=True)


def _recount(
    word_counts: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int]:
    out: dict[tuple[str, ...], int] = {}
    for seq, count in word_counts.items():
        merged = _merge_sequence(seq, pair)
        out[merged] = out.get(merged, 0) + count
    return out