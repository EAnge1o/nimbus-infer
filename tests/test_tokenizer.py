import os
import tempfile
import unittest

from nimbus_infer.tokenizer import BPETokenizer, IncrementalDecoder

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox is quick and the dog is lazy. "
    "nimbus infer is a from scratch inference engine built with numpy. "
) * 20


class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = BPETokenizer.train(CORPUS, vocab_size=400)

    def test_vocab_size_bounded_by_corpus(self):
        # A byte-level trainer cannot invent more merges than the corpus allows:
        # once every word is a single token there are no adjacent pairs left.
        self.assertGreaterEqual(self.tok.vocab_size, 256)
        self.assertLessEqual(self.tok.vocab_size, 400)

    def test_richer_corpus_yields_larger_vocab(self):
        small = BPETokenizer.train(CORPUS, vocab_size=400)
        big = BPETokenizer.train(CORPUS + " ".join(f"word{i}" for i in range(300)), vocab_size=900)
        self.assertGreater(big.vocab_size, small.vocab_size)

    def test_roundtrip_ascii(self):
        text = "the quick brown fox jumps over the lazy dog"
        ids = self.tok.encode(text)
        self.assertEqual(self.tok.decode(ids), text)

    def test_roundtrip_unicode(self):
        text = "nimbus 推理引擎 — héllo, 世界! 🚀"
        ids = self.tok.encode(text)
        self.assertEqual(self.tok.decode(ids), text)

    def test_roundtrip_empty(self):
        self.assertEqual(self.tok.encode(""), [])

    def test_merges_reduce_token_count(self):
        text = "the quick brown fox"
        # Trained merges must compress at least as well as raw bytes.
        self.assertLess(len(self.tok.encode(text)), len(text.encode("utf-8")))

    def test_no_unk_token_needed(self):
        # Byte-level BPE covers every possible input.
        ids = self.tok.encode("\x00\x01\x02 arbitrary \xff bytes")
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.tok.save(tmp)
            self.assertTrue(os.path.exists(os.path.join(tmp, "vocab.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "merges.txt")))
            reloaded = BPETokenizer.load(tmp)
        self.assertEqual(reloaded.vocab_size, self.tok.vocab_size)
        text = "nimbus infer engine"
        self.assertEqual(reloaded.encode(text), self.tok.encode(text))
        self.assertEqual(reloaded.decode(reloaded.encode(text)), text)

    def test_deterministic_training(self):
        again = BPETokenizer.train(CORPUS, vocab_size=400)
        self.assertEqual(again.encoder, self.tok.encoder)

    def test_token_strings_are_human_readable(self):
        ids = self.tok.encode("the")
        pieces = self.tok.token_strings(ids)
        self.assertEqual("".join(pieces), "the")

    def test_token_bytes_roundtrip(self):
        ids = self.tok.encode("hello")
        joined = b"".join(self.tok.token_bytes(i) for i in ids)
        self.assertEqual(joined, "hello".encode("utf-8"))

    def test_incremental_decoder_never_splits_a_character(self):
        # "你" = E4 BD A0, delivered as three separate byte-level tokens.
        decoder = IncrementalDecoder(self.tok)
        out = decoder.push(0xE4) + decoder.push(0xBD) + decoder.push(0xA0)
        out += decoder.flush()
        self.assertEqual(out, "你")

    def test_incremental_decoder_matches_single_shot_decode(self):
        text = "nimbus 引擎 héllo 世界"
        ids = self.tok.encode(text)
        # Feed the ids one at a time; the joined output must be identical.
        decoder = IncrementalDecoder(self.tok)
        streamed = "".join(decoder.push(i) for i in ids) + decoder.flush()
        self.assertEqual(streamed, self.tok.decode(ids))
        self.assertEqual(streamed, text)


if __name__ == "__main__":
    unittest.main()