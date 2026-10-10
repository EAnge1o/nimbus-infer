import os
import tempfile
import unittest

import numpy as np

from nimbus_infer.config import tiny
from nimbus_infer.engine import Engine
from nimbus_infer.sampling import SamplingParams
from nimbus_infer.tokenizer import BPETokenizer

CORPUS = "the nimbus inference engine runs on the cpu with numpy. " * 30


def _engine_with_tokenizer() -> Engine:
    cfg = tiny()
    tok = BPETokenizer.train(CORPUS, vocab_size=320)
    cfg.vocab_size = tok.vocab_size
    cfg.__post_init__()
    return Engine.random(cfg, tokenizer=tok, seed=11)


class TestEngine(unittest.TestCase):
    def test_generate_ids_length(self):
        engine = _engine_with_tokenizer()
        params = SamplingParams(max_new_tokens=8, temperature=0.0)
        ids = engine.generate_ids("the nimbus", params)
        self.assertEqual(len(ids), 8)

    def test_greedy_is_deterministic(self):
        engine = _engine_with_tokenizer()
        params = SamplingParams(max_new_tokens=10, temperature=0.0)
        self.assertEqual(engine.generate_ids("the engine", params),
                         engine.generate_ids("the engine", params))

    def test_cache_and_no_cache_agree_under_greedy(self):
        engine = _engine_with_tokenizer()
        params = SamplingParams(max_new_tokens=6, temperature=0.0)
        with_cache = engine.generate_ids("nimbus infer", params, use_cache=True)
        no_cache = engine.generate_ids("nimbus infer", params, use_cache=False)
        self.assertEqual(with_cache, no_cache)

    def test_max_new_tokens_clipped_by_window(self):
        engine = _engine_with_tokenizer()
        params = SamplingParams(max_new_tokens=10_000, temperature=0.0)
        ids = engine.generate_ids("the", params)
        self.assertLessEqual(len(ids), engine.config.max_seq_len - 1)

    def test_stop_token_halts_generation(self):
        engine = _engine_with_tokenizer()
        first = engine.generate_ids("the nimbus", SamplingParams(max_new_tokens=1, temperature=0.0))[0]
        params = SamplingParams(max_new_tokens=20, temperature=0.0, stop_tokens=(first,))
        self.assertEqual(engine.generate_ids("the nimbus", params), [first])

    def test_stream_concatenates_to_generate(self):
        engine = _engine_with_tokenizer()
        params = SamplingParams(max_new_tokens=12, temperature=0.0)
        streamed = "".join(engine.stream("the nimbus", params))
        self.assertEqual(streamed, engine.generate("the nimbus", params))

    def test_save_and_load_pretrained(self):
        engine = _engine_with_tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            engine.save_pretrained(tmp)
            self.assertTrue(os.path.exists(os.path.join(tmp, "config.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "weights.safetensors")))
            loaded = Engine.from_pretrained(tmp)
        self.assertEqual(loaded.config.vocab_size, engine.config.vocab_size)
        self.assertEqual(loaded.n_params, engine.n_params)
        for key, value in engine.params.items():
            self.assertTrue(np.allclose(loaded.params[key], value), msg=key)
        self.assertIsNotNone(loaded.tokenizer)
        self.assertEqual(loaded.tokenizer.encode("nimbus"), engine.tokenizer.encode("nimbus"))

    def test_info_report(self):
        engine = _engine_with_tokenizer()
        info = engine.info()
        self.assertEqual(info["parameters"], engine.n_params)
        self.assertGreater(info["kv_cache_bytes"], 0)
        self.assertEqual(info["architecture"]["kv_heads"], engine.config.resolved_n_kv_heads)

    def test_empty_prompt_rejected(self):
        engine = _engine_with_tokenizer()
        with self.assertRaises(ValueError):
            engine.generate_ids("", SamplingParams(max_new_tokens=1))

    def test_iter_token_ids_matches_generate_ids(self):
        engine = _engine_with_tokenizer()
        params = SamplingParams(max_new_tokens=5, temperature=0.0)
        self.assertEqual(
            list(engine.iter_token_ids("the engine", params)),
            engine.generate_ids("the engine", params),
        )

    def test_generate_into_reuses_cache_across_turns(self):
        engine = _engine_with_tokenizer()
        cache = engine.model.new_cache()
        params = SamplingParams(max_new_tokens=3, temperature=0.0)
        first = list(engine.generate_into(cache, engine.encode("the engine"), params))
        after_first = cache.seq_len
        self.assertGreater(after_first, len(first))
        second = list(engine.generate_into(cache, engine.encode(" nimbus"), params))
        self.assertEqual(len(second), 3)
        # The cache kept growing instead of being reset between turns.
        self.assertGreater(cache.seq_len, after_first)

    def test_demo_engine_is_self_contained(self):
        engine = Engine.demo("tiny", seed=0)
        self.assertIsNotNone(engine.tokenizer)
        text = engine.generate("the engine", SamplingParams(max_new_tokens=4, temperature=0.0))
        self.assertIsInstance(text, str)
        self.assertTrue(engine.info()["random_weights"])

    def test_stream_is_equivalent_when_a_character_spans_tokens(self):
        engine = _engine_with_tokenizer()
        ids = [0xE4, 0xBD, 0xA0]  # "你" split across three byte-level tokens
        engine.iter_token_ids = lambda *a, **k: iter(ids)
        streamed = "".join(engine.stream("x"))
        self.assertEqual(streamed, engine.tokenizer.decode(ids))
        self.assertEqual(streamed, "你")


if __name__ == "__main__":
    unittest.main()