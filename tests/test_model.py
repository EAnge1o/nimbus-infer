import unittest

import numpy as np

from nimbus_infer.config import ModelConfig, gpt2_124m, llama_style, tiny
from nimbus_infer.model import Transformer, build_rope, init_params


class TestModelConfig(unittest.TestCase):
    def test_resolved_layout_defaults(self):
        cfg = tiny()
        self.assertEqual(cfg.resolved_n_kv_heads, cfg.n_heads)
        self.assertEqual(cfg.resolved_head_dim, cfg.d_model // cfg.n_heads)
        self.assertEqual(cfg.kv_group_size, 1)

    def test_gqa_group_size(self):
        cfg = llama_style()
        self.assertEqual(cfg.kv_group_size, 4)

    def test_invalid_flags_rejected(self):
        with self.assertRaises(ValueError):
            ModelConfig(norm="batchnorm")
        with self.assertRaises(ValueError):
            ModelConfig(activation="relu")
        with self.assertRaises(ValueError):
            ModelConfig(pos_encoding="sinusoidal")
        with self.assertRaises(ValueError):
            ModelConfig(n_heads=5, d_model=64)  # not divisible, no head_dim

    def test_json_roundtrip(self):
        cfg = llama_style()
        self.assertEqual(ModelConfig.from_json(cfg.to_json()), cfg)

    def test_analytic_param_count_matches_materialised(self):
        for cfg in (tiny(), llama_style(vocab_size=256), gpt2_124m()):
            params = init_params(cfg, seed=0)
            materialised = sum(int(v.size) for v in params.values())
            self.assertEqual(cfg.param_count(), materialised, msg=cfg.name)


class TestTransformer(unittest.TestCase):
    def test_forward_shapes_gpt2(self):
        cfg = tiny()
        model = Transformer(cfg, init_params(cfg, seed=1))
        logits = model.forward([1, 2, 3, 4])
        self.assertEqual(logits.shape, (4, cfg.vocab_size))

    def test_forward_shapes_llama(self):
        cfg = llama_style(vocab_size=128)
        model = Transformer(cfg, init_params(cfg, seed=1))
        logits = model.forward(np.array([1, 2, 3]))
        self.assertEqual(logits.shape, (3, 128))

    def test_attention_is_causal(self):
        cfg = tiny()
        model = Transformer(cfg, init_params(cfg, seed=2))
        base = model.forward([1, 2, 3, 4])
        changed = model.forward([1, 2, 3, 99])  # only the last token differs
        self.assertTrue(np.allclose(base[:3], changed[:3], atol=1e-5))
        self.assertFalse(np.allclose(base[3], changed[3]))

    def test_kv_cache_matches_full_recompute(self):
        cfg = llama_style(vocab_size=64)
        model = Transformer(cfg, init_params(cfg, seed=3))
        ids = list(range(1, 12))
        full = model.forward(ids)

        cache = model.new_cache()
        prefill = model.forward(ids[:5], cache)
        cache.advance(5)
        decode = model.forward(ids[5:], cache)
        cache.advance(len(ids) - 5)

        self.assertTrue(np.allclose(prefill, full[:5], atol=1e-4))
        self.assertTrue(np.allclose(decode, full[5:], atol=1e-4))

    def test_rope_tables_shape(self):
        cos, sin = build_rope(head_dim=32, max_seq_len=16, theta=10000.0)
        self.assertEqual(cos.shape, (16, 16))
        self.assertEqual(sin.shape, (16, 16))
        self.assertTrue(np.allclose(cos[0], 1.0) and np.allclose(sin[0], 0.0))

    def test_max_seq_len_enforced(self):
        cfg = tiny()
        model = Transformer(cfg, init_params(cfg, seed=0))
        with self.assertRaises(ValueError):
            model.forward(list(range(cfg.max_seq_len + 1)))

    def test_determinism(self):
        cfg = tiny()
        a = Transformer(cfg, init_params(cfg, seed=5)).forward([1, 2, 3])
        b = Transformer(cfg, init_params(cfg, seed=5)).forward([1, 2, 3])
        self.assertTrue(np.array_equal(a, b))

    def test_attention_capture_shape_and_normalisation(self):
        cfg = tiny()
        model = Transformer(cfg, init_params(cfg, seed=1))
        ids = [1, 2, 3, 4, 5]
        model.forward(ids, capture_attention=True)
        self.assertEqual(len(model.captured_attention), cfg.n_layers)
        for matrix in model.captured_attention:
            self.assertEqual(matrix.shape, (5, 5))
            self.assertTrue(np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5))

    def test_captured_attention_is_causal(self):
        cfg = tiny()
        model = Transformer(cfg, init_params(cfg, seed=1))
        model.forward([1, 2, 3, 4], capture_attention=True)
        row0 = model.captured_attention[0][0]
        # The first query can only attend to the first key.
        self.assertAlmostEqual(float(row0[0]), 1.0, places=5)
        self.assertTrue(np.allclose(row0[1:], 0.0, atol=1e-6))

    def test_attention_not_captured_by_default(self):
        cfg = tiny()
        model = Transformer(cfg, init_params(cfg, seed=1))
        model.forward([1, 2, 3])
        self.assertEqual(model.captured_attention, [])


if __name__ == "__main__":
    unittest.main()