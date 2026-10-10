import os
import tempfile
import unittest

import numpy as np

from nimbus_infer.config import ModelConfig
from nimbus_infer.weights import (
    bfloat16_to_float32,
    convert_hf_gpt2,
    convert_hf_llama,
    load_npz,
    read_safetensors,
    save_npz,
    save_safetensors,
)


class TestSafetensors(unittest.TestCase):
    def test_roundtrip_multiple_dtypes(self):
        tensors = {
            "f32": np.random.default_rng(0).standard_normal((4, 5)).astype(np.float32),
            "f16": np.ones((3,), dtype=np.float16),
            "i32": np.arange(6, dtype=np.int32).reshape(2, 3),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.safetensors")
            save_safetensors(path, tensors, metadata={"format": "nimbus"})
            loaded = read_safetensors(path)
        self.assertEqual(set(loaded), set(tensors))
        for key in tensors:
            self.assertEqual(loaded[key].shape, tensors[key].shape)
            self.assertTrue(np.allclose(loaded[key], tensors[key]))

    def test_bfloat16_widening_roundtrip(self):
        original = np.array([1.0, -2.5, 0.25, 100.0], dtype=np.float32)
        u32 = original.view(np.uint32)
        bf16 = (u32 >> 16).astype(np.uint16)
        restored = bfloat16_to_float32(bf16)
        # bfloat16 keeps ~8 bits of mantissa.
        self.assertTrue(np.allclose(restored, original, rtol=1e-2))

    def test_npz_roundtrip(self):
        params = {"a": np.ones((2, 2), dtype=np.float32), "b": np.zeros(3, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "w.npz")
            save_npz(path, params)
            loaded = load_npz(path)
        self.assertTrue(np.array_equal(loaded["a"], params["a"]))


class TestHFConversion(unittest.TestCase):
    def test_gpt2_split_of_combined_qkv(self):
        cfg = ModelConfig(vocab_size=16, n_layers=1, n_heads=2, d_model=8, d_mlp=16,
                          max_seq_len=8, norm="layernorm", activation="gelu",
                          pos_encoding="learned", tie_embeddings=True)
        d = 8
        tensors = {
            "transformer.wte.weight": np.zeros((16, d), np.float32),
            "transformer.wpe.weight": np.zeros((8, d), np.float32),
            "transformer.ln_f.weight": np.ones(d, np.float32),
            "transformer.ln_f.bias": np.zeros(d, np.float32),
        }
        for i in range(1):
            p = f"transformer.h.{i}."
            tensors[p + "ln_1.weight"] = np.ones(d, np.float32)
            tensors[p + "ln_1.bias"] = np.zeros(d, np.float32)
            tensors[p + "ln_2.weight"] = np.ones(d, np.float32)
            tensors[p + "ln_2.bias"] = np.zeros(d, np.float32)
            tensors[p + "attn.c_attn.weight"] = np.arange(d * 3 * d, dtype=np.float32).reshape(d, 3 * d)
            tensors[p + "attn.c_attn.bias"] = np.arange(3 * d, dtype=np.float32)
            tensors[p + "attn.c_proj.weight"] = np.zeros((d, d), np.float32)
            tensors[p + "attn.c_proj.bias"] = np.zeros(d, np.float32)
            tensors[p + "mlp.c_fc.weight"] = np.zeros((d, 16), np.float32)
            tensors[p + "mlp.c_fc.bias"] = np.zeros(16, np.float32)
            tensors[p + "mlp.c_proj.weight"] = np.zeros((16, d), np.float32)
            tensors[p + "mlp.c_proj.bias"] = np.zeros(d, np.float32)

        params = convert_hf_gpt2(tensors, cfg)
        self.assertEqual(params["h.0.attn.q.weight"].shape, (d, d))
        self.assertEqual(params["h.0.attn.k.weight"].shape, (d, d))
        # The combined bias must be split in thirds.
        self.assertTrue(np.allclose(params["h.0.attn.q.bias"], np.arange(d)))
        self.assertTrue(np.allclose(params["h.0.attn.v.bias"], np.arange(2 * d, 3 * d)))

    def test_llama_projection_transpose(self):
        cfg = ModelConfig(vocab_size=16, n_layers=1, n_heads=2, n_kv_heads=1, d_model=8,
                          d_mlp=16, head_dim=4, max_seq_len=8, norm="rmsnorm",
                          activation="swiglu", pos_encoding="rope", tie_embeddings=False)
        tensors = {
            "model.embed_tokens.weight": np.zeros((16, 8), np.float32),
            "model.norm.weight": np.ones(8, np.float32),
            "lm_head.weight": np.zeros((16, 8), np.float32),
            "model.layers.0.input_layernorm.weight": np.ones(8, np.float32),
            "model.layers.0.post_attention_layernorm.weight": np.ones(8, np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.arange(8 * 8, dtype=np.float32).reshape(8, 8),
            "model.layers.0.self_attn.k_proj.weight": np.arange(4 * 8, dtype=np.float32).reshape(4, 8),
            "model.layers.0.self_attn.v_proj.weight": np.arange(4 * 8, dtype=np.float32).reshape(4, 8),
            "model.layers.0.self_attn.o_proj.weight": np.arange(8 * 8, dtype=np.float32).reshape(8, 8),
            "model.layers.0.mlp.gate_proj.weight": np.zeros((16, 8), np.float32),
            "model.layers.0.mlp.up_proj.weight": np.zeros((16, 8), np.float32),
            "model.layers.0.mlp.down_proj.weight": np.zeros((8, 16), np.float32),
        }
        params = convert_hf_llama(tensors, cfg)
        self.assertEqual(params["h.0.attn.q.weight"].shape, (8, 8))
        self.assertEqual(params["h.0.attn.k.weight"].shape, (8, 4))
        self.assertEqual(params["h.0.mlp.down.weight"].shape, (16, 8))
        # Transpose verification: (out,in) source becomes (in,out).
        self.assertTrue(
            np.allclose(params["h.0.attn.q.weight"], tensors["model.layers.0.self_attn.q_proj.weight"].T)
        )


if __name__ == "__main__":
    unittest.main()