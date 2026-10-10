import json
import os
import tempfile
import unittest

import numpy as np

from nimbus_infer.engine import Engine
from nimbus_infer.hf import config_from_hf, convert_hf_checkpoint, is_llama_config
from nimbus_infer.sampling import SamplingParams
from nimbus_infer.tokenizer import BPETokenizer
from nimbus_infer.weights import save_safetensors


class TestHFConfigMapping(unittest.TestCase):
    def test_gpt2_config(self):
        cfg = config_from_hf({
            "model_type": "gpt2", "vocab_size": 100, "n_embd": 32, "n_layer": 2,
            "n_head": 4, "n_positions": 64, "layer_norm_epsilon": 1e-5,
            "tie_word_embeddings": True,
        })
        self.assertEqual((cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_mlp), (2, 4, 32, 128))
        self.assertEqual(cfg.norm, "layernorm")
        self.assertEqual(cfg.pos_encoding, "learned")
        self.assertFalse(is_llama_config({"model_type": "gpt2"}))

    def test_llama_config(self):
        cfg = config_from_hf({
            "model_type": "llama", "vocab_size": 100, "hidden_size": 32,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 1,
            "intermediate_size": 64, "max_position_embeddings": 128, "rms_norm_eps": 1e-6,
        })
        self.assertEqual(cfg.norm, "rmsnorm")
        self.assertEqual(cfg.pos_encoding, "rope")
        self.assertEqual(cfg.resolved_n_kv_heads, 1)
        self.assertEqual(cfg.kv_group_size, 4)
        self.assertTrue(is_llama_config({"model_type": "llama"}))


def _write_synthetic_gpt2_hf_dir(directory: str) -> int:
    """Materialise a tiny HF-layout GPT-2 checkpoint so `convert` can be tested offline.

    Returns the vocabulary size so the caller can assert against it.
    """
    d, heads, layers, mlp, positions = 8, 2, 1, 16, 16
    rng = np.random.default_rng(0)

    # Train the tokenizer first and size the model to exactly its vocabulary.
    tokenizer = BPETokenizer.train("the engine caches keys and values. " * 20, vocab_size=280)
    vocab_size = tokenizer.vocab_size

    def r(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    tensors = {
        "transformer.wte.weight": r(vocab_size, d),
        "transformer.wpe.weight": r(positions, d),
        "transformer.ln_f.weight": np.ones(d, np.float32),
        "transformer.ln_f.bias": np.zeros(d, np.float32),
        "lm_head.weight": r(vocab_size, d),
    }
    p = "transformer.h.0."
    tensors[p + "ln_1.weight"] = np.ones(d, np.float32)
    tensors[p + "ln_1.bias"] = np.zeros(d, np.float32)
    tensors[p + "ln_2.weight"] = np.ones(d, np.float32)
    tensors[p + "ln_2.bias"] = np.zeros(d, np.float32)
    tensors[p + "attn.c_attn.weight"] = r(d, 3 * d)
    tensors[p + "attn.c_attn.bias"] = np.zeros(3 * d, np.float32)
    tensors[p + "attn.c_proj.weight"] = r(d, d)
    tensors[p + "attn.c_proj.bias"] = np.zeros(d, np.float32)
    tensors[p + "mlp.c_fc.weight"] = r(d, mlp)
    tensors[p + "mlp.c_fc.bias"] = np.zeros(mlp, np.float32)
    tensors[p + "mlp.c_proj.weight"] = r(mlp, d)
    tensors[p + "mlp.c_proj.bias"] = np.zeros(d, np.float32)
    save_safetensors(os.path.join(directory, "model.safetensors"), tensors)

    with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "model_type": "gpt2", "vocab_size": vocab_size, "n_embd": d, "n_layer": layers,
            "n_head": heads, "n_inner": mlp, "n_positions": positions,
            "layer_norm_epsilon": 1e-5, "tie_word_embeddings": True,
        }, fh)

    tokenizer.save(directory)
    return vocab_size


class TestHFConversionOffline(unittest.TestCase):
    def test_convert_local_directory_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "hf")
            out = os.path.join(tmp, "nimbus")
            os.makedirs(src)
            vocab_size = _write_synthetic_gpt2_hf_dir(src)

            engine = convert_hf_checkpoint(src, out, local=True, progress=None)

            self.assertEqual(engine.config.n_layers, 1)
            self.assertEqual(engine.config.d_model, 8)
            self.assertEqual(engine.config.vocab_size, vocab_size)
            self.assertIsNotNone(engine.tokenizer)

            reloaded = Engine.from_pretrained(out)
            self.assertTrue(os.path.exists(os.path.join(out, "weights.safetensors")))
            # The converted checkpoint must actually generate.
            text = reloaded.generate(
                "the engine", SamplingParams(max_new_tokens=4, temperature=0.0)
            )
            self.assertIsInstance(text, str)


if __name__ == "__main__":
    unittest.main()