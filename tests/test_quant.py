import unittest

import numpy as np

from nimbus_infer.quant import (
    compression_ratio,
    dequantize,
    dequantize_params,
    effective_bits,
    is_quantized,
    quantize,
    quantize_params,
    size_bytes,
)


class TestQuantization(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.weight = (rng.standard_normal((64, 32)) * 0.1).astype(np.float32)

    def test_int8_error_within_half_scale(self):
        packed = quantize(self.weight, bits=8)
        restored = dequantize(packed)
        err = np.abs(self.weight - restored)
        bound = float(packed["scale"].max()) / 2 + 1e-5
        self.assertLessEqual(float(err.max()), bound)

    def test_int4_error_within_half_scale(self):
        packed = quantize(self.weight, bits=4)
        restored = dequantize(packed)
        err = np.abs(self.weight - restored)
        bound = float(packed["scale"].max()) / 2 + 1e-4
        self.assertLessEqual(float(err.max()), bound)

    def test_shape_preserved(self):
        for bits in (4, 8):
            packed = quantize(self.weight, bits=bits)
            self.assertEqual(dequantize(packed).shape, self.weight.shape)

    def test_group_quantization(self):
        packed = quantize(self.weight, bits=8, group_size=16)
        self.assertEqual(packed["group_size"], 16)
        self.assertEqual(dequantize(packed).shape, self.weight.shape)

    def test_group_size_must_divide(self):
        with self.assertRaises(ValueError):
            quantize(self.weight, bits=8, group_size=7)

    def test_invalid_bits(self):
        with self.assertRaises(ValueError):
            quantize(self.weight, bits=3)

    def test_compression_ratio_int8(self):
        packed = quantize(self.weight, bits=8)
        ratio = compression_ratio({"w": self.weight}, {"w": packed})
        self.assertGreater(ratio, 3.0)

    def test_int4_is_actually_packed(self):
        int8 = quantize(self.weight, bits=8)
        int4 = quantize(self.weight, bits=4)
        # int4 must occupy roughly half the storage of int8, not the same.
        self.assertLess(size_bytes({"w": int4}), size_bytes({"w": int8}) * 0.6)
        self.assertAlmostEqual(effective_bits({"w": int4}), 4.0, delta=0.2)
        # ...and it must still dequantize to the original shape.
        self.assertEqual(dequantize(int4).shape, self.weight.shape)

    def test_int4_group_quantization_dequantizes(self):
        packed = quantize(self.weight, bits=4, group_size=16)
        restored = dequantize(packed)
        self.assertEqual(restored.shape, self.weight.shape)
        err = float(np.abs(self.weight - restored).max())
        self.assertLess(err, float(packed["scale"].max()) / 2 + 1e-3)

    def test_param_dict_only_touches_matrices(self):
        params = {
            "wte": self.weight,
            "norm": np.ones(8, dtype=np.float32),
        }
        packed = quantize_params(params, bits=8)
        self.assertTrue(is_quantized(packed["wte"]))
        self.assertFalse(is_quantized(packed["norm"]))
        restored = dequantize_params(packed)
        self.assertLess(float(np.abs(restored["wte"] - params["wte"]).max()), 0.05)
        self.assertTrue(np.array_equal(restored["norm"], params["norm"]))

    def test_size_bytes_reflects_quantization(self):
        packed = quantize(self.weight, bits=8)
        self.assertLess(size_bytes({"w": packed}), size_bytes({"w": self.weight}))


if __name__ == "__main__":
    unittest.main()