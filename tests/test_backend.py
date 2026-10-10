import unittest

import numpy as np

from nimbus_infer import backend as B


class TestBackend(unittest.TestCase):
    def test_softmax_is_a_distribution(self):
        x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
        probs = B.softmax(x, axis=-1)
        self.assertTrue(np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6))
        self.assertTrue(np.all(probs >= 0))

    def test_softmax_is_shift_invariant(self):
        x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        self.assertTrue(np.allclose(B.softmax(x), B.softmax(x + 1000.0), atol=1e-6))

    def test_gelu_and_silu_reference_values(self):
        self.assertAlmostEqual(float(B.gelu(np.array([0.0]))[0]), 0.0, places=6)
        self.assertAlmostEqual(float(B.silu(np.array([0.0]))[0]), 0.0, places=6)
        # silu(x) = x * sigmoid(x) -> at x=1 it is 0.7310586
        self.assertAlmostEqual(float(B.silu(np.array([1.0]))[0]), 0.7310586, places=5)

    def test_layer_norm_normalises(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        out = B.layer_norm(x, w, b)
        self.assertAlmostEqual(float(out.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(out.var()), 1.0, places=4)

    def test_rms_norm_matches_definition(self):
        x = np.array([[3.0, 4.0]], dtype=np.float32)
        w = np.ones(2, dtype=np.float32)
        out = B.rms_norm(x, w, eps=0.0)
        rms = np.sqrt((9 + 16) / 2)
        self.assertTrue(np.allclose(out, x / rms, atol=1e-6))

    def test_causal_mask_blocks_the_future(self):
        mask = B.causal_mask(3, 3, offset=0)
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1])
        self.assertTrue(mask[2, 2])

    def test_causal_mask_with_offset_sees_full_cache(self):
        mask = B.causal_mask(1, 5, offset=4)
        self.assertTrue(mask.all())  # a query at position 4 can see all 5 keys

    def test_linear_with_bias(self):
        x = np.ones((2, 3), dtype=np.float32)
        w = np.ones((3, 2), dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        out = B.linear(x, w, b)
        self.assertTrue(np.allclose(out, [[4.0, 5.0], [4.0, 5.0]]))


if __name__ == "__main__":
    unittest.main()