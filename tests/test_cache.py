import unittest

import numpy as np

from nimbus_infer.cache import KVCache


class TestKVCache(unittest.TestCase):
    def test_write_cursor_and_readback(self):
        cache = KVCache(n_layers=2, n_kv_heads=3, head_dim=4, max_seq_len=8)
        k = np.ones((3, 2, 4), dtype=np.float32)
        v = np.full((3, 2, 4), 2.0, dtype=np.float32)
        cache.append(0, k, v)
        cache.advance(2)
        self.assertEqual(cache.seq_len, 2)
        self.assertTrue(np.allclose(cache.keys(0, 2), 1.0))
        self.assertTrue(np.allclose(cache.values(0, 2), 2.0))

    def test_overflow_raises(self):
        cache = KVCache(n_layers=1, n_kv_heads=1, head_dim=2, max_seq_len=2)
        cache.append(0, np.zeros((1, 2, 2), dtype=np.float32), np.zeros((1, 2, 2), dtype=np.float32))
        cache.advance(2)
        with self.assertRaises(ValueError):
            cache.append(0, np.zeros((1, 1, 2), dtype=np.float32), np.zeros((1, 1, 2), dtype=np.float32))

    def test_reset(self):
        cache = KVCache(n_layers=1, n_kv_heads=1, head_dim=2, max_seq_len=4)
        cache.append(0, np.zeros((1, 3, 2), dtype=np.float32), np.zeros((1, 3, 2), dtype=np.float32))
        cache.advance(3)
        cache.reset()
        self.assertEqual(cache.seq_len, 0)

    def test_memory_accounting(self):
        cache = KVCache(n_layers=4, n_kv_heads=2, head_dim=8, max_seq_len=16)
        expected = 2 * 4 * 2 * 16 * 8 * 4  # 2 (k&v) * layers * heads * seq * dim * 4 bytes
        self.assertEqual(cache.bytes, expected)
        cache.append(0, np.zeros((2, 5, 8), dtype=np.float32), np.zeros((2, 5, 8), dtype=np.float32))
        cache.advance(5)
        self.assertEqual(cache.used_bytes, 2 * 4 * 2 * 5 * 8 * 4)


if __name__ == "__main__":
    unittest.main()